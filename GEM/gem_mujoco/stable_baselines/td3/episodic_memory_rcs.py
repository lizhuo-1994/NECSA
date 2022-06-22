from stable_baselines.td3.episodic_memory import EpisodicMemory
import numpy as np
import time
from stable_baselines.td3.abstracter import Abstracter, ScoreInspector


class EpisodicMemoryRCS(EpisodicMemory):
    def __init__(self, buffer_size, state_dim, action_shape, obs_space, q_func, repr_func, obs_ph, action_ph, sess,
                 gamma=0.99,
                 alpha=0.6,max_step=1000,
                 order = 1,
                 grid_num = 10, 
                 decay = 0.2, 
                 state_len = 8,
                 state_min = -6, 
                 state_max = 6,
                 action_dim = 2,
                 action_min = -1,
                 action_max = -1, 
                 mode = 'state_action'):
        self.order = order
        self.grid_num = grid_num
        self.decay = decay
        self.state_len = state_len
        self.state_min = state_min
        self.state_max = state_max
        self.action_dim = action_dim
        self.action_min = action_min
        self.action_max = action_max
        self.mode = mode
        super(EpisodicMemoryRCS, self).__init__(buffer_size, state_dim, action_shape, obs_space, q_func, repr_func,
                                                obs_ph, action_ph, sess,
                                                gamma, alpha,max_step)
        del self._q_values
        self._q_values = -np.inf * np.ones((buffer_size + 1, 2))


        self.abstracter = Abstracter(self.order, self.decay)
        self.abstracter.inspector = ScoreInspector(
            self.order, self.grid_num, self.state_len, self.state_min, self.state_max, self.action_dim, self.action_min, self.action_max, self.mode
            )
        
        # self.max_step = max_step

    def compute_approximate_return_double(self, obses, actions=None):
        return np.array(self.sess.run(self.q_func, feed_dict={self.obs_ph: obses}))

    def update_memory(self, q_base=0, use_knn=False, beta=-1):
        discount_beta = beta ** np.arange(self.max_step)
        trajs = self.retrieve_trajectories()
        for traj in trajs:
            # print(np.array(traj))
            approximate_qs = self.compute_approximate_return_double(self.replay_buffer[traj], self.action_buffer[traj])
            num_q = len(approximate_qs)
            if num_q >= 4:
                approximate_qs = approximate_qs.reshape((2, num_q//2, -1))
                approximate_qs = np.min(approximate_qs, axis=1)  # clip double q

            else:
                assert num_q == 2
                approximate_qs = approximate_qs.reshape(2, -1)
            approximate_qs = np.concatenate([np.zeros((2, 1)), approximate_qs], axis=1)
            self.q_values[traj] = 0

            rtn_1 = np.zeros((len(traj), len(traj)))
            rtn_2 = np.zeros((len(traj), len(traj)))

            for i, s in enumerate(traj):
                rtn_1[i, 0], rtn_2[i, 0] = self.reward_buffer[s] + \
                                           self.gamma * (1 - self.truly_done_buffer[s]) * (
                                                   approximate_qs[:, i] - q_base)
            for i, s in enumerate(traj):
                rtn_1[i, 1:] = self.reward_buffer[s] + self.gamma * rtn_1[i - 1, :-1]
                rtn_2[i, 1:] = self.reward_buffer[s] + self.gamma * rtn_2[i - 1, :-1]

            if beta > 0:

                double_rtn = [
                    [np.dot(rtn_2[i, :min(i + 1, self.max_step)], discount_beta[:min(i + 1, self.max_step)]) / np.sum(
                        discount_beta[:min(i + 1, self.max_step)]),
                     np.dot(rtn_1[i, :min(i + 1, self.max_step)], discount_beta[:min(i + 1, self.max_step)]) / np.sum(
                         discount_beta[:min(i + 1, self.max_step)])]
                    for i in range(len(traj))]
            else:
                double_rtn = [
                    [rtn_2[i, np.argmax(rtn_1[i, :min(i + 1, self.max_step)])],
                     rtn_1[i, np.argmax(rtn_2[i, :min(i + 1, self.max_step)])]] for i
                    in
                    range(len(traj))]
                # double_rtn = [
                #     [rtn_1[i, np.argmax(rtn_1[i, :min(i + 1, self.max_step)])],
                #      rtn_2[i, np.argmax(rtn_2[i, :min(i + 1, self.max_step)])]] for i
                #     in
                #     range(len(traj))]
                # double_rtn = np.min(np.array(double_rtn),axis=1,keepdims=True)
                # double_rtn = np.repeat(double_rtn,2,axis=1)
            # self.q_values[traj] = np.maximum(np.array(double_rtn),np.minimum(rtn_1[:,0],rtn_2[:,0]))
            one_step_q = np.array([rtn_1[:, 0], rtn_2[:, 0]]).transpose()
            self.q_values[traj] = np.maximum(np.array(double_rtn),
                                             # np.min(one_step_q,axis=1,keepdims=True))
                                             one_step_q)
    def update_sequence_with_qs(self, sequence):
        next_id = -1
        Rtd = 0

        state_action_list = []
        reward_list = []
        r_list = []
        new_sequence = []
        for obs, a, z, q_t, r, truly_done, done in sequence:
            self.abstracter.append(list(obs) + list(a), r, done)
            if self.abstracter.inspector.mode == 'state':
                state_action_list.append(list(obs))
            elif self.abstracter.inspector.mode == 'state_action':
                state_action_list.append(list(obs) + list(a))
            reward_list.append(r)

            if done:
                reward_list = self.abstracter.reward_shaping(np.array(state_action_list), np.array(reward_list))
                r_list = r_list + reward_list.tolist()
                state_action_list = []
                reward_list = []
                self.abstracter.inspector.sync_scores()

        for i in range(len(sequence)):
            item_i = list(sequence[i])
            item_i[4] = r_list[i]
            sequence[i] = tuple(item_i)

        for obs, a, z, q_t, r, truly_done, done in reversed(sequence):
            # print(np.mean(z))
            if truly_done:
                Rtd = r
            else:
                Rtd = self.gamma * Rtd + r
            
            current_id = self.add(obs, a, z, Rtd, next_id)

            if done:
                self.end_points.append(current_id)
            self.replay_buffer[current_id] = obs
            self.reward_buffer[current_id] = r
            self.truly_done_buffer[current_id] = truly_done
            self.done_buffer[current_id] = done
            next_id = int(current_id)
        # self.update_priority()
        return
