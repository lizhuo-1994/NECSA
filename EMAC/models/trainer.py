import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from models.utils import EpisodicReplayBuffer, RcsEpisodicReplayBuffer, RcsReplayBuffer
from models.TD3 import TD3
from models.DDPG import DDPG
from models.EMAC import EMAC
from models.RCS_DDPG import RCS_DDPG
from models.RCS_TD3 import RCS_TD3

from .utils import eval_policy, RewardLogger, estimate_true_q, determine_state_scales
from .mem import MemBuffer


class Trainer:

    def __init__(self, config):
        self.c = config

    def train(self, exp_dir):
        expl_noise = self.c["expl_noise"]
        max_timesteps = self.c["max_timesteps"]
        start_timesteps = self.c["start_timesteps"]
        batch_size = self.c["batch_size"]
        eval_freq = self.c["eval_freq"]
        save_model = self.c["save_model"]
        save_buffer = self.c["save_buffer"]
        save_memory = self.c["save_memory"]
        save_model_every = self.c["save_model_every"]
        device = self.c["device"]
        env_name = self.c["env"]
        env = gym.make(self.c["env"])
        substeps = self.c["substeps"]


        
        # Logger
        tb_logger = SummaryWriter(f"{exp_dir}/tb")
        reward_logger = RewardLogger(self.c["results_dir"] + "_rewards")

        # Set seeds
        seed = self.c["seed"]
        env.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        raw_state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        kwargs = {
            "raw_state_dim": raw_state_dim,
            "action_dim": action_dim,
            "max_action": max_action,
            "discount": self.c["discount"],
            "tau": self.c["tau"],
            "device": self.c["device"],
            "log_dir": f"{exp_dir}/tb"
        }
        print('Initialize policy')
        # Initialize policy
        method = self.c["policy"]
        if method == "TD3":
            # Target policy smoothing is scaled wrt the action scale
            kwargs["policy_noise"] = self.c["policy_noise"] * max_action
            kwargs["noise_clip"] = self.c["noise_clip"] * max_action
            kwargs["policy_freq"] = self.c["policy_freq"]
            del kwargs["raw_state_dim"]
            kwargs["state_dim"] = raw_state_dim
            policy = TD3(**kwargs)
        elif method == "DDPG":
            del kwargs["raw_state_dim"]
            kwargs["state_dim"] = raw_state_dim
            policy = DDPG(**kwargs)
        elif method == "EMAC":
            del kwargs["raw_state_dim"]
            kwargs["state_dim"] = raw_state_dim
            kwargs["alpha"] = self.c["alpha"]
            policy = EMAC(**kwargs)
        elif "RCS" in method:
            kwargs["alpha"] = self.c["alpha"]
            kwargs["order"] = self.c["order"]
            kwargs["grid_num"] = self.c["grid_num"]
            kwargs["decay"] = self.c["decay"]
            kwargs["repair_scope"] = self.c["repair_scope"]

            kwargs["raw_state_dim"] = raw_state_dim
            kwargs["reduction"] = self.c["reduction"]
            if kwargs["reduction"]:
                kwargs["state_dim"] = self.c["state_dim"]
            else:
                kwargs["state_dim"] = raw_state_dim

            kwargs["state_min"] = self.c["state_min"]
            kwargs["state_max"] = self.c["state_max"]
            kwargs["action_min"] = np.min(env.action_space.low)
            kwargs["action_max"] = np.min(env.action_space.high)
            kwargs["mode"] = self.c["mode"]
            
            if method == 'RCS_DDPG':
                del kwargs["alpha"]
                policy = RCS_DDPG(**kwargs)
            elif method == 'RCS_TD3':
                del kwargs["alpha"]
                policy = RCS_TD3(**kwargs)

            ####### configure the state abstraction #############
        
        print('Configured policy')
        load_model = self.c["load_model"]
        if load_model != "":
            policy.load(f"{exp_dir}/models/{load_model}")

        mem = MemBuffer(raw_state_dim, action_dim,
                        capacity=self.c["max_timesteps"],
                        k=self.c["k"],
                        mem_dim=self.c["mem_dim"],
                        device=kwargs["device"])
        
        if 'RCS' in method :
            replay_buffer = RcsEpisodicReplayBuffer(raw_state_dim, action_dim,mem,
                                             device=device,
                                             prioritized=self.c["prioritized"],
                                             beta=self.c["beta"],
                                             start_timesteps=self.c["start_timesteps"],
                                             expl_noise=self.c["expl_noise"])
        
        else:
            replay_buffer = EpisodicReplayBuffer(raw_state_dim, action_dim, mem,
                                             device=device,
                                             prioritized=self.c["prioritized"],
                                             beta=self.c["beta"],
                                             start_timesteps=self.c["start_timesteps"],
                                             expl_noise=self.c["expl_noise"])

        print('Evaluate untrained policy')
        # Evaluate untrained policy
        ep_reward = eval_policy(policy, env_name, seed)
        tb_logger.add_scalar("agent/eval_reward", ep_reward, 0)

        state = env.reset()
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0

        # Evaluate random policy 
        ep_reward = eval_policy(policy, env_name, seed)
        tb_logger.add_scalar("agent/eval_reward", ep_reward, 0)
        reward_logger.log(ep_reward, 0)

        total_reward = 0

        for t in range(1, int(max_timesteps)+1):
            episode_timesteps += 1

            # Select action randomly or according to policy
            if t < start_timesteps:
                action = env.action_space.sample()
            else:
                action = (
                        policy.select_action(np.array(state))
                        + np.random.normal(0, max_action * expl_noise, size=action_dim)
                ).clip(-max_action, max_action)

            # Perform action
            next_state, reward, done_env, _ = env.step(action)
            done_limit = done_env if episode_timesteps < self.c["ep_len"] else True

            replay_buffer.add(state, action, next_state, reward, done_env, done_limit, env, policy, t, max_timesteps)

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if t >= start_timesteps:
                for _ in range(substeps):
                    policy.train(replay_buffer, batch_size)

            if done_limit:
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                # print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                # Reset environment
                total_reward += episode_reward
                print('The average performance is :', episode_reward, ' in round ', t)
                state = env.reset()
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
                

            
            # Evaluate episode
            if t % eval_freq == 0:
                print("Step ", t)
                ep_reward = eval_policy(policy, env_name, seed)
                tb_logger.add_scalar("agent/eval_reward", ep_reward, t)
                reward_logger.log(ep_reward, t)

            # Save model
            if save_model and t % save_model_every == 0:
                print("Saving model...")
                policy.save(f"{exp_dir}/models/model_step_{t}")

            if t % 250000 == 0 and save_buffer:
                print(f"Saving buffer at {t} timestep...")
                replay_buffer.save(f"{exp_dir}/buffers/replay_buffer")

            if t % 250000 == 0 and save_memory:
                print(f"Saving memory at {t} timesteps...")
                replay_buffer.mem.save(f"{exp_dir}/buffers/memory")

            if t % 1000 == 0 and self.c["estimate_q"]:
                print("Calculating true Q")
                true_q = estimate_true_q(policy, self.c["env"], 0.99, replay_buffer)
                tb_logger.add_scalar("q_estimate/true_q", true_q, t)

        print("Dumping reward...")
        env = self.c["env"]
        policy = self.c["policy"]
        exp = self.c["exp_name"]
        seed = self.c["seed"]
        if 'RCS' in policy:
            order = self.c['order']
            fn = f"{env}/{policy}_{order}/{exp}_{seed}.json"
        else:
            fn = f"{env}/{policy}/{exp}_{seed}.json"
        reward_logger.dump(fn)
