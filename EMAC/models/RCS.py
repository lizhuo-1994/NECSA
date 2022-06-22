import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from .nn import Actor, Critic

from .abstracter import Abstracter, ScoreInspector

class RCS(object):
    def __init__(self, state_dim, action_dim, max_action, discount=0.99,
        alpha=0.0,tau=0.005, device="cuda", log_dir="tb", order=1, grid_num = 5, 
        decay=0.1, repair_scope=0.25, state_max = 10, state_min = -10, action_min=-1, action_max = 1,
        mode = 'state_action'):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.alpha = alpha
        self.discount = discount
        self.tau = tau
        self.device = device
        self.q = 0

        self.step = 0
        self.tb_logger = SummaryWriter(log_dir)

        self.abstracter = Abstracter(order, decay, repair_scope)
        self.abstracter.inspector = ScoreInspector(
            order, grid_num, state_dim, state_min, state_max, action_dim, action_min, action_max, mode
            )


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100):
        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size, self.step)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        mem_q = replay_buffer.mem.retrieve_cuda(state, action)
        mem_q = torch.from_numpy(mem_q).float().to(self.device)

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        q_loss = F.mse_loss(current_Q, target_Q)
        q_loss_mem = F.mse_loss(current_Q, mem_q)
        critic_loss = (1 - self.alpha) * q_loss + self.alpha * q_loss_mem

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Logging
        if self.step % 250 == 0:
            q = np.mean(current_Q.detach().cpu().numpy())
            self.tb_logger.add_scalar("algo/q", q, self.step)
            q_mem = np.mean(mem_q.cpu().numpy())
            self.tb_logger.add_scalar("algo/q_mem", q_mem, self.step)
            q_loss = q_loss.detach().cpu().item()
            self.tb_logger.add_scalar("algo/q_cur_loss", q_loss, self.step)
            q_mem_loss = q_loss_mem.detach().cpu().item()
            self.tb_logger.add_scalar("algo/q_mem_loss", q_mem_loss, self.step)
            q_total_loss = q_loss + q_mem_loss
            self.tb_logger.add_scalar("algo/critic_loss", q_total_loss, self.step)
            pi_loss = actor_loss.detach().cpu().item()
            self.tb_logger.add_scalar("algo/pi_loss", pi_loss, self.step)
        self.step += 1

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
            
