import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from utils.networks import DGNController
from torch.optim import Adam
from utils.misc import soft_update, hard_update
from utils.agents import DDPGAgent
import numpy as np
from torch.distributions import Categorical

MSELoss = torch.nn.MSELoss()

class DGN(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """

    def __init__(self, alg_types,
                 dim_features, dim_acts,
                 gamma=0.99, tau=0.01, lr_crit=0.01,
                 hiddens = [100,100,100,70,25],
                 reg=0.01, temp=0.3, epsilon=1.0):

        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.hiddens = hiddens
        self.gamma = gamma
        self.tau = tau

        self.lr_crit = lr_crit
        self.reg = reg
        self.temp = temp
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0

        self.dim_features = dim_features
        self.dim_acts = dim_acts
        self.critic_net = DGNController(dim_features, dim_acts-1, hiddens, tau=temp)
        self.target_critic_net = DGNController(dim_features, dim_acts-1, hiddens, tau=temp)
        self.critic_optimizer = Adam(self.critic_net.parameters(), lr=lr_crit)

        hard_update(self.target_critic_net, self.critic_net)

        self.init_epsilon = epsilon
        self.epsilon = epsilon

    def scale_noise(self, scale):
        self.epsilon = scale

    def reset_noise(self):
        self.epsilon = self.init_epsilon

    def indiv_step(self, obs, explore=False):
        num_nodes = torch.sum(obs[:,:,-1] != -1, axis= -1)
        filtered_obs = obs[obs[:,:,-1] != -1]

        qval, _ = self.critic_net(filtered_obs, num_nodes)
        q_val = qval.detach().numpy()
        all_actions = (self.dim_acts - 1) * np.ones([obs.shape[0], self.nagents])
        greedy_acts = np.argmax(q_val, axis=-1)

        if explore:
            for idx in range(greedy_acts):
                if np.random.uniform() < self.epsilon:
                    greedy_acts[idx] = np.random.randint(self.dim_acts-1)

        all_actions[obs[:, :, -1] != -1] = greedy_acts
        return [acts[0].astype(int) for acts in all_actions]

    def step(self, obs, explore=False):
        num_nodes = torch.sum(obs[:, :, -1] != -1, dim=-1)
        filtered_obs = obs[obs[:, :, -1] != -1]

        qval, _ = self.critic_net(filtered_obs, num_nodes)
        q_val = qval.detach().numpy()
        all_actions = (self.dim_acts - 1) * np.ones([obs.shape[0], self.nagents])
        greedy_acts = np.argmax(q_val, axis=-1)

        if explore:
            for idx, _ in enumerate(greedy_acts):
                if np.random.uniform() < self.epsilon:
                    greedy_acts[idx] = np.random.randint(self.dim_acts - 1)

        all_actions[obs[:, :, -1] != -1] = greedy_acts
        return [acts.astype(int)for acts in all_actions]

    def update(self, sample, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        obs, acs, rews, next_obs, dones = sample

        valid_flags = obs[:,:,-1] != -1
        num_nodes = valid_flags.sum(dim=-1).tolist()
        valid_obs = obs[valid_flags]
        valid_rews = rews[valid_flags].view(-1,1)
        valid_dones = rews[valid_flags].view(-1,1)

        self.critic_optimizer.zero_grad()
        predicted_values, attention_weights = self.critic_net(valid_obs, num_nodes)
        _, attention_weights_next = self.critic_net(next_obs[valid_flags], num_nodes)
        target_vals, _ = self.target_critic_net(next_obs[valid_flags], num_nodes)
        pred = predicted_values.gather(-1,(acs[acs[:,:] != self.dim_acts-1]).view(-1,1).long())

        target_values = valid_rews + self.gamma * (1 - valid_dones) * target_vals.max(1)[0].view(-1,1)
        qloss = MSELoss(pred, target_values.detach())
        p, q = Categorical(probs=attention_weights), Categorical(probs=attention_weights_next)
        KLLoss = (p.probs * (p.logits - q.logits)).sum(axis=-1).mean()
        loss = qloss + self.reg * KLLoss
        loss.backward()

        self.critic_optimizer.step()

        if logger is not None:
            logger.add_scalars('losses',
                               {'q_loss': qloss,
                               'kl_loss': KLLoss},
                               self.niter)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.target_critic_net, self.critic_net, self.tau)
        self.niter += 1

    def prep_training(self, device='gpu'):
        self.critic_net.train()
        self.target_critic_net.train()

        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()

        if not self.critic_dev == device:
            self.critic_net = fn(self.target_critic_net)
            self.critic_dev = device

        if not self.trgt_critic_dev == device:
            self.target_critic_net = fn(self.target_critic_net)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        self.critic_net.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.critic_dev == device:
            self.critic_net = fn(self.critic_net)
            self.critic_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict, 'critic_params': self.critic_net.state_dict()}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, alg="DGN", gamma=0.99, tau=0.01, lr_crit=0.01, reg=0.01, temp=0.3):
        """
        Instantiate instance of this class from multi-agent environment
        """
        alg_types = [alg for
                     _ in range(len(env.observation_space))]
        dim_features = env.observation_space[0].shape[0]
        dim_acts = env.action_space[0].n
        init_dict = {'gamma': gamma, 'tau': tau, 'lr_crit': lr_crit,
                     'dim_features': dim_features,
                     'dim_acts': dim_acts,
                     'alg_types': alg_types,
                     'reg': reg, 'temp': temp}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        return instance
