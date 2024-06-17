import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pylab as plt

from models import bottle
from modules.dtw import DTW


class CEM(nn.Module):
    def __init__(self, 
                 num_iterations, 
                 population_size, 
                 elite_fraction, 
                 action_size, 
                 min_std= 0.05, 
                 temperature=0.5,
                 momentum=0.1):
        self.num_iterations = num_iterations
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.action_size = action_size
        self.min_std = min_std
        self.temperature = temperature
        self.momentum = momentum


    @torch.no_grad()
    def train(self, 
              initial_beliefs, 
              initial_states, 
              observations, 
              rewards,
              trajectory_length, 
              dynamics_model, 
              observation_model,
              save_output=False):
        means, stds = torch.zeros(trajectory_length, 1, self.action_size, device=initial_beliefs.device), \
            torch.ones(trajectory_length, 1, self.action_size, device=initial_beliefs.device)

        observations = observations.unsqueeze(0).expand(self.population_size, -1, -1, -1, -1)

        for iteration in range(self.num_iterations):
            beliefs, states = (
                [torch.empty(0)] * (trajectory_length+1),
                [torch.empty(0)] * (trajectory_length+1)
            )

            # sample N trajectories of length ${trajectory_length}
            beliefs[0], states[0] = initial_beliefs.repeat(self.population_size, 1),\
                                      initial_states.repeat(self.population_size, 1)
            actions = torch.clamp(
                means + stds *
                torch.randn(trajectory_length,
                            self.population_size, 
                            self.action_size,
                            device=initial_beliefs.device),
                -1, 1)
            
            for i_traj in range(trajectory_length):
                b, s, _, _ = dynamics_model(
                    states[i_traj],
                    actions[[i_traj]],
                    beliefs[i_traj],
                )
                beliefs[i_traj+1], states[i_traj+1] = b.squeeze(0), s.squeeze(0)

            beliefs, states = torch.stack(beliefs[1:], dim=1), torch.stack(states[1:], dim=1)
            
            reconst_observations = bottle(observation_model, (beliefs, states))

            # c_ = torch.pow(0.9, torch.arange(trajectory_length, device=initial_beliefs.device))
            dists, directions = DTW(observations, reconst_observations)

            elite_idxs = torch.topk(dists, int(self.population_size*self.elite_fraction), largest=False).indices

            elite_diffs, elite_actions = dists[elite_idxs], actions[:, elite_idxs]
            
            min_diff = elite_diffs.min(0)[0]
            score = torch.exp(self.temperature * (min_diff - elite_diffs))
            score /= score.sum(0)

            _mean = torch.sum(
                score.view(1, -1, 1) * elite_actions, 
                dim=1, keepdim=True
                ) / (score.sum(0) + 1e-9)
            _std = torch.sqrt(
                torch.sum(
                    score.view(1, -1, 1) * (elite_actions - _mean) ** 2, 
                    dim=1, keepdim=True
                    ) / (score.sum(0) + 1e-9))

            _std = _std.clamp_(self.min_std, 1)
            means, stds = self.momentum * means + (1 - self.momentum) * _mean, _std

        # # Outputs
        # score = score.squeeze(1).cpu().numpy()
        # actions = elite_actions[:, np.random.choice(
        #     np.arange(score.shape[0]), p=score)]
        # self._prev_mean = mean
        # mean, std = actions[0], _std[0]
        # a = mean
        # if not eval_mode:
        #     a += std * torch.randn(self.cfg.action_dim, device=std.device)
        # return a

        best_i = torch.topk(dists, 1, largest=False).indices[0]
        trans_rewards = CEM.calc_reward(directions[best_i], rewards)

        if(save_output):
            obs = np.clip(observations[best_i.item()].cpu().permute(0,2,3,1).numpy() + 0.5, 0, 1)
            r_obs = np.clip(reconst_observations[best_i.item()].cpu().permute(0,2,3,1).numpy() + 0.5, 0, 1)

            fig, ax = plt.subplots(2, max(obs.shape[0], r_obs.shape[0]), figsize=(16, 4))
            for i in range(obs.shape[0]):
                ax[0,i].imshow(obs[i])
                ax[0,i].axis("off")
                ax[1,i].axis("off")
            for i in range(r_obs.shape[0]):
                ax[1,i].imshow(r_obs[i])
                ax[0,i].axis("off")
                ax[1,i].axis("off")
            fig.savefig('./results/N-'+ str(time.time()) + ".png")
            plt.close()

        return beliefs[best_i], states[best_i], trans_rewards
    
    @staticmethod
    def calc_reward(routes, rewards):
        coefs = torch.zeros_like(routes, device=rewards.device, requires_grad=False)
        pos = [routes.shape[0]-1, routes.shape[1]-1]
        path = []
        
        while(pos[0] >= 0 and pos[1] >= 0):
            path.insert(0, routes[pos[0], pos[1]].item())
            coefs[pos[0], pos[1]] = 1

            if routes[pos[0], pos[1]] == 0:
                pos[0] -= 1
            elif routes[pos[0], pos[1]] == 1:
                pos[1] -= 1
            elif routes[pos[0], pos[1]] == 2:
                pos[0] -= 1
                pos[1] -= 1
            else:
                raise Exception("Illegal value in chosen route")
            
        coefs = coefs / coefs.sum(dim=0, keepdim=True)
        trans_rewards = (coefs * rewards.unsqueeze(0)).sum(1)
            
        return trans_rewards
    