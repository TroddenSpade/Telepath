import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models import bottle


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
    def train(self, initial_beliefs, initial_states, observations, trajectory_length, dynamics_model, observation_model, encoder):
        means, stds = torch.zeros(trajectory_length, 1, self.action_size, device=initial_beliefs.device), \
            torch.ones(trajectory_length, 1, self.action_size, device=initial_beliefs.device)
        observations = observations[:trajectory_length].unsqueeze(0).repeat(self.population_size, 1, 1)

        for iteration in tqdm(range(self.num_iterations), leave=False):
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
            encoded_observations = bottle(encoder, (reconst_observations, ))

            obs_diffs = F.mse_loss(
                encoded_observations,
                observations,
                reduction='none'
            ).sum(dim=2).mean(1)

            elite_idxs = torch.topk(obs_diffs, int(self.population_size*self.elite_fraction), largest=False).indices

            elite_diffs, elite_actions = obs_diffs[elite_idxs], actions[:, elite_idxs]
            
            max_diff = elite_diffs.max(0)[0]
            score = torch.exp(self.temperature * (elite_diffs - max_diff))
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

            _std = _std.clamp_(self.min_std, 2)
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
        return means.squeeze(1)
        # return reconst_observations
