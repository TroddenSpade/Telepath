import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models import bottle


@torch.no_grad()
def DTW(observations, reconstructed_observations):

    dim_1, dim_2 = observations.size(1), reconstructed_observations.size(1)
    shape = observations.shape
    observations = observations\
                    .unsqueeze(1).expand(*shape[0:1], dim_2, *shape[1:])
    shape = reconstructed_observations.shape
    reconstructed_observations = reconstructed_observations\
                    .unsqueeze(2).expand(*shape[0:2], dim_1, *shape[2:])
    
    obs_diffs = F.mse_loss(
        reconstructed_observations,
        observations,
        reduction='none'
    ).sum(dim=(3,4,5))

    diff_matrix = torch.ones(obs_diffs.shape[0], 
                             obs_diffs.shape[1]+1, 
                             obs_diffs.shape[2]+1, 
                             device=observations.device) * float('inf')
    
    diff_matrix[:,0,0] = 0
    directions = torch.zeros_like(obs_diffs)
    for i in range(obs_diffs.shape[1]):
        for j in range(obs_diffs.shape[2]):
            min_ = torch.min(torch.stack([
                        diff_matrix[:,i,j+1],
                        diff_matrix[:,i+1,j],
                        diff_matrix[:,i,j]], dim=0), dim=0)
            directions[:,i,j] = min_.indices
            diff_matrix[:, i+1, j+1] = min_.values + obs_diffs[:, i, j]

    return  diff_matrix[:, -1, -1], directions
