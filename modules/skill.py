import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMBeliefPrior(nn.Module):
    """LSTM encoder returns tanh normal distribution of latents."""

    def __init__(self, embedding_size, hidden_size, belief_size, activation_function='relu', min_std_dev=0.1):
        super().__init__()

        self.act_fn = getattr(F, activation_function)  # activation function
        self.min_std_dev = min_std_dev  # minimum standard deviation of stochastic states

        self.lstm = nn.LSTM(embedding_size, hidden_size, 1, batch_first=False)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, belief_size),
        )
        self.fc_embd = nn.Linear(belief_size, belief_size)
        self.fc_stoch = nn.Linear(belief_size, belief_size * 2)

    def forward(self, input):
        out, _ = self.lstm(input)
        # encoded = self.fc(out)
        encoded = self.act_fn(self.fc(out))
        # encoded = self.act_fn(self.fc_embd(encoded))

        shape_ = encoded.shape
        # Compute state prior
        prior_means, _prior_std_dev = torch.chunk(
            self.fc_stoch(encoded.flatten(0,1)), 2, dim=1)
        prior_std_devs = F.softplus(_prior_std_dev) + self.min_std_dev
        prior_states = prior_means + prior_std_devs * \
            torch.randn_like(prior_means)

        return prior_means.reshape(*shape_), prior_std_devs.reshape(*shape_), prior_states.reshape(*shape_)
