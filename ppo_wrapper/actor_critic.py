import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, input_dim=7, hidden_sizes=[512, 256, 128], output_dim=2):
        super(ActorCritic, self).__init__()

        # Actor Network (Policy)
        self.actor_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
        )

        # Policy Mean Output (mu)
        self.mu = nn.Linear(hidden_sizes[2], output_dim)

        # Policy Variance Output (sigma) - Learnable log std
        self.sigma = nn.Parameter(torch.zeros(output_dim))

        # Value Network (Critic)
        self.value = nn.Linear(hidden_sizes[2], 1)

    def forward(self, x):
        features = self.actor_mlp(x)
        mu = self.mu(features)  # Policy mean
        sigma = torch.exp(self.sigma)  # Convert log std to std
        value = self.value(features)  # State value estimation
        return mu, sigma, value  # Return action mean, std, and value estimate

# Utility function to strip a prefix from state_dict keys
def strip_prefix(state_dict, prefix="a2c_network."):
    new_state = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            new_state[new_key] = value
        else:
            # Skip keys not starting with the prefix (e.g., normalization stats)
            pass
    return new_state
