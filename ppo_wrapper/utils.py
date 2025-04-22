import torch
import torch.nn as nn

# Load the model checkpoint
model_path = "/home/lee/ros2_ws/src/ppo_wrapper/model/hunter_hybrid.pth"  # Change this to your actual file path
checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

# print checkpoint keys
print(checkpoint.keys())

# Extract model state dictionary
model_state_dict = checkpoint["model"]

# Extract layer names and their shapes
layer_info = {name: param.shape for name, param in model_state_dict.items()}

# Print basic model information
print("=== Model Checkpoint Analysis ===")
print(f"Keys in Checkpoint: {checkpoint.keys()}")
print(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
print(f"Loss: {checkpoint.get('loss', 'Unknown')}")
print("\n=== Model Layer Structure ===")
for layer, shape in layer_info.items():
    print(f"{layer}: {shape}")


# # Instantiate the model and load weights
# policy_model = RLPolicyMLP()
# policy_model.load_state_dict(model_state_dict)
# policy_model.eval()  # Set to inference mode

# # Print the full model architecture
# print("\n=== Model Architecture ===")
# print(policy_model)


# Define the A2C network architecture
class A2CNetwork(nn.Module):
    def __init__(self, input_dim=7, hidden_sizes=[512, 256, 128], output_dim=2):
        super(A2CNetwork, self).__init__()

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

# Load the checkpoint (ensure weights_only=False to load the full checkpoint)
# model_path = "hunter_hybrid.pth"  # Adjust the path as needed
# checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

# Extract the model state dictionary from the checkpoint
state_dict = checkpoint["model"]

# Strip the "a2c_network." prefix from keys
stripped_state_dict = strip_prefix(state_dict, prefix="a2c_network.")

# Filter only the keys expected by the A2CNetwork
expected_keys = {
    "sigma",
    "actor_mlp.0.weight", "actor_mlp.0.bias",
    "actor_mlp.2.weight", "actor_mlp.2.bias",
    "actor_mlp.4.weight", "actor_mlp.4.bias",
    "mu.weight", "mu.bias",
    "value.weight", "value.bias"
}
filtered_state_dict = {k: v for k, v in stripped_state_dict.items() if k in expected_keys}

# Instantiate the model and load the filtered state dictionary
model = A2CNetwork()
model.load_state_dict(filtered_state_dict, strict=False)
model.eval()  # Set to evaluation mode

# Print the loaded model architecture
print("=== Loaded A2C Model Architecture ===")
print(model)
# Test the loaded model with a dummy input
sample_input = torch.randn(1, 7)  # A sample observation with 7 features
mu, sigma, value = model(sample_input)
print("Policy Mean (mu):", mu)
print("Policy Std Dev (sigma):", sigma)
print("State Value:", value)

# Test the loaded model with a dummy input
sample_input = torch.randn(1, 7)  # A sample observation with 7 features
mu, sigma, value = model(sample_input)
print("Policy Mean (mu):", mu)
print("Policy Std Dev (sigma):", sigma)
print("State Value:", value)

