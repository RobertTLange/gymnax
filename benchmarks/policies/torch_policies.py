import torch.nn as nn
import torch


class MLP_Policy(nn.Module):
    def __init__(self, input_dim, output_dim, discrete, hidden_dim=64):
        super(MLP_Policy, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.discrete = discrete

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        # Return raw continuous x if not discrete action space
        if not self.discrete:
            return x
        # Else sample action from multionomial
        else:
            probs = nn.functional.softmax(x, dim=1)
            return torch.multinomial(probs, num_samples=1)
