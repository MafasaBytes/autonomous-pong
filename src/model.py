import torch.nn as nn
import torch.nn.functional as F

class DQN_Agent(nn.Module):
    def __init__(self, action_size):
        super(DQN_Agent, self).__init__()
        
        # Define layers matching the loaded model
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)  # Assume 4-channel input
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of the fully connected layer input
        self.fc_input_dim = 64 * 7 * 7  # Assuming input size is 84x84 after convolutions

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(-1, self.fc_input_dim)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
