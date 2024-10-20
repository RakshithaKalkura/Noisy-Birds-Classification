import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Convolutional layers with fewer filters
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # 16 filters
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # 32 filters
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 64 filters

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Adjusted based on output size after conv layers
        self.fc2 = nn.Linear(128, 4)  # Output for 4 classes

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)

    ###########DO NOT CHANGE THIS PART##################
    def init(self):
        self.load_state_dict(torch.load("model.pth", weights_only=True))
    ####################################################
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # Pooling layer

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        
        return x
