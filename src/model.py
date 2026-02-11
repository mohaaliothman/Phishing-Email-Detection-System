import torch
import torch.nn as nn

class PhishingDetector(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.drop3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.drop4 = nn.Dropout(0.2)

        self.out = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.drop1(self.relu(self.bn1(self.fc1(x))))
        x = self.drop2(self.relu(self.bn2(self.fc2(x))))
        x = self.drop3(self.relu(self.bn3(self.fc3(x))))
        x = self.drop4(self.relu(self.bn4(self.fc4(x))))
        logits = self.out(x)  
        return logits
