import torch
import torch.nn as nn

class Classification(nn.Module):
    def __init__(self, embed_dim=1536, n_classes=1000):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, 1300)
        self.fc2 = nn.Linear(1300 , 1200)
        self.fc3 = nn.Linear(1200 , n_classes)
        

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x