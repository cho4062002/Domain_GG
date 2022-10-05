import torch
import torch.nn as nn
import torch.nn.functional as F

class Classification(nn.Module):
    def __init__(self, embed_dim=768, n_classes=1000):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, n_classes)
       
        

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x