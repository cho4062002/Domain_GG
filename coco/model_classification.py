import torch
import torch.nn as nn

class Classification(nn.Module):
    def __init__(self, embed_dim=1536, n_classes=80):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, 1300)
        self.fc2 = nn.Linear(1300, 1600)
        self.fc3 = nn.Linear(1600, 512)
        self.fc4 = nn.Linear(512 , 256)
        self.fc5 = nn.Linear(256 , n_classes)
        

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x