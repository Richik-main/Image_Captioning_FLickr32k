import torch
from torch import nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim):
        super(ProjectionHead, self).__init__()
        self.iniprojection = nn.Linear(embedding_dim, 1024)
        self.relu = nn.ReLU()
        self.projection = nn.Linear(1024, projection_dim)

    def forward(self, x):
        x = self.iniprojection(x)
        x = self.relu(x)
        x = self.projection(x)
        return F.normalize(x, p=2, dim=-1)
