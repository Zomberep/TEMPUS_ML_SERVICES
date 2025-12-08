import torch.nn as nn

class AdvancedMultiLabelMLP(nn.Module):
    def __init__(self, input_dim, num_labels, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            ResidualBlock(hidden_dim),   # ← добавляем остаточные блоки
            ResidualBlock(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(x + self.block(x))