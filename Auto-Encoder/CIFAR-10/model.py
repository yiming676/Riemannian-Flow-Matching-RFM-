import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- 全局 ----------
LATENT = 128

# ---------- 网络 ----------
class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_out)
        )
        self.skip = nn.Conv2d(c_in, c_out, 1, bias=False) if c_in != c_out else nn.Identity()
    def forward(self, x):
        return F.relu(self.skip(x) + self.net(x))

class Encoder(nn.Module):
    def __init__(self, latent=LATENT):
        super().__init__()
        self.net = nn.Sequential(
            ResidualBlock(3, 64),   nn.AvgPool2d(2),    # 16
            ResidualBlock(64, 128), nn.AvgPool2d(2),    # 8
            ResidualBlock(128, 256), nn.AvgPool2d(2),   # 4
            ResidualBlock(256, 512), nn.AvgPool2d(2),   # 2
            nn.Flatten(),
            nn.Linear(512*2*2, 512), nn.ReLU(inplace=True),
            nn.Linear(512, latent)
        )
    def forward(self, x):
        # 注意：在推理时，我们仍然需要归一化以确保向量在单位球面上
        return F.normalize(self.net(x), dim=1)

class Decoder(nn.Module):
    def __init__(self, latent=LATENT):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512*2*2),
            nn.ReLU(inplace=True)
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), # 2→4
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 4→8
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 8→16
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 16→32
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1),
            nn.Sigmoid()
        )
    def forward(self, z):
        h = self.fc(z).view(-1, 512, 2, 2)
        return self.deconv(h)