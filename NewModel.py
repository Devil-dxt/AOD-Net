import torch
import torch.nn as nn
import torch.nn.functional as F



# ---------- 🔸 Position-wise Normalization ----------
class PONO(nn.Module):
    def __init__(self, eps=1e-5):
        super(PONO, self).__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = (x.var(dim=1, keepdim=True) + self.eps).sqrt()
        x = (x - mean) / std
        return x, mean, std


# ---------- 🔸 Feature modulation（用于恢复被归一化掉的信息） ----------
class MS(nn.Module):
    def forward(self, x, beta=None, gamma=None):
        if gamma is not None:
            x = x * gamma
        if beta is not None:
            x = x + beta
        return x


# ---------- 🔸 残差块 ----------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv(x)


# ---------- 🔸 注意力模块：轻量通道注意力（SE Block） ----------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.se(x)
        return x * weight


# ---------- 🔸 改进版 AOD-PONO-Net ----------
class AOD_pono_net(nn.Module):
    def __init__(self):
        super(AOD_pono_net, self).__init__()
        self.pono = PONO()
        self.ms = MS()

        # conv1 + PONO
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm2d(16)

        # conv2 + PONO
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm2d(16)

        # 中间残差模块
        self.resblock = ResidualBlock(16)

        # conv3 + IN
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.in3 = nn.InstanceNorm2d(16)

        # conv4 + IN
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.in4 = nn.InstanceNorm2d(16)

        # SE 注意力模块
        self.se = SEBlock(16)

        # 输出层
        self.conv_out = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.b = 1  # AOD-Net 中的常数偏移项

    def forward(self, x):
        # conv1 + PONO
        x1 = self.relu(self.in1(self.conv1(x)))
        x1, mean1, std1 = self.pono(x1)

        # conv2 + PONO
        x2 = self.relu(self.in2(self.conv2(x1)))
        x2, mean2, std2 = self.pono(x2)

        # 恢复特征
        x2 = self.ms(x2, mean1, std1)

        # 残差块增强
        x3 = self.resblock(x2)

        # conv3 + IN
        x4 = self.relu(self.in3(self.conv3(x3)))

        # conv4 + IN
        x5 = self.relu(self.in4(self.conv4(x4)))

        # 注意力模块
        x_att = self.se(x5)

        # 输出层
        k = self.relu(self.conv_out(x_att))

        # AOD 输出结构
        if k.size() != x.size():
            raise ValueError("k, input size mismatch!")

        out = k * x - k + self.b
        output = torch.sigmoid(out)  # 限制在 [0, 1] 区间
        return self.relu(out)
