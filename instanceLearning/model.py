
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torchvision.models import resnet
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights
# models/gcn_module.py
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv, global_mean_pool
from torchvision.models import resnet50, ResNet50_Weights

class NonParametricClassifierOP(Function):
    @staticmethod
    def forward(ctx, x, y, memory, params):

        tau = params[0].item()
        out = x.mm(memory.t()) #2d时用
        # 调试：打印 x 的维度
        # print("x.shape:", x.shape)
        # print("x.unsqueeze(0).shape:", x.unsqueeze(0).shape)
        # out = x.unsqueeze(0).mm(memory.t()) # 1d数据转成2d
        out.div_(tau)
        ctx.save_for_backward(x, memory, y, params)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, memory, y, params = ctx.saved_tensors
        tau = params[0]
        momentum = params[1]

        grad_output.div_(tau)

        grad_input = grad_output.mm(memory)
        grad_input.resize_as_(x)

        weight_pos = memory.index_select(0, y.view(-1)).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(x.mul(1 - momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)

        return grad_input, None, None, None, None


class NonParametricClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, tau=1.0, momentum=0.5):
        super(NonParametricClassifier, self).__init__()
        self.register_buffer('params', torch.tensor([tau, momentum]))
        stdv = 1. / np.sqrt(input_dim / 3.)
        self.register_buffer(
            'memory',
            torch.rand(output_dim, input_dim).mul_(2 * stdv).add_(-stdv))

    def forward(self, x, y):
        out = NonParametricClassifierOP.apply(x, y, self.memory, self.params)
        return out


import torch.nn as nn
from torchvision import models


class CustomResNet18(nn.Module):
    def __init__(self, low_dim=128):
        super(CustomResNet18, self).__init__()
        self.net = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], num_classes=low_dim)
        self.net.conv1 = nn.Conv2d(1, 64, kernel_size=11, stride=1, padding=0, bias=False)
        self.net.maxpool = nn.Identity()

    def forward(self, x):
        return self.net(x)


class CustomResNet50(nn.Module):
    def __init__(self, in_channels=1, low_dim=128, use_pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if use_pretrained else None
        base_model = resnet50(weights=weights)

        # 原始 7x7 权重
        old_conv = base_model.conv1
        old_weights = old_conv.weight.data  # [64, 3, 7, 7]

        # 创建新 11x11 卷积
        new_conv = nn.Conv2d(in_channels, 64, kernel_size=11, stride=1, padding=0, bias=False)

        if use_pretrained:
            # 插值扩展
            upsampled = F.interpolate(old_weights, size=(11, 11), mode='bilinear', align_corners=True)
            with torch.no_grad():
                new_conv.weight[:, :in_channels] = upsampled[:, :in_channels]

        base_model.conv1 = new_conv
        base_model.maxpool = nn.Identity()
        base_model.fc = nn.Linear(base_model.fc.in_features, low_dim)

        self.backbone = base_model

    def forward(self, x):
        return self.backbone(x)

class MultiScaleDilatedConv(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        self.dilations = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 3, padding=d, dilation=d, bias=False)
            for d in [1, 2, 3, 4, 5]  # dilation = 1~5
        ])

    def forward(self, x):
        convs = [conv(x) for conv in self.dilations]
        # for i, c in enumerate(convs):  # 输出检查
        #     print(f'd{i + 1}conv.shape = {c.shape}')
        return torch.cat(convs, dim=1)  # shape: [B, out_channels * 5, H, W]


class DilatedConvResnet(nn.Module):
    def __init__(self, in_channels=1, mid_channels=1, final_dim=64):
        super().__init__()
        self.multi_scale = MultiScaleDilatedConv(in_channels, out_channels=mid_channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_channels * 5, 64, kernel_size=1),  # 压缩通道
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, final_dim)
        )

    def forward(self, x):
        x = self.multi_scale(x)   # 多尺度感受野
        x = self.fuse(x)          # 融合卷积 + 非线性
        x = self.global_head(x)   # 全局平均池化 -> 表示向量
        return x

class MLP(nn.Module):
    def __init__(self, input_dim=121, hidden_dim=100, output_dim=64):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),  # Flatten the 11x11 input to a 121-dimensional vector
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class MLPnoFlatten(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=10, output_dim=9):
        super(MLPnoFlatten, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class SmallCNN(nn.Module):
    def __init__(self, num_channels=2, feature_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, feature_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
