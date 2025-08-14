'''
这个是对的 20250805

'''
import os
import torch
import torch.nn as nn
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns  # （如果不需要画 seaborn 图可以删掉）
import torch.nn.functional as F
from torchvision.models import resnet
import matplotlib
matplotlib.use('TkAgg')   # 或 'Qt5Agg'，前提是你的环境里装了对应的 GUI 库

# ==============================================
# 部分一：SHAP 分析与保存
# ==============================================
# -------------------------
# 0. 路径 & 参数
# -------------------------
# PT_PATH        = "G:/project1_obesity_pathway/common_codes/SUV/SUV1185/inputs/szwj12groups_70each_50.pt"
# ENCODER_PATH   = "G:/project1_obesity_pathway/project1_2EJMMN_obesityPathway/codes/UnsupervisedClassification_subgroup/results/szwj_subgroup/pretext/model.pth.tar"
# SCAN_PATH      = "G:/project1_obesity_pathway/project1_2EJMMN_obesityPathway/codes/UnsupervisedClassification_subgroup/results/szwj_subgroup/scan/model_10heads.pth.tar"
# SELFLABEL_PATH = "G:/project1_obesity_pathway/project1_2EJMMN_obesityPathway/codes/UnsupervisedClassification_subgroup/results/szwj_subgroup/selflabel/model.pth.tar"
PT_PATH        = "G:/project1_obesity_pathway/common_codes/SUV/SUV1185/inputs/harmonizedABC_leaveout_diagmean_A.pt"
ENCODER_PATH   = "G:/project1_obesity_pathway/project1_2EJMMN_obesityPathway/codes/UnsupervisedClassification_subgroup/results/weak_A/pretext/individual_model.pth.tar"
SCAN_PATH      = "G:/project1_obesity_pathway/project1_2EJMMN_obesityPathway/codes/UnsupervisedClassification_subgroup/results/weak_A/scan/individual_model_10heads.pth.tar"
SELFLABEL_PATH = "G:/project1_obesity_pathway/project1_2EJMMN_obesityPathway/codes/UnsupervisedClassification_subgroup/results/weak_A/selflabel/individual_model.pth.tar"
OUT_SHAP_PATH  = "G:/project1_obesity_pathway/project1_2EJMMN_obesityPathway/codes/UnsupervisedClassification_subgroup/results/szwj_subgroup/scan/output_SHAP.pt"

NUM_CLASSES = 12
# (0, 9), (1, 2), (2, 4), (3, 5), (4, 11), (5, 7), (6, 8), (7, 3), (8, 6), (9, 10), (10, 1), (11, 0)
CLUSTERS = [2,8,6,0]
# -------------------------
# 1. 设备 & 加载特征
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# X, y = torch.load(PT_PATH, map_location="cpu")
# X = X.unsqueeze(1)
data = torch.load(PT_PATH, map_location="cpu")
X = data['tensor']
y = data['labels']

# -------------------------
# 2. 定义模型（Encoder + Head）
# -------------------------
class CustomResNet18(nn.Module):
    def __init__(self, low_dim=64):
        super().__init__()
        self.net = resnet.ResNet(resnet.BasicBlock, [2,2,2,2], num_classes=low_dim)
        self.net.conv1   = nn.Conv2d(1, 64, kernel_size=11, stride=1, bias=False)
        self.net.maxpool = nn.Identity()
    def forward(self, x):
        return self.net(x)

# 2.1 加载 encoder（Pretext 阶段）
encoder = CustomResNet18(low_dim=64).to(device).eval()
enc_state = torch.load(ENCODER_PATH, map_location="cpu")
encoder.load_state_dict(enc_state)

# 2.2 加载 Self‑Label 阶段训练好的 head  <<< 【改动】
self_ckpt = torch.load(SELFLABEL_PATH, map_location="cpu")  # state_dict
# 如果保存格式是 {'model': state_dict}，则需要改成:
# self_ckpt = self_ckpt['model']
head = nn.Linear(64, NUM_CLASSES)
head.load_state_dict({
    'weight': self_ckpt['cluster_head.0.weight'],
    'bias':   self_ckpt['cluster_head.0.bias']
})

# 2.3 组合成一个可调用模型
class Composite(nn.Module):
    def __init__(self, enc, head):
        super().__init__()
        self.enc  = enc
        self.head = head
    def forward(self, x):
        return self.head(self.enc(x))

model = Composite(encoder, head).to(device).eval()

# ==============================================
# 部分二：使用 SHAP 解释
# ==============================================
def predict_fn(X_flat):
    X_torch = torch.from_numpy(X_flat).float().to(device).view(-1, 1, 11, 11)
    with torch.no_grad():
        logits = model(X_torch)
        probs = torch.softmax(logits, dim=1)
        obesity_prob = probs[:, CLUSTERS].sum(dim=1, keepdim=True)
        return obesity_prob.cpu().numpy()  # shape (N, 1)
        # return probs.cpu().numpy()

# 准备背景集和全量输入
X_flat = X.view(X.shape[0], -1).cpu().numpy()
X100_flat = X_flat[np.random.choice(X_flat.shape[0], 100, replace=False)]

# 构建 feature_names
organs = [
    "brain", "kidney_left", "kidney_right",
    "liver", "pancreas", "stomach",
    "thyroid_left", "thyroid_right",
    "colon", "duodenum", "small_bowel"
]
feature_names = [
    f"{organs[i]}-{organs[j]}"
    for i in range(len(organs))
    for j in range(len(organs))
]

# 执行 SHAP 解释
explainer = shap.Explainer(
    predict_fn,
    X100_flat,
    feature_names=feature_names
)
shap_values = explainer(X_flat)


# 保存结果
import pickle
with open("G:/project1_obesity_pathway/project1_2EJMMN_obesityPathway/codes/instanceLearning/shap_figures/output_SHAP_self.pkl", "wb") as f:
    pickle.dump(shap_values, f)


shap.summary_plot(
    shap_values.values,       # (N, F)
    X_flat,                   # (N, F)
    feature_names=feature_names,
    show=False
)

# plt.xlim(-0.02, 0.02)
# # plt.title("SHAP Beeswarm for Top20 Obesity Features")
# plt.tight_layout()
# plt.savefig(
#     f"G:/project1_obesity_pathway/project1_2EJMMN_obesityPathway/"
#     "codes/instanceLearning/shap_figures/beeswarm0.png",
#     dpi=300,
#     bbox_inches='tight'
# )
# plt.close()

