import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import shap
import pickle

# === 0. 路径 & 参数（沿用你原来的） ===
# PT_PATH        = "G:/project1_obesity_pathway/common_codes/SUV/SUV1185/inputs/szwj12groups_70each_50.pt"
# ENCODER_PATH   = "G:/project1_obesity_pathway/project1_2EJMMN_obesityPathway/codes/UnsupervisedClassification_subgroup/results/szwj_subgroup/pretext/model.pth.tar"
# SELFLABEL_PATH = "G:/project1_obesity_pathway/project1_2EJMMN_obesityPathway/codes/UnsupervisedClassification_subgroup/results/szwj_subgroup/selflabel/model.pth.tar"
# OUT_DIR        = "G:/project1_obesity_pathway/project1_2EJMMN_obesityPathway/codes/instanceLearning/shap_figures/"
PT_PATH        = "G:/project1_obesity_pathway/common_codes/SUV/SUV1185/inputs/weak_A.pt"
ENCODER_PATH   = "G:/project1_obesity_pathway/project1_2EJMMN_obesityPathway/codes/UnsupervisedClassification_subgroup/results/harmonizedABC_leaveout_diagmean/pretext/individual_model.pth.tar"
SCAN_PATH      = "G:/project1_obesity_pathway/project1_2EJMMN_obesityPathway/codes/UnsupervisedClassification_subgroup/results/harmonizedABC_leaveout_diagmean/scan/individual_model_10heads.pth.tar"
SELFLABEL_PATH = "G:/project1_obesity_pathway/project1_2EJMMN_obesityPathway/codes/UnsupervisedClassification_subgroup/results/harmonizedABC_leaveout_diagmean/selflabel/individual_model.pth.tar"
OUT_DIR        = "G:/project1_obesity_pathway/project1_2EJMMN_obesityPathway/codes/instanceLearning/shap_figures/"

# 簇数量
NUM_CLASSES = 12

# ====================================
# 1. 设备 & 加载数据
# ====================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# X, y   = torch.load(PT_PATH, map_location="cpu")
# X      = X.unsqueeze(1).to(device)
data = torch.load(PT_PATH, map_location="cpu")
X = data['tensor'].to(device)
y = data['labels']

# Flatten numpy arrays for SHAP
X_flat    = X.view(X.shape[0], -1).cpu().numpy()
bg_flat   = X_flat[np.random.choice(X_flat.shape[0], 100, replace=False)]

# ====================================
# 2. 定义并加载模型
# ====================================
from torchvision.models import resnet
class CustomResNet18(torch.nn.Module):
    def __init__(self, low_dim=64):
        super().__init__()
        self.net = resnet.ResNet(resnet.BasicBlock, [2,2,2,2], num_classes=low_dim)
        self.net.conv1   = torch.nn.Conv2d(1, 64, kernel_size=11, stride=1, bias=False)
        self.net.maxpool = torch.nn.Identity()
    def forward(self, x):
        return self.net(x)

# encoder
encoder = CustomResNet18(low_dim=64).to(device).eval()
enc_state = torch.load(ENCODER_PATH, map_location="cpu")
encoder.load_state_dict(enc_state)

# head
self_ckpt = torch.load(SELFLABEL_PATH, map_location="cpu")
head = torch.nn.Linear(64, NUM_CLASSES)
head.load_state_dict({
    'weight': self_ckpt['cluster_head.0.weight'],
    'bias':   self_ckpt['cluster_head.0.bias']
})

# composite
class Composite(torch.nn.Module):
    def __init__(self, enc, head):
        super().__init__()
        self.enc, self.head = enc, head
    def forward(self, x):
        return self.head(self.enc(x))

model = Composite(encoder, head).to(device).eval()

# ====================================
# 3. 准备 feature_names
# ====================================
organs = [
    "brain","kidney_left","kidney_right",
    "liver","pancreas","stomach",
    "thyroid_left","thyroid_right",
    "colon","duodenum","small_bowel"
]
feature_names = [
    f"{organs[i]}–{organs[j]}"
    for i in range(len(organs))
    for j in range(len(organs))
]

# ====================================
# 4. 循环每个簇，解释并画图
# ====================================
for cid in range(NUM_CLASSES):
    # 4.1 针对 cid 构建 predict_fn（返回 shape (N,1)）
    def make_predict_fn(cluster_id):
        def predict_fn(x_np):
            x_t = torch.from_numpy(x_np).float().to(device).view(-1,1,11,11)
            with torch.no_grad():
                logits = model(x_t)
                probs  = torch.softmax(logits, dim=1)
                return probs[:, [cluster_id]].cpu().numpy()
        return predict_fn

    predict_fn = make_predict_fn(cid)

    # 4.2 构造解释器 & 计算 SHAP
    explainer = shap.Explainer(predict_fn, bg_flat, feature_names=feature_names)
    shap_values = explainer(X_flat)     # shap_values.values.shape == (N, F)

    # 4.3 （可选）保存解释结果 pkl
    with open(os.path.join(OUT_DIR, f"shap_cluster{cid}.pkl"), "wb") as f:
        pickle.dump(shap_values, f)

    # 4.4 绘制完整 Beeswarm（不限制前 20）
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values.values,
        X_flat,
        feature_names=feature_names,
        max_display=10,   # 显示所有特征
        show=False
    )
    plt.xlim(-0.02, 0.02)
    # plt.title(f"Cluster {cid} SHAP Beeswarm (all features)")
    plt.tight_layout()

    # 4.5 保存图片
    plt.savefig(
        os.path.join(OUT_DIR, f"G:/project1_obesity_pathway/project1_2EJMMN_obesityPathway/codes/instanceLearning/shap_figures/beeswarm_cluster{cid}.png"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
