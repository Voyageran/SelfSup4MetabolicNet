import pickle
import numpy as np
import matplotlib.pyplot as plt
import shap

# 1. 器官与 feature_names（与你的一致）
organs = [
    "brain", "kidney_left", "kidney_right",
    "liver", "pancreas", "stomach",
    "thyroid_left", "thyroid_right",
    "colon", "duodenum", "small_bowel"
]
feature_names = [
    f"{organs[i]}–{organs[j]}"
    for i in range(len(organs))
    for j in range(len(organs))
]

# 2. 加载之前保存的 shap_values（完整路径，无省略）
with open(
    "G:/project1_obesity_pathway/project1_2EJMMN_obesityPathway/"
    "codes/instanceLearning/shap_figures/output_SHAP_self.pkl", "rb"
) as f:
    shap_values = pickle.load(f)

# 3. 提取 SHAP 数值与原始输入
sv     = shap_values.values   # shape: (N, 121)
X_flat = shap_values.data     # shape: (N, 121)

# 4. 打印所有特征及其行号，便于手动选择要删除的那些重复项
print("===== 0-based 行号 : 特征名称 =====")
for idx, name in enumerate(feature_names):
    print(f"{idx:3d}: {name}")

# TODO: 删除重复的ROI
#  73: thyroid_left–thyroid_right  44: pancreas–brain  42: liver–duodenum
#  70: thyroid_left–pancreas  83: thyroid_right–thyroid_left 103: duodenum–pancreas
drops = {73,44,42,70,83,103}

# 6. 在原始 121 特征里，剔除掉指定的行
keep2  = [i for i in range(len(feature_names)) if i not in drops]
names2 = [feature_names[i] for i in keep2]

# 7. 切片出新的 SHAP 数值矩阵和输入矩阵
sv2 = sv[:, keep2]      # shape: (N, len(keep2))
X2  = X_flat[:, keep2]  # shape: (N, len(keep2))

# 8. 绘制最终的 Beeswarm 图，保持横轴 ±0.02
plt.figure(figsize=(10, 8))
shap.summary_plot(
    sv2,
    X2,
    feature_names=names2,
    max_display=15,
    show=False
)
plt.xlim(-0.03, 0.03)
# plt.title("SHAP Beeswarm – Manual Filtered Features")
plt.tight_layout()
plt.savefig(
    "G:/project1_obesity_pathway/project1_2EJMMN_obesityPathway/"
    "codes/instanceLearning/shap_figures/beeswarm_top15.png",
    dpi=300,
    bbox_inches='tight'
)
plt.close()
