import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------
# 配置
# -------------------------
SHAP_PATH = "G:/project1_obesity_pathway/project1_2EJMMN_obesityPathway/codes/UnsupervisedClassification_subgroup/results/szwj_subgroup/scan/output_SHAP.pt"
OUT_DIR   = "G:/project1_obesity_pathway/project1_2EJMMN_obesityPathway/codes/abstract_submisson/shap_figures"
os.makedirs(OUT_DIR, exist_ok=True)

roi_labels = [
    "brain", "kidney_left", "kidney_right", "liver", "pancreas",
    "stomach", "thyroid_left", "thyroid_right", "colon", "duodenum", "small_bowel"
]

# -------------------------
# 1. 读取 SHAP [12,11,11]
# -------------------------
shap_tensor = torch.load(SHAP_PATH, map_location="cpu")
assert shap_tensor.shape == (12, 11, 11), "SHAP shape mismatch"
shap_tensor = shap_tensor.abs()  # 加绝对值更稳定

# -------------------------
# 2. 强化对称性（(A+B.T)/2）
# -------------------------
for i in range(shap_tensor.shape[0]):
    mat = shap_tensor[i]
    shap_tensor[i] = (mat + mat.T) / 2

# -------------------------
# 3. Bar 图（对角线）
# -------------------------
diag_values = shap_tensor[:, range(11), range(11)]  # shape [12,]
mean_diag = diag_values.mean(dim=0).numpy()

plt.figure(figsize=(10,4))
plt.bar(roi_labels, mean_diag)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Mean SHAP (diagonal)")
plt.title("Average ROI Importance (across clusters)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "bar_diag.png"))
plt.close()

# -------------------------
# 4. Heatmap（簇 × ROI）
# -------------------------
heat_vals = diag_values.numpy()  # shape [12,11]

plt.figure(figsize=(10,6))
sns.heatmap(heat_vals, xticklabels=roi_labels, yticklabels=[f"C{i}" for i in range(12)],
            cmap='YlGnBu', cbar_kws={"label": "|SHAP|"})
plt.title("SHAP Diagonal Heatmap (Clusters × ROI)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "heatmap_diag.png"))
plt.close()

# -------------------------
# 5. Radar 图（每类一张图）
# -------------------------
from math import pi

angles = np.linspace(0, 2 * np.pi, len(roi_labels), endpoint=False).tolist()
angles += angles[:1]  # 闭环

for i in range(12):
    values = diag_values[i].numpy().tolist()
    values += values[:1]

    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(roi_labels, fontsize=9)
    ax.set_title(f"Cluster {i} ROI Importance", y=1.08)
    ax.set_yticklabels([])
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"radar_cluster{i}.png"))
    plt.close()

print("[✓] All SHAP visualizations saved.")
