import pickle
import numpy as np
import matplotlib.pyplot as plt

# ========== 参数设置 ==========
PATH_NN  = r"G:/project1_obesity_pathway/project1_2EJMMN_obesityPathway/codes/instanceLearning/shap_figures/output_SHAP_self.pkl"
PATH_XGB = r"G:/project1_obesity_pathway/project1_2EJMMN_obesityPathway/codes/instanceLearning/supervised/shap_values_xgb_selected.pkl"
TOP_K = 20
N_ITER = 1000
DRAW_PLOT = False

# ========== ROI 名称 ==========
organs = [
    "brain", "kidney_left", "kidney_right",
    "liver", "pancreas", "stomach",
    "thyroid_left", "thyroid_right",
    "colon", "duodenum", "small_bowel"
]

# 上三角含对角线索引 (66个)
def get_upper_triangle_indices():
    idx_map = []
    for i in range(11):
        for j in range(i, 11):
            idx_map.append((i, j))
    return idx_map  # 66对

upper_idx_map = get_upper_triangle_indices()
feature_names_66 = [f"{organs[i]}–{organs[j]}" for i, j in upper_idx_map]

# ========== 加载 SHAP 文件 ==========
with open(PATH_NN, "rb") as f:
    shap_nn_obj = pickle.load(f)
with open(PATH_XGB, "rb") as f:
    shap_xgb_obj = pickle.load(f)

shap_nn = shap_nn_obj.values  # (N, 121)
shap_xgb = shap_xgb_obj["shap_combined"]  # (N, 121)

# ========== 对称合并：121 → 66 ==========
def reduce_symmetric_shap(shap_full):
    shap_reduced = []
    for i, j in upper_idx_map:
        idx1 = i * 11 + j
        idx2 = j * 11 + i
        if idx1 == idx2:
            shap_reduced.append(shap_full[:, idx1])
        else:
            shap_reduced.append((shap_full[:, idx1] + shap_full[:, idx2]) / 2)
    return np.stack(shap_reduced, axis=1)  # (N, 66)

shap_nn_reduced = reduce_symmetric_shap(shap_nn)
shap_xgb_reduced = reduce_symmetric_shap(shap_xgb)

# ========== Top-K 重合函数 ==========
def compute_topk_overlap(shap_nn, shap_xgb, k=10):
    mean_nn = np.abs(shap_nn).mean(axis=0)
    mean_xgb = np.abs(shap_xgb).mean(axis=0)
    topk_nn = np.argsort(-mean_nn)[:k]
    topk_xgb = np.argsort(-mean_xgb)[:k]
    overlap = len(set(topk_nn).intersection(set(topk_xgb)))
    return overlap, topk_nn, topk_xgb

# ========== 真实 Top-K 重合 ==========
overlap_real, topk_nn_real, topk_xgb_real = compute_topk_overlap(shap_nn_reduced, shap_xgb_reduced, k=TOP_K)

# ========== 随机检验 ==========
null_overlaps = []
for _ in range(N_ITER):
    shuffled = shap_xgb_reduced.copy()
    np.random.shuffle(shuffled.T)
    overlap_null, _, _ = compute_topk_overlap(shap_nn_reduced, shuffled, k=TOP_K)
    null_overlaps.append(overlap_null)

null_overlaps = np.array(null_overlaps)
p_value = (np.sum(null_overlaps >= overlap_real) + 1) / (N_ITER + 1)

# ========== 输出结果 ==========
print(f"Observed Top-{TOP_K} Overlap = {overlap_real}")
print(f"P-value from permutation test = {p_value:.4f}")

nn_names = [feature_names_66[i] for i in topk_nn_real]
xgb_names = [feature_names_66[i] for i in topk_xgb_real]
print(f"\nTop-{TOP_K} NN ROI–ROI:")
for name in nn_names: print(f"  {name}")
print(f"\nTop-{TOP_K} XGB ROI–ROI:")
for name in xgb_names: print(f"  {name}")

# ========== 可选：绘制图 ==========
if DRAW_PLOT:
    plt.figure(figsize=(7, 5))
    plt.hist(null_overlaps, bins=range(0, TOP_K + 2), color='skyblue', edgecolor='black', rwidth=0.9)
    plt.axvline(overlap_real, color='red', linestyle='--', linewidth=2, label=f'Observed Overlap = {overlap_real}')
    plt.title(f'Top-{TOP_K} SHAP Overlap Null Distribution\np = {p_value:.4f}')
    plt.xlabel('Overlap Count')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig("shap_overlap_pval_symmetric.png", dpi=300)
    plt.show()
