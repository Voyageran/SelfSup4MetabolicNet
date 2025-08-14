import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# ========= 路径配置 =========
csv_path = "G:/project1_obesity_pathway/common_codes/SUV/harmonized_ABC_ComBat.csv"
output_json = "G:/project1_obesity_pathway/common_codes/SUV/SUV1185/inputs/subtype_by_decisiontree_py.json"

# ========= Step 1: 读取数据 =========
df = pd.read_csv(csv_path)
roi_cols = df.columns[1:12]  # 默认你前11列是ROI

# 去除缺失
df = df.dropna(subset=["BMI", "gender", "FBG"] + list(roi_cols)).reset_index(drop=True)

# ========= Step 2: 训练决策树 =========
features = df[["BMI", "gender", "FBG"]]
clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=42)
clf.fit(features, np.zeros(len(df)))  # dummy label，只为建树用

# 每个样本的叶节点 ID（即“亚型标签”）
leaf_ids = clf.apply(features)
df["subtype_label"] = pd.Series(leaf_ids).astype("category").cat.codes  # 编号为 0 ~ n-1

# 可选：打印树
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=["BMI", "gender", "FBG"], filled=True)
plt.title("Decision Tree Subtypes")
plt.tight_layout()
plt.show()

# ========= Step 3: 构造样本协方差矩阵 + 对角线 BMI =========
data_all = df[roi_cols].to_numpy()
M_all = np.corrcoef(data_all, rowvar=False)  # 全体协方差

group_mean_bmi_dict = df.groupby("subtype_label")["BMI"].mean().to_dict()

matrices = []
labels = []

for idx in range(len(df)):
    data_excluded = np.delete(data_all, idx, axis=0)
    M_exclude = np.corrcoef(data_excluded, rowvar=False)
    D_i = M_all - M_exclude
    channel1 = 1000 * D_i

    subtype = int(df.iloc[idx]["subtype_label"])
    diag = np.eye(D_i.shape[0]) * group_mean_bmi_dict[subtype]
    final_matrix = channel1 + diag

    matrices.append(final_matrix)
    labels.append(subtype)

# ========= Step 4: 保存为 JSON =========
output = {
    "label": labels,
    "matrix": [m.tolist() for m in matrices]
}

with open(output_json, "w") as f:
    json.dump(output, f)

print(f"✅ 已保存 {len(labels)} 个样本，类别数 = {df['subtype_label'].nunique()}，保存路径：\n{output_json}")
