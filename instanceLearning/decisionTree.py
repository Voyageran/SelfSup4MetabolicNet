import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, confusion_matrix

# ====== 路径配置 ======
csv_path = r"G:/project1_obesity_pathway/common_codes/SUV/harmonized_ABC_ComBat.csv"
label_json = r"G:/project1_obesity_pathway/common_codes/SUV/SUV1185/inputs/diagnostic_group12_benchmark_py.json"

# ====== 1. 读取 benchmark 标签 ======
with open(label_json, "r") as f:
    js = json.load(f)

# 自动判断结构（顶层是 list 还是 dict）
if isinstance(js, dict):
    # 自动尝试识别唯一的 int 类型标签数组
    label_key = None
    for key, val in js.items():
        if isinstance(val, list) and all(isinstance(x, int) for x in val):
            label_key = key
            break
    if label_key is None:
        raise ValueError("JSON 中未找到合适的标签键")
    y = np.array(js[label_key], dtype=int)
elif isinstance(js, list):
    y = np.array(js, dtype=int)
else:
    raise ValueError("无法识别 JSON 结构")

print(f"读取标签成功，样本数: {len(y)}")

# ====== 2. 读取 BMI/gender/FBG 特征 ======
df = pd.read_csv(csv_path)
X = df[["BMI", "gender", "FBG"]].copy()

# ====== 3. 丢弃缺失值并对齐 ======
mask = ~(X.isna().any(axis=1))
X = X[mask].reset_index(drop=True)
y = y[mask.values]

assert len(X) == len(y), "特征和标签长度不一致"

# ====== 4. 划分训练/测试集 ======
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ====== 5. 决策树训练 ======
clf = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,            # 可调整：控制层数
    min_samples_leaf=20,    # 每叶子节点最小样本数
    random_state=42
)
clf.fit(X_train, y_train)

# ====== 6. 评估性能 ======
y_pred = clf.predict(X_test)
print("\n=== 分类报告 ===")
print(classification_report(y_test, y_pred, digits=4))
print("\n=== 混淆矩阵 ===")
print(confusion_matrix(y_test, y_pred))

# ====== 7. 打印树结构（自动cutoffs） ======
print("\n=== 决策树分裂规则 ===")
rules = export_text(clf, feature_names=["BMI", "gender", "FBG"])
print(rules)

# ====== 8. 可视化（可选） ======
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(12, 8))
plot_tree(
    clf,
    feature_names=["BMI", "gender", "FBG"],
    class_names=[str(i) for i in sorted(np.unique(y))],
    filled=True,
    rounded=True
)
plt.title("Decision Tree for Predicting Cluster Labels")
plt.tight_layout()
plt.show()
