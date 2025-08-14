import json
import numpy as np
import shap
import matplotlib.pyplot as plt
import pickle
from xgboost import XGBClassifier

# ===================== 1. 参数与路径 ======================
JSON_PATH = r"G:/project1_obesity_pathway/common_codes/SUV/SUV1185/inputs/harmonizedABC_leaveout_diagmean_A.json"
MODEL_PATH = r"xgb_roi_classifier.json"
OUTPUT_PKL = r"shap_values_xgb_selected.pkl"

# 自定义类别（按你需求修改）
CLUSTERS = [8,9,10,11]


# ===================== 2. 数据加载 ======================
def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    mats, labels = [], []
    for entry in data:
        mat = np.array(entry['matrix'], dtype=np.float32)
        mats.append(mat.flatten())  # 展平为121维
        labels.append(entry['label'])

    X = np.stack(mats)
    y = np.array(labels)
    return X, y

def main():
    X, y = load_data(JSON_PATH)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    X = X.astype(np.float32)  # 强制 float 类型，防止 dtype 不匹配

    # ===================== 3. 构建 ROI-ROI 名称 ======================
    organs = [
        "brain", "kidney_left", "kidney_right",
        "liver", "pancreas", "stomach",
        "thyroid_left", "thyroid_right",
        "colon", "duodenum", "small_bowel"
    ]

    feature_names = [
        f"{organs[i]}–{organs[j]}"
        for i in range(11)
        for j in range(11)
    ]

    # ===================== 4. 加载 XGBoost 模型 ======================
    model = XGBClassifier()
    model.load_model(MODEL_PATH)

    # ===================== 5. 计算 SHAP 值 ======================
    # 背景样本：随机抽100个
    X100 = X[np.random.choice(X.shape[0], 100, replace=False)]

    # 创建 SHAP Explainer
    explainer = shap.Explainer(model, X100, feature_names=feature_names)
    shap_values = explainer(X)  # shape = (N, 121, C)
    # shap_values.shape (1160, 121, 12)
    # 模型概率
    pred_probs = model.predict_proba(X)  # shape = (N, C)

    # 合并指定类别 SHAP 值与概率
    selected_prob = pred_probs[:, CLUSTERS].sum(axis=1)  # shape = (N,)
    # shap_combined = shap_values[:, :, CLUSTERS].values.sum(axis=1)  # shape = (N,121)
    # shap_values.values: shape (1160, 121, 12) → 交换为 (1160, 12, 121)
    shap_vals_corrected = np.transpose(shap_values.values, (0, 2, 1))  # [N, C, F]
    # 然后 sum over CLUSTERS → shape (1160, 121)
    shap_combined = shap_vals_corrected[:, CLUSTERS, :].sum(axis=1)
    print(shap_combined.shape)  # (1160, 121)

    # ===================== 6. 绘制 summary_plot ======================
    shap.summary_plot(
        shap_combined,
        X,
        feature_names=feature_names,
        max_display=15,
        show=False
    )
    plt.title(f"SHAP Summary for Selected Classes")
    plt.tight_layout()
    plt.savefig("shap_summary_xgb_selected_clusters.png", dpi=300)
    plt.show()

    # ===================== 7. 绘制平均热图（11×11） ======================
    # 计算所有样本 SHAP 的均值
    shap_mean = shap_combined.mean(axis=0).reshape(11, 11)

    plt.figure(figsize=(6, 5))
    plt.imshow(shap_mean, cmap='RdBu_r', interpolation='nearest', vmin=-np.max(np.abs(shap_mean)),
               vmax=np.max(np.abs(shap_mean)))
    plt.colorbar(label="Mean SHAP value")
    plt.xticks(ticks=range(11), labels=organs, rotation=90)
    plt.yticks(ticks=range(11), labels=organs)
    plt.title(f"Mean SHAP Heatmap for Selected Classes")
    plt.tight_layout()
    plt.savefig("shap_heatmap_xgb_selected_clusters.png", dpi=300)
    plt.show()

    # ===================== 8. 保存结果 ======================
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump({
            "shap_combined": shap_combined,
            "feature_names": feature_names,
            "selected_prob": selected_prob,
            "shap_mean_matrix": shap_mean
        }, f)

    print(f"SHAP results saved to {OUTPUT_PKL}")

if __name__== "__main__":
    main()