#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supervised classification of 11×11 ROI matrices using XGBoost.
Data format: JSON list of {"matrix": [[...]], "label": ...}.
"""

import json
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data_from_json(json_path):
    """
    从 JSON 文件加载数据。
    JSON 文件格式：顶层为列表，每项为 dict，包含 'matrix' (11×11 数值列表) 和 'label'。
    返回：
        X: np.ndarray, shape=(N, 121)
        y: np.ndarray, shape=(N,)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    mats = []
    labels = []
    for entry in data:
        mat = np.array(entry['matrix'], dtype=np.float32)
        if mat.shape != (11, 11):
            raise ValueError(f"Unexpected matrix shape: {mat.shape}")
        mats.append(mat.flatten())
        labels.append(entry['label'])

    X = np.stack(mats, axis=0)     # shape = (N, 121)
    y = np.array(labels, dtype=np.int64)
    return X, y

def main():
    JSON_PATH = r"G:/project1_obesity_pathway/common_codes/SUV/SUV1185/inputs/harmonizedABC_leaveout_diagmean_A.json"

    # 1. 加载数据
    X, y = load_data_from_json(JSON_PATH)
    print(f"Loaded X shape = {X.shape}, y shape = {y.shape}")

    # 2. 划分训练/测试集（80/20，保持类别分布）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.01,
        stratify=y,
        random_state=123
    )
    print(f"Train shape = {X_train.shape}, Test shape = {X_test.shape}")

    # 3. 定义并训练 XGBoost 分类器
    model = XGBClassifier(
        objective='multi:softprob',
        use_label_encoder=False,
        eval_metric='mlogloss',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=123
    )
    model.fit(X_train, y_train)

    # 4. 在测试集上预测并评估
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 5. 保存为原生 XGBoost 模型，供后续 SHAP 分析
    booster = model.get_booster()
    booster.save_model("xgb_roi_classifier.json")
    print("Native XGBoost model saved to xgb_roi_classifier.json")



if __name__ == "__main__":
    main()
