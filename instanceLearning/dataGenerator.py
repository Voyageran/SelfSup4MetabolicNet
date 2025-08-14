import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

#----------------- 参数 -------------------------
path1k = "G:/project1_obesity_pathway/common_codes/SUV/"
data_csv = "harmonized_ABC_ComBat.csv"
output_json = "G:/project1_obesity_pathway/common_codes/SUV/SUV1185/inputs/harmonizedABC_leaveout_diagmean_py.json"

#----------------- 读取数据 ----------------------
data1k = pd.read_csv(path1k + data_csv)

# BMI聚类
k = 6
km_bmi = KMeans(n_clusters=k, n_init=50, random_state=123)
data1k['cluster_kmeans'] = km_bmi.fit_predict(data1k[['BMI']])

roi_cols = data1k.columns[1:12]  # 11个ROI

#----------------- 生成双通道矩阵 ----------------
matrices = []
labels = []

for g in sorted(data1k['cluster_kmeans'].unique()):
    data_tmp = data1k[data1k['cluster_kmeans'] == g]
    data_SUL = data_tmp[roi_cols].to_numpy()
    group_mean = data_tmp['BMI'].mean()

    # 小组整体相关矩阵 M1
    M1 = np.corrcoef(data_SUL, rowvar=False)

    for i in range(len(data_SUL)):
        data_excluded = np.delete(data_SUL, i, axis=0)
        M2 = np.corrcoef(data_excluded, rowvar=False)
        D_i = M1 - M2

        channel1 = 1500 * D_i  # 放大个体差异
        # channel2 = np.full_like(D_i, group_mean)  # group_mean
        # matrices.append([channel1, channel2])  # shape (2,11,11)
        diag = np.eye(D_i.shape[0])*group_mean
        matrices.append(channel1 + diag)
        labels.append(int(g))

matrices = np.array(matrices)  # 直接变成 [B, 2, 11, 11]
labels = np.array(labels)

#----------------- 保存为 JSON -------------------
# 为了紧凑存储，我们将数据压缩为一个dict
output_data = {
    "label": labels.tolist(),
    "matrix": matrices.tolist()
}

with open(output_json, "w") as f:
    json.dump(output_data, f)
