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
# data1k = data1k[data1k['batch'] == 'A']

# BMI聚类
# k = 3
# km_bmi = KMeans(n_clusters=k, n_init=50, random_state=123)
# data1k['cluster_kmeans'] = km_bmi.fit_predict(data1k[['BMI']])
# ---------------- 分成12分位组 --------------------
n_bins = 12
quantiles = np.quantile(data1k["BMI"], q=np.linspace(0, 1, n_bins + 1))
# 去重确保cut成功
quantiles = np.unique(quantiles)
data1k["BMI_group"] = pd.cut(
    data1k["BMI"],
    bins=quantiles,
    include_lowest=True,
    labels=False
)

group_mean_bmi_dict = data1k.groupby("BMI_group")["BMI"].mean().to_dict()

# # 打印各subtype样本数，供验证
print("4平分样本分布：")
print(data1k["BMI_group"].value_counts().sort_index())

roi_cols = data1k.columns[1:12]  # 11个ROI
data_all = data1k[roi_cols].to_numpy()

#----------------- 生成双通道矩阵 ----------------
matrices = []
labels = []

M_all = np.corrcoef(data_all, rowvar=False)  # 全体整体相关矩阵

for idx in range(len(data1k)):
    data_excluded = np.delete(data_all, idx, axis=0)
    M2 = np.corrcoef(data_excluded, rowvar=False)
    D_i = M_all - M2

    channel1 = 1000 * D_i
    group_label = int(data1k.iloc[idx]['BMI_group'])
    group_mean = group_mean_bmi_dict[group_label]  # 取该 group 的均值
    diag = np.eye(D_i.shape[0]) * group_mean
    # patient_bmi = data1k.iloc[idx]['BMI']
    # diag = np.eye(D_i.shape[0]) * patient_bmi
    matrices.append(channel1 + diag)
    # labels.append(int(data1k.iloc[idx]['cluster_kmeans']))
    labels.append(int(data1k.iloc[idx]['BMI_group']))

matrices = np.array(matrices)
labels = np.array(labels)

#----------------- 保存为 JSON -------------------
output_data = {
    "label": labels.tolist(),
    "matrix": matrices.tolist()
}



with open(output_json, "w") as f:
    json.dump(output_data, f)
