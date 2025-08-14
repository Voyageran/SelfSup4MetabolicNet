import pickle
import pandas as pd
import matplotlib.pyplot as plt

with open('G:\\JiangRan\\networkAnalysis\\pathway_info\\codes\\Clustering_friendly_representation_learning-master\\clustering_metrics.pkl', 'rb') as f:
    history = pickle.load(f)

data = history
if isinstance(data, dict):
    df = pd.DataFrame([data])
elif isinstance(data, list):
    df = pd.DataFrame(data)
else:
    raise ValueError("Unsupported data format")


# 每隔10行取一次数据
df_sampled = df.iloc[::1]

# 绘制三条曲线
plt.figure(figsize=(10, 6))
plt.plot(df_sampled['epoch'], df_sampled['acc'], label='Accuracy', color='blue', linewidth=2)
plt.plot(df_sampled['epoch'], df_sampled['nmi'], label='NMI', color='green', linewidth=2)
plt.plot(df_sampled['epoch'], df_sampled['ari'], label='ARI', color='red', linewidth=2)

# 添加标题和标签
plt.title('Performance Metrics Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Score')

# 添加图例
plt.legend()

folder_path = 'G:\\JiangRan\\networkAnalysis\\pathway_info\\augumenDat\\cormatrices_sameInt\\figures\\'
plt.savefig(folder_path+"meanszwj_addInd89_corsSum10.jpg", dpi=300)



# 显示图表
# plt.show()
with open('G:\\JiangRan\\networkAnalysis\\pathway_withXie\\codes\\Clustering_friendly_representation_learning\\y_pred_history.pkl', 'rb') as f:
    y_history = pickle.load(f)

data = y_history
if isinstance(data, dict):
    df = pd.DataFrame([data])
elif isinstance(data, list):
    df = pd.DataFrame(data)
else:
    raise ValueError("Unsupported data format")

with open('G:\\JiangRan\\networkAnalysis\\pathway_info\\codes\\Clustering_friendly_representation_learning-master\\prototype_history.pkl', 'rb') as f:
    prototype_history = pickle.load(f)
data = prototype_history
if isinstance(data, dict):
    df = pd.DataFrame([data])
elif isinstance(data, list):
    df = pd.DataFrame(data)
else:
    raise ValueError("Unsupported data format")

final_prototypes = prototype_history[-1]['prototype']
final_df = pd.DataFrame(final_prototypes)

final_df.to_json('G:\\JiangRan\\networkAnalysis\\pathway_info\\augumenDat\\cormatrices_sameInt\\cm23_t1.json', orient='records', lines=True)