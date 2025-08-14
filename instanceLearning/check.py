from my_dataset import CorrelationMatrixDataset
import torch

folder_path = 'G:/project1_obesity_pathway/common_codes/SUV/SUV1185/inputs/'
dataset_path = folder_path + "szwj16groups_50each_50.json"
trainset = CorrelationMatrixDataset(dataset_path)


labels = trainset.labels

# 若是 list，转为 Tensor（更方便处理）
if isinstance(labels, list):
    labels = torch.tensor(labels)

print("标签最小值:", labels.min().item())
print("标签最大值:", labels.max().item())
print("唯一标签值:", torch.unique(labels))
