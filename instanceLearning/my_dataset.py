
import json
import torch
from torch.utils.data import Dataset
import numpy as np
############### modification ###############
############### orig
# class CIFAR10(datasets.CIFAR10):
#     def __getitem__(self, index):
#         img, target = super().__getitem__(index)
#         return img, target, index
############### modifications
# class CorrelationMatrixDataset(Dataset):
#     def __init__(self, json_file, label_mapping=None):
#         if label_mapping is None:
#             # label_mapping = {"normal": 0, "overweight": 1, "obesity": 2}
#             label_mapping = {str(i): i - 1 for i in range(1, 24)}
#         self.label_mapping = label_mapping
#
#         # 从JSON文件中加载数据（返回的是一个列表，每个元素是一个字典）
#         with open(json_file, 'r') as f:
#             data = json.load(f)
#
#         self.matrices = []
#         self.labels = []
#
#         # 遍历每个字典，提取"matrix"和"label"
#         for item in data:
#             self.matrices.append(item["matrix"])
#             # 如果标签是列表，则取第一个元素
#             label_val = item["label"]
#             if isinstance(label_val, list):
#                 label_val = label_val[0]
#             self.labels.append(self.label_mapping[label_val])
#
#     def __len__(self):
#         return len(self.matrices)
#
#     def __getitem__(self, idx):
#         matrix = torch.tensor(self.matrices[idx], dtype=torch.float32).unsqueeze(0)  # 扩展为 (1,11,11)
#         labels = torch.tensor(self.labels[idx], dtype=torch.long)
#         return matrix, labels, idx
class CorrelationMatrixDataset(Dataset):
    def __init__(self, json_file):
        # 从 JSON 文件中加载数据（返回的是一个列表，每个元素是一个字典）
        with open(json_file, 'r') as f:
            data = json.load(f)

        self.matrices = []
        self.labels = []

        # 遍历每个字典，提取 "matrix" 和 "label"
        for item in data:
            self.matrices.append(item["matrix"])
            label_val = item["label"]
            # 如果标签是列表，则取第一个元素
            if isinstance(label_val, list):
                label_val = label_val[0]
            self.labels.append(int(label_val))

    def __len__(self):
        return len(self.matrices)

    def __getitem__(self, idx):
        matrix = torch.tensor(self.matrices[idx], dtype=torch.float32).unsqueeze(0)  # 例如扩展为 (1, 11, 11)
        labels = torch.tensor(self.labels[idx], dtype=torch.long)
        return matrix, labels, idx


class CorrelationTensorDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)

        self.matrices = np.array(data["matrix"], dtype=np.float32)  # [B, 2, 11, 11]
        self.labels = np.array(data["label"], dtype=np.int64)       # [B]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        matrix = torch.tensor(self.matrices[idx], dtype=torch.float32)  # (2,11,11)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return matrix, label, idx



class CorrelationMatrixDataset_py(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)

        self.matrices = data["matrix"]
        self.labels = data["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        matrix = torch.tensor(self.matrices[idx], dtype=torch.float32).unsqueeze(0)  # [1,11,11]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return matrix, label, idx

# class CorrelationMatrixDataset(Dataset):
#     def __init__(self, json_file):
#         # 从 JSON 文件中加载数据（返回的是一个列表，每个元素是一个字典）
#         with open(json_file, 'r') as f:
#             data = json.load(f)
#
#         self.matrices = []
#         self.labels = []
#
#         # 遍历每个字典，提取 "matrix" 和 "label"
#         for item in data:
#             self.matrices.append(item["matrix"])
#             label_val = item["label"]
#
#             # 如果标签是列表，则取第一个元素
#             if isinstance(label_val, list):
#                 label_val = label_val[0]
#
#             # 确保标签是整数类型（如果它是字符串，则转换为整数）
#             if isinstance(label_val, str):
#                 label_val = int(label_val)  # 将标签转换为整数
#
#             # 根据要求映射标签
#             if 0 <= label_val <= 7:
#                 mapped_label = 0
#             elif 8 <= label_val <= 12:
#                 mapped_label = 1
#             elif 13 <= label_val <= 15:
#                 mapped_label = 2
#             else:
#                 mapped_label = label_val  # 如果有其他标签的情况，可以根据需求处理
#
#             self.labels.append(mapped_label)
#
#     def __len__(self):
#         return len(self.matrices)
#
#     def __getitem__(self, idx):
#         matrix = torch.tensor(self.matrices[idx], dtype=torch.float32).unsqueeze(0)  # 扩展为 (1, 11, 11)
#         labels = torch.tensor(self.labels[idx], dtype=torch.long)
#         return matrix, labels, idx


class szwj1185Dataset(Dataset):
    def __init__(self, json_file, label_mapping=None):
        if label_mapping is None:
            label_mapping = {"normal": 0, "overweight": 1, "obesity": 2}
        self.label_mapping = label_mapping

        # 从JSON文件中加载数据（返回的是一个列表，每个元素是一个字典）
        with open(json_file, 'r') as f:
            data = json.load(f)

        self.SUVmax = []
        self.labels = []

        # 遍历每个字典，提取"matrix"和"label"
        for item in data:
            # self.SUVmax.append(item["data"])
            # 提取 data 字典中的值
            data_dict = item["data"][0]  # 因为 "data" 是一个列表，里面有一个字典
            data_vals = list(data_dict.values())  # 转换为列表
            self.SUVmax.append(data_vals)
            # 如果标签是列表，则取第一个元素
            label_val = item["label"]
            if isinstance(label_val, list):
                label_val = label_val[0]
            self.labels.append(self.label_mapping[label_val])

    def __len__(self):
        return len(self.SUVmax)

    def __getitem__(self, idx):
        SUV = torch.tensor(self.SUVmax[idx], dtype=torch.float32)
        if SUV.dim() > 1:
            SUV = SUV.squeeze()
        labels = torch.tensor(self.labels[idx], dtype=torch.long)
        return SUV, labels, idx

# from torch.utils.data import Dataset

class IndexedTensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.labels = y  # ✅ 添加这行以支持 clustering 时使用
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], idx
    def __len__(self):
        return len(self.X)

