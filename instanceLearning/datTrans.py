import torch
from torchvision import transforms

# 定义一个将 numpy 数组转换为 tensor 的转换类
class ToTensor(object):
    def __call__(self, sample):
        # sample 是一个 numpy 数组
        return torch.tensor(sample, dtype=torch.float32)

# 如果你需要对 SUV 值做归一化，可以自定义如下转换：
class NormalizeNumeric(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std

