#! /usr/bin/env python

import os
import argparse
import json
import pickle

import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans
import tqdm.autonotebook as tqdm
import faiss

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.quantization import resnet50
import torchvision.models as models
import matplotlib
matplotlib.use("TkAgg")  # 设置 GUI 后端
import matplotlib.pyplot as plt


from my_dataset import CorrelationMatrixDataset, szwj1185Dataset
from AverageTracker import AverageTracker
from metrics import metrics
from model import NonParametricClassifier, MLP, MLPnoFlatten, CustomResNet18, CustomResNet50, DilatedConvResnet
from Normalize import Normalize
from loss import Loss
from Json2pt import Json2pt
from my_dataset import IndexedTensorDataset


#torch.cuda.isavaliable()
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpus", type=str, default="")
    # parser.add_argument("-n", "--num_workers", type=int, default=8)
    parser.add_argument("-n", "--num_workers", type=int, default=1)
    parser.add_argument("--dataset_path", type=str,
                        default="G:/project1_obesity_pathway/common_codes/SUV/SUV1185/inputs/harmonizedABC_leaveout_diagmean_AAplus_B.json",
                        help="Path to dataset JSON") #smooth_quantile16_corr.json
                                                    # szwj16groups_50each_50.json smooth_quantile16_corr.json synthetic_autoBMI_3groups.json

    parser.add_argument("--low_dim", type=int, default=64, help="Dimension of embedding")
    parser.add_argument("--iter_num", type=int, default=150, help="Number of epoch iteration")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")

    parser.add_argument("--net_type", type=str, default="CustomResNet18",
                        choices=["CustomResNet18", "CustomResNet50", "DilatedConvResnet"],
                        help="Type of backbone network")
    parser.add_argument("--lib", type=str, default="faiss", choices=["faiss", "sklearn"],
                        help="Clustering backend library")
    parser.add_argument("--init", type=str, default="kmeans", choices=["random", "kmeans++"],
                        help="KMeans initialization method")

    parser.add_argument("--tau", type=float, default=1.0, help="Tau for NonParametricClassifier")
    parser.add_argument("--momentum", type=float, default=0.5, help="Momentum for NonParametricClassifier")
    parser.add_argument("--tau2", type=float, default=1.5, help="Tau2 for Loss")
    # optimizer
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate")
    parser.add_argument("--opt_momentum", type=float, default=0.9, help="Momentum for optimizer")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")


    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    return args


def main():
    args = parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取 JSON 文件
    dataset_path = args.dataset_path
    trainset = CorrelationMatrixDataset(dataset_path)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    if args.net_type == "CustomResNet18":
        net = CustomResNet18(low_dim=args.low_dim)
    elif args.net_type == "CustomResNet50":
        net = CustomResNet50(low_dim=args.low_dim)
    elif args.net_type == "DilatedConvResnet":
        net = DilatedConvResnet(in_channels=1, mid_channels=1, final_dim=args.low_dim)
    else:
        raise ValueError(f"Unsupported net_type: {args.net_type}")

    norm = Normalize(2)
    npc = NonParametricClassifier(input_dim=args.low_dim, output_dim=len(trainset),
                                  tau=args.tau, momentum=args.momentum)
    loss = Loss(tau2=args.tau2)

    net, norm = net.to(device), norm.to(device)
    npc, loss = npc.to(device), loss.to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.opt_momentum,
                                weight_decay=args.weight_decay, nesterov=False, dampening=0)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [600, 950, 1300, 1650], gamma=0.1)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    torch.backends.cudnn.benchmark = True

    trackers = {n: AverageTracker() for n in ["loss", "loss_id", "loss_fd"]}
    clustering_metrics_history = []
    y_pred_history = []  #

    best_acc = -1.0
    best_model_state = None
    best_epoch = 1
    with tqdm.trange(args.iter_num) as epoch_bar:
        for epoch in epoch_bar:
            net.train()
            for batch_idx, (inputs, _, indexes) in enumerate(tqdm.tqdm(train_loader)):
                optimizer.zero_grad()
                inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
                indexes = indexes.to(device, non_blocking=True)
                features = norm(net(inputs))
                outputs = npc(features, indexes)
                loss_id, loss_fd = loss(outputs, features, indexes)
                tot_loss = loss_id + loss_fd
                tot_loss.backward()
                optimizer.step()

                # track loss
                trackers["loss"].add(tot_loss)
                trackers["loss_id"].add(loss_id)
                trackers["loss_fd"].add(loss_fd)
            lr_scheduler.step()
            postfix = {name: t.avg() for name, t in trackers.items()}
            epoch_bar.set_postfix(**postfix)

            # === 评估聚类表现，每2轮保存一次 ===
            if (epoch == 0) or (((epoch + 1) % 2) == 0):
                acc, nmi, ari, centroids, y_pred = check_clustering_metrics(npc, train_loader,lib=args.lib, init=args.init)

                print(
                    "Epoch:{} | loss={:.2f}, loss_fd={:.2f}, loss_id={:.2f} | Kmeans ACC={:.5f}, NMI={:.4f}, ARI={:.4f}".format(
                        epoch + 1,
                        trackers["loss"].avg(),
                        trackers["loss_fd"].avg(),
                        trackers["loss_id"].avg(),
                        acc, nmi, ari
                    ))

                # 保存历史聚类预测
                y_pred_history.append({
                    'epoch': epoch + 1,
                    'y_pred': y_pred.tolist()
                })

                # if acc > best_acc:
                #     best_acc = acc
                #     best_model_state = {
                #         "net": net.module.state_dict() if isinstance(net, torch.nn.DataParallel) else net.state_dict(),
                #         "npc": npc.state_dict()
                #     }
                #     torch.save(best_model_state, save_path)
                #     best_epoch = epoch + 1
                #     print(f"New best ACC={best_acc:.5f} at epoch {epoch + 1}, model saved to {save_path}")
                # else:
                #     print(f"The best ACC={best_acc:.5f} at epoch {best_epoch}.")
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch + 1
                    # 推荐使用原始字符串或正斜杠：
                    save_dir = r'G:/project1_obesity_pathway/project1_2EJMMN_obesityPathway/codes/UnsupervisedClassification_subgroup/results/szwj_subgroup/pretext'
                    save_path = os.path.join(save_dir, 'model.pth.tar')
                    # 保存完整 state_dict
                    state = net.state_dict()
                    torch.save(state, save_path)
                    print(f"New best ACC={best_acc:.5f} at epoch {epoch + 1}, model saved.")
                else:
                    print(f"The best ACC={best_acc:.5f} at epoch {best_epoch}.")


                clustering_metrics_history.append({
                    'epoch': epoch + 1,
                    'acc': acc,
                    'nmi': nmi,
                    'ari': ari,
                    'loss': float(trackers["loss"].avg()),
                    'loss_fd': float(trackers["loss_fd"].avg()),
                    'loss_id': float(trackers["loss_id"].avg())
                })

            for t in trackers.values():
                t.reset()

    # 保存最后一轮聚类标签 y_pred（或者改为 best acc 的 y_pred）
    y_pred_final = y_pred_history[-1]['y_pred']  # 最后一轮
    np.save("y_pred_final.npy", np.array(y_pred_final))  # 可直接加载为 np.array
    pd.DataFrame({'cluster': y_pred_final}).to_csv("y_pred_final.csv", index_label="net_id")

    print("已保存最终聚类标签: y_pred_final.npy 和 y_pred_final.csv")

    # ---- 训练完成之后 ----
    # print("训练结束，开始绘制激活图...")
    # all_inputs = [inputs for inputs, _, _ in train_loader]
    # input_tensor = torch.cat(all_inputs, dim=0).to(device)
    # # Step 2: 注册 hook，准备保存中间输出
    # activations, hooks = register_hooks(net)
    # # Step 3: forward 一次，触发 hook
    # net.eval()
    # with torch.no_grad():
    #     _ = net(input_tensor)
    # print(activations.keys())  # 打印激活图的 keys，确认是否有 'conv1'
    # conv1_activation = None
    # for module in activations:
    #     # 使用 str(module) 来匹配 conv1
    #     if str(module) == 'Conv2d(1, 64, kernel_size=(11, 11), stride=(1, 1), bias=False)':
    #         conv1_activation = activations[module]
    #         break
    #
    # if conv1_activation is None:
    #     print("未能找到 'conv1' 激活图，检查 hook 注册情况")
    # else:
    #     # Step 4: 检查尺度并标准化激活图
    #     print(f"Activation min: {conv1_activation.min()}")
    #     print(f"Activation max: {conv1_activation.max()}")
    #     print(f"Activation mean: {conv1_activation.mean()}")
    #
    #     # 对激活图进行标准化，使其范围在 [0, 1] 之间
    #     conv1_activation = (conv1_activation - conv1_activation.min()) / (
    #                 conv1_activation.max() - conv1_activation.min())
    #
    #     # Step 5: 可视化前10个样本
    #     plot_activation_map(conv1_activation[:10], prefix="conv1_sample")
    #
    # # Step 6: 清理 hook（建议，防止后面干扰）
    # for h in hooks:
    #     h.remove()

        # === 训练后保存 CSV 和可视化 ===
    df = pd.DataFrame(clustering_metrics_history)
    df.to_csv("clustering_metrics_and_loss_combined.csv", index=False)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    epochs = df["epoch"]
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("ACC / NMI / ARI", color="tab:blue")
    ax1.plot(epochs, df["acc"], label="ACC", color="tab:blue", marker="o")
    ax1.plot(epochs, df["nmi"], label="NMI", color="tab:green", marker="x")
    ax1.plot(epochs, df["ari"], label="ARI", color="tab:orange", marker="^")
    ax1.tick_params(axis='y', labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Loss", color="tab:red")
    ax2.plot(epochs, df["loss"], label="Loss", color="tab:red", linestyle="--")
    ax2.plot(epochs, df["loss_fd"], label="Loss_fd", color="tab:purple", linestyle="--")
    ax2.plot(epochs, df["loss_id"], label="Loss_id", color="tab:pink", linestyle="--")
    ax2.tick_params(axis='y', labelcolor="tab:red")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.title("Training Metrics Over Epochs")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("metrics_plot.png")
    plt.show()

    #  保存聚类预测历史
    with open("y_pred_history.pkl", "wb") as f:
        pickle.dump(y_pred_history, f)
    print(" 已保存 y_pred_history.pkl，包含所有聚类预测结果")

    return clustering_metrics_history



# 定义hook来获取中间层的输出
def register_hooks(model):
    activations = {}

    def hook_fn(module, input, output):
        activations[module] = output

    hooks = []
    for name, module in model.named_modules():
        print(f"注册模块: {name}")  # 打印每个模块的名称
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(hook_fn))

    return activations, hooks


def plot_activation_map(activation_batch, num_columns=8, save_dir="activation_maps", prefix="sample"):
    os.makedirs(save_dir, exist_ok=True)
    batch_size = activation_batch.shape[0]

    for idx in range(batch_size):
        activation = activation_batch[idx]  # shape: (C, H, W), where C=64, H=11, W=11
        num_activations = activation.shape[0]  # 获取通道数 C = 64

        # 计算所需行数
        num_rows = num_activations // num_columns + (num_activations % num_columns != 0)

        fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns * 1.5, num_rows * 1.5))
        axes = axes.flatten()

        # 显示每个通道的激活图
        for i in range(num_activations):
            ax = axes[i]
            # activation[i] 是 (11, 11)，符合 imshow 的要求
            ax.imshow(activation[i].cpu().detach().numpy(), cmap='viridis')
            ax.axis('off')

        # 如果有多余的子图，关闭它们
        for i in range(num_activations, len(axes)):
            axes[i].axis('off')

        plt.suptitle(f"{prefix}_{idx}", fontsize=16)
        plt.tight_layout()

        # 保存为 PNG 文件
        save_path = os.path.join(save_dir, f"{prefix}_{idx}.png")
        plt.savefig(save_path)
        print(f"保存成功: {save_path}")
        plt.close()


# clusters的数量 K
def check_clustering_metrics(npc, train_loader, lib="sklearn", init="kmeans"):

    trainFeatures = npc.memory  # (N, D)
    z = trainFeatures.cpu().numpy()
    y = np.array(train_loader.dataset.labels)
    n_clusters = len(np.unique(y))
    d = z.shape[1]

    if lib == "sklearn":
        if init == "kmeans++":
            kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=20)
        elif init == "kmeans":
            kmeans = KMeans(n_clusters=n_clusters, init="random", n_init=20)
        else:
            raise ValueError(f"[sklearn] 不支持 init='{init}'")
        y_pred = kmeans.fit_predict(z)
        centroids = kmeans.cluster_centers_

    elif lib == "faiss":
        z = z.astype("float32")  # faiss 要求 float32

        if init == "kmeans++":
            clus = faiss.Clustering(d, n_clusters)
            clus.niter = 100
            clus.seed = 123
            clus.init_index = faiss.IndexFlatL2(d)  # 用 L2 初始化（相当于 kmeans++）
            index = faiss.IndexFlatL2(d)
            clus.train(z, index)
            _, y_pred = index.search(z, 1)
            centroids = faiss.vector_to_array(clus.centroids).reshape(n_clusters, d)

        elif init == "kmeans":
            kmeans = faiss.Kmeans(d=d, k=n_clusters, niter=100, verbose=True)
            kmeans.train(z)
            _, y_pred = kmeans.index.search(z, 1)
            centroids = kmeans.centroids
        else:
            raise ValueError(f"[faiss] 不支持 init='{init}'")

        y_pred = y_pred.reshape(-1)

    else:
        raise ValueError(f"不支持 lib='{lib}'")

    return (
        metrics.acc(y, y_pred),
        metrics.nmi(y, y_pred),
        metrics.ari(y, y_pred),
        centroids,
        y_pred,
    )


if __name__ == "__main__":
    clustering_metrics_history = main()
    with open('clustering_metrics.pkl', 'wb') as f:
        pickle.dump(clustering_metrics_history, f)
    print("训练后的指标历史:", clustering_metrics_history)