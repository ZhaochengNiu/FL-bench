from typing import Any

import torch

from src.client.fedavg import FedAvgClient


class CCVRClient(FedAvgClient):
    # 定义 CCVRClient 类，继承自 FedAvgClient 类。
    def __init__(self, **commons):
        # 构造函数接收任意数量的关键字参数 commons。
        super().__init__(**commons)
        # 调用父类 FedAvgClient 的构造函数。

    def get_classwise_feature_means_and_covs(self, server_package: dict[str, Any]):
        # 定义 get_classwise_feature_means_and_covs 方法，用于获取每个类别的特征均值和协方差矩阵。
        # 该方法接收一个 server_package 参数，这是一个字典，包含服务器发送的包裹。
        self.set_parameters(server_package)
        # 使用服务器包裹中的参数更新客户端的模型参数。
        self.model.eval()
        # 将模型设置为评估模式。
        features = []
        targets = []
        feature_length = None
        # 初始化空列表 features 和 targets 用于存储特征和目标，feature_length 用于存储特征长度。
        for x, y in self.trainloader:
            # 遍历客户端的训练数据加载器。
            x, y = x.to(self.device), y.to(self.device)
            # 将数据和标签移动到适当的设备上。
            features.append(self.model.get_last_features(x))
            # 获取模型的最后一个特征层的输出，并将其添加到 features 列表中。
            targets.append(y)
            # 将标签添加到 targets 列表中。

        targets = torch.cat(targets)
        # 将所有标签连接成一个单一的张量。
        features = torch.cat(features)
        # 将所有特征连接成一个单一的张量。
        feature_length = features.shape[-1]
        # 获取特征的维度。
        indices = [
            torch.where(targets == i)[0] for i in range(len(self.dataset.classes))
        ]
        # 为每个类别创建一个索引列表，包含该类别在 targets 中的索引。
        classes_features = [features[idxs] for idxs in indices]
        # 根据索引列表获取每个类别的特征。
        classes_means, classes_covs = [], []
        # 初始化两个空列表，classes_means 用于存储每个类别的均值，classes_covs 用于存储每个类别的协方差矩阵。
        for fea in classes_features:
            if fea.shape[0] > 0:
                classes_means.append(fea.mean(dim=0))
                classes_covs.append(fea.t().cov(correction=0))
            else:
                classes_means.append(torch.zeros(feature_length, device=self.device))
                classes_covs.append(
                    torch.zeros(feature_length, feature_length, device=self.device)
                )
        # 遍历每个类别的特征，计算均值和协方差矩阵。如果某个类别没有样本，则初始化为零矩阵。
        return dict(
            means=classes_means,
            covs=classes_covs,
            counts=[len(idxs) for idxs in indices],
        )
        # 返回一个字典，包含每个类别的均值、协方差矩阵和样本计数。
