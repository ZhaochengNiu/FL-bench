from copy import deepcopy
from typing import Any

import torch

from src.client.fedavg import FedAvgClient


class APFLClient(FedAvgClient):
    # 定义 APFLClient 类，继承自 FedAvgClient 类。
    def __init__(self, **commons):
        # 构造函数接收任意数量的关键字参数 commons。
        super().__init__(**commons)
        # 调用父类 FedAvgClient 的构造函数。
        self.alpha = torch.tensor(self.args.apfl.alpha, device=self.device)
        # 初始化 alpha 参数，这是一个可学习的参数，用于个性化调整客户端的模型更新。
        self.local_model = deepcopy(self.model)
        # 创建一个本地模型的深拷贝。

        def _re_init(src):
            # 定义一个内部函数 _re_init，用于重新初始化模型的特定层。
            target = deepcopy(src)
            # 对传入的模型 src 进行深拷贝。
            for module in target.modules():
                if (
                    isinstance(module, torch.nn.Conv2d)
                    or isinstance(module, torch.nn.BatchNorm2d)
                    or isinstance(module, torch.nn.Linear)
                ):
                    module.reset_parameters()
            # 遍历模型中的所有模块，如果模块是卷积层、批量归一化层或线性层，重置其参数。
            return deepcopy(target.state_dict())
            # 返回模型状态字典的深拷贝。
        self.optimizer.add_param_group({"params": self.local_model.parameters()})
        # 向优化器添加一个参数组，包含本地模型的参数。
        self.init_optimizer_state = deepcopy(self.optimizer.state_dict())
        # 创建优化器状态的深拷贝。

    def set_parameters(self, package: dict[str, Any]):
        # 定义 set_parameters 方法，用于设置客户端的参数。
        super().set_parameters(package)
        # 调用父类方法设置模型参数。
        self.local_model.load_state_dict(package["local_model_params"])
        # 从包裹中加载本地模型参数。
        self.alpha = package["alpha"].to(self.device)
        # 设置 alpha 参数，并将其移动到适当的设备上。

    def package(self):
        # 定义 package 方法，用于打包客户端的更新。
        client_parckage = super().package()
        # 调用父类方法获取基础的客户端包裹。
        client_parckage["local_model_params"] = {
            key: param.cpu().clone()
            for key, param in self.local_model.state_dict().items()
        }
        # 将本地模型的参数添加到客户端包裹中。
        client_parckage["alpha"] = self.alpha.cpu().clone()
        # 将 alpha 参数添加到客户端包裹中。
        return client_parckage
        # 返回客户端包裹。

    def fit(self):
        # 定义 fit 方法，用于客户端的训练过程。
        self.model.train()
        # 将模型设置为训练模式。
        self.local_model.train()
        # 将本地模型设置为训练模式。
        self.dataset.train()
        # 将数据集设置为训练模式。
        for i in range(self.local_epoch):
            # 遍历本地训练周期。
            for x, y in self.trainloader:
                # 遍历训练数据加载器。
                if len(x) <= 1:
                    continue
                    # 如果输入样本数少于或等于1，则跳过当前迭代。
                x, y = x.to(self.device), y.to(self.device)
                # 将数据和标签移动到适当的设备上。
                logit_g = self.model(x)
                # 获取全局模型的预测结果。
                loss = self.criterion(logit_g, y)
                # 计算全局模型的损失。
                self.optimizer.zero_grad()
                # 清除优化器的梯度。
                loss.backward()
                # 执行反向传播。
                self.optimizer.step()
                # 执行优化器的步进。
                logit_l = self.local_model(x)
                # 获取本地模型的预测结果。
                logit_g = self.model(x)
                # 重新获取全局模型的预测结果。
                logit_p = self.alpha * logit_l + (1 - self.alpha) * logit_g.detach()
                # 计算个性化模型的预测结果，这是本地模型和全局模型的加权组合。
                loss = self.criterion(logit_p, y)
                # 计算个性化模型的损失。
                loss.backward()
                # 对个性化模型执行反向传播。
                self.optimizer.step()
                # 更新个性化模型的参数。
                if self.args.apfl.adaptive_alpha and i == 0:
                    self.update_alpha()
                # 如果启用了自适应 alpha 并且是第一个训练周期，则更新 alpha。
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            # 如果存在学习率调度器，则更新学习率。
    # refers to https://github.com/MLOPTPSU/FedTorch/blob/b58da7408d783fd426872b63fbe0c0352c7fa8e4/fedtorch/comms/utils/flow_utils.py#L240

    def update_alpha(self):
        # 定义 update_alpha 方法，用于更新 alpha 参数。
        alpha_grad = 0
        # 初始化 alpha 的梯度。
        for local_param, global_param in zip(
            self.local_model.parameters(), self.model.parameters()
        ):
            # 遍历本地模型和全局模型的参数。
            if local_param.requires_grad:
                # 如果本地参数需要梯度，则继续。
                diff = (local_param.data - global_param.data).flatten()
                # 计算本地参数和全局参数的差异，并将结果展平。
                grad = (
                    self.alpha * local_param.grad.data
                    + (1 - self.alpha) * global_param.grad.data
                ).flatten()
                # 计算加权梯度。
                alpha_grad += diff @ grad
                # 更新 alpha 的梯度。
        alpha_grad += 0.02 * self.alpha
        # 正则化 alpha 的梯度。
        self.alpha.data -= self.args.common.optimizer.lr * alpha_grad
        # 更新 alpha 参数。
        self.alpha.clip_(0, 1.0)
        # 将 alpha 限制在0到1之间。

    def evaluate(self):
        # 定义 evaluate 方法，用于评估模型。
        return super().evaluate(
            model=MixedModel(self.local_model, self.model, alpha=self.alpha)
        )
        # 使用混合模型进行评估，混合模型结合了本地模型、全局模型和 alpha 参数。


# 这个 APFLClient 类实现了个性化的客户端训练逻辑，通过 alpha 参数为每个客户端提供个性化的学习率调整。
# MixedModel 类用于在评估时结合本地模型和全局模型的预测结果。
class MixedModel(torch.nn.Module):
    # 定义 MixedModel 类，继承自 torch.nn.Module。
    def __init__(
        self, local_model: torch.nn.Module, global_model: torch.nn.Module, alpha: float
    ):
        # 构造函数接收本地模型、全局模型和 alpha 参数。
        super().__init__()
        # 调用父类构造函数。
        self.local_model = local_model
        # 保存本地模型。
        self.global_model = global_model
        # 保存全局模型。
        self.alpha = alpha
        # 保存 alpha 参数。

    def forward(self, x):
        return (
            self.alpha * self.local_model(x)
            + (1 - self.alpha) * self.global_model(x).detach()
            # 计算混合模型的输出，结合本地模型和全局模型的预测结果。
        )
