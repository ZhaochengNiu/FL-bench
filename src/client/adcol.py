from typing import Any
# 这行代码从 typing 模块导入了 Any 类型。Any 是一个特殊的类型，它可以用来指示函数或方法可以接受任何类型的参数。
import torch
# 这行代码导入了 PyTorch 库。PyTorch 是一个流行的开源机器学习库，广泛用于计算机视觉和自然语言处理等深度学习任务。
import torch.nn as nn
# 这行代码从 PyTorch 库中导入了神经网络模块 torch.nn，并将其重命名为 nn。
# 这个模块包含了构建神经网络所需的类和函数，比如层（Layer）和损失函数。
import torch.nn.functional as F
# 这行代码从 PyTorch 库中导入了神经网络函数模块 torch.nn.functional，并将其重命名为 F。
# 这个模块提供了一些函数式接口，用于执行神经网络中的常见操作，如激活函数（如 ReLU）、池化函数等。
from src.client.fedavg import FedAvgClient
# 这行代码从项目的 src.client.fedavg 模块中导入了 FedAvgClient 类。


# 这个 ADCOLClient 类实现了 ADCOL 算法的客户端逻辑，包括训练模型、收集特征、设置参数和打包客户端包裹。
# 通过这种方式，客户端可以在联邦学习环境中与其他客户端协同训练模型。
class ADCOLClient(FedAvgClient):
    # 这段代码定义了一个名为 ADCOLClient 的类，它继承自 FedAvgClient 类。
    # ADCOLClient 类实现了一个客户端，用于在联邦学习中使用对抗性领域适应（ADCOL）算法。
    # 定义了一个名为 ADCOLClient 的新类，它继承自 FedAvgClient 类。
    def __init__(self, discriminator: torch.nn.Module, client_num: int, **commons):
        # 构造函数接收以下参数：
        # discriminator：一个 torch.nn.Module 对象，用于区分不同客户端的特征。
        # client_num：客户端的数量。
        # **commons：其他通用参数，这些参数将被传递给父类 FedAvgClient。
        super(ADCOLClient, self).__init__(**commons)
        # 调用父类 FedAvgClient 的构造函数，传递通用参数。
        self.discriminator = discriminator.to(self.device)
        # 将 discriminator 模型移动到客户端的设备上（CPU或GPU）。
        self.client_num = client_num
        # 保存客户端的数量。
        self.features_list = []
        # 初始化一个空列表，用于存储客户端的特征。

    def fit(self):
        # 定义了 fit 方法，用于训练模型。
        self.model.train()
        # 将模型设置为训练模式。
        self.discriminator.eval()
        # 将判别器设置为评估模式。
        self.dataset.train()
        # 将数据集设置为训练模式。
        self.features_list = []
        # 重置特征列表。
        for i in range(self.local_epoch):
            # 遍历本地训练周期。
            for x, y in self.trainloader:
                # 遍历训练数据加载器。
                if len(x) <= 1:
                    continue
                # 如果输入数据 x 的长度小于等于1，则跳过当前迭代。
                x, y = x.to(self.device), y.to(self.device)
                # 将数据和标签移动到客户端的设备上。
                try:
                    features = self.model.base(x)
                    logit = self.model.classifier(F.relu(features))
                except:
                    raise ValueError(
                        "model may have no feature extractor + classifier architecture"
                    )
                # 尝试从模型的基础部分提取特征，并使用分类器生成预测结果。如果模型没有特征提取器和分类器的架构，则抛出异常。
                cross_entropy = self.criterion(logit, y).mean()
                # 计算交叉熵损失。
                client_index = self.discriminator(features)
                # 使用判别器获取客户端索引。
                client_index_softmax = F.log_softmax(client_index, dim=-1)
                # 对客户端索引进行 softmax 转换。
                target_index = torch.full(client_index.shape, 1 / self.client_num).to(
                    self.device
                )
                # 创建一个目标索引张量，其值均匀分布。
                target_index_softmax = F.softmax(target_index, dim=-1)
                # 对目标索引进行 softmax 转换。
                kl_loss_func = nn.KLDivLoss(reduction="batchmean").to(self.device)
                # 创建 KL 散度损失函数。
                kl_loss = kl_loss_func(client_index_softmax, target_index_softmax)
                # 计算 KL 散度损失。
                mu = self.args.adcol.mu
                # 获取 ADCOL 算法的参数 mu。
                loss = cross_entropy + mu * kl_loss
                # 计算总损失。
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # 执行反向传播和优化器步骤。
                # collect features in the last epoch
                if i == self.local_epoch - 1:
                    self.features_list.append(features.detach().clone().cpu())
                # 在最后一个训练周期收集特征。
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                # 如果存在学习率调度器，则更新学习率。
        self.feature_list = torch.cat(self.features_list, dim=0)
        # 将特征列表连接成一个张量。

    def set_parameters(self, package: dict[str, Any]):
        # 定义了 set_parameters 方法，用于设置模型参数。
        super().set_parameters(package)
        # 调用父类方法来设置模型参数。
        self.discriminator.load_state_dict(package["new_discriminator_params"])
        # 加载判别器的新参数。

    def package(self):
        # 定义了 package 方法，用于打包客户端的包裹。
        client_package = super().package()
        # 调用父类方法来获取客户端包裹。
        client_package["features_list"] = self.features_list
        # 将特征列表添加到客户端包裹中。
        return client_package
        # 返回客户端包裹。
