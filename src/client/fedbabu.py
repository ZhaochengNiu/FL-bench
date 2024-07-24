from src.client.fedavg import FedAvgClient
# 从 src.client.fedavg 模块导入 FedAvgClient 类。


class FedBabuClient(FedAvgClient):
    # 定义 FedBabuClient 类，继承自 FedAvgClient 类。
    def __init__(self, **commons):
        # 构造函数接收任意数量的关键字参数 commons。
        super().__init__(**commons)
        # 调用父类 FedAvgClient 的构造函数。

    # fit 方法的主要作用是在客户端进行本地训练，通过多次迭代训练数据来更新模型的参数。在这个实现中，特别之处在于“固定头部（分类器）”的步骤，
    # 这可能意味着在训练过程中不更新模型的分类器部分。这种方法在某些联邦学习场景中可能有助于提高性能或减少过拟合。
    def fit(self):
        # 定义 fit 方法，用于执行客户端的训练过程。
        self.model.train()
        # 将模型设置为训练模式。
        self.dataset.train()
        # 将数据集设置为训练模式。
        for _ in range(self.local_epoch):
            # 遍历客户端的本地训练周期，self.local_epoch 是客户端进行本地训练的周期数。
            for x, y in self.trainloader:
                # 遍历训练数据加载器 self.trainloader，它提供了数据的批次。
                if len(x) <= 1:
                    continue
                # 如果当前批次的大小小于等于1，跳过当前的迭代。
                # 这是为了避免在模型中的批量归一化层（batchNorm2d）出现错误，因为这些层需要至少一个样本来计算统计量。
                x, y = x.to(self.device), y.to(self.device)
                # 将数据 x 和标签 y 移动到客户端的设备上（CPU或GPU）。
                logit = self.model(x)
                # 通过模型传递数据 x，获取未经激活的输出 logit。
                loss = self.criterion(logit, y)
                # 使用损失函数 self.criterion 计算 logit 和标签 y 之间的损失。
                self.optimizer.zero_grad()
                # 清除优化器的梯度，为反向传播准备。
                loss.backward()
                # 对损失值执行反向传播，计算损失相对于模型参数的梯度。
                # fix head(classifier)
                # 这一行是注释，说明接下来的代码将“固定头部（分类器）”。
                for param in self.model.classifier.parameters():
                    if param.requires_grad:
                        param.grad.zero_()
                # 遍历模型分类器的参数，如果参数需要梯度，则将其梯度清零。这通常是在模型的某些部分不需要更新时使用的技术。
                self.optimizer.step()
                # 根据计算得到的梯度，使用优化器更新模型的参数。
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                # 如果存在学习率调度器，则更新学习率。
