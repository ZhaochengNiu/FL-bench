from argparse import ArgumentParser, Namespace
# 从 argparse 模块导入 ArgumentParser 和 Namespace。ArgumentParser 用于解析命令行参数，Namespace 用于存储解析后的参数。
from copy import deepcopy
# 从 copy 模块导入 deepcopy 函数，该函数用于创建对象的深拷贝，这对于复制包含可变类型的对象（如列表或字典）非常有用。
import torch
# 导入 PyTorch 库，这是一个流行的开源机器学习库，广泛用于计算机视觉和自然语言处理等深度学习任务。
import torch.nn as nn
# 从 PyTorch 库中导入神经网络模块 torch.nn，并将其重命名为 nn。这个模块包含了构建神经网络所需的类和函数。
from torch.utils.data import DataLoader, Dataset
# 从 PyTorch 的 torch.utils.data 模块导入 DataLoader 和 Dataset。
# DataLoader 用于加载数据集并提供批量数据，Dataset 是所有自定义数据集类的基类。
from src.client.adcol import ADCOLClient
# 从项目的 src.client.adcol 模块导入 ADCOLClient 类。这个类实现了对抗性领域适应（ADCOL）算法的客户端逻辑，用于联邦学习。
from src.server.fedavg import FedAvgServer
# 从项目的 src.server.fedavg 模块导入 FedAvgServer 类。这个类实现了联邦平均（FedAvg）算法的服务器逻辑，用于联邦学习。
from src.utils.tools import NestedNamespace
# 从项目的 src.utils.tools 模块导入 NestedNamespace 类。这个类提供了一个可以处理嵌套字典结构的命令行参数存储解决方案。


# Discriminator 类的目的是创建一个判别器网络，它能够根据输入的特征区分不同的客户端。
# 在 ADCOL 算法中，判别器用于对抗性训练，帮助客户端学习更具代表性的特征，从而提高模型在不同领域上的泛化能力。
class Discriminator(nn.Module):
    # 这段代码定义了一个名为 Discriminator 的类，它是一个 PyTorch nn.Module 的子类，
    # 用于在 ADCOL（对抗性领域适应的联邦学习算法）中进行对抗性训练。以下是对类及其方法的逐行解释：
    # 定义了一个名为 Discriminator 的新类，它继承自 PyTorch 的 nn.Module，这意味着 Discriminator 是一个神经网络模块。
    # discriminator for adversarial training in ADCOL 这是对类作用的注释说明，表明这个判别器用于 ADCOL 中的对抗性训练。
    def __init__(self, base_model, client_num):
        # 构造函数接收两个参数：
        # base_model：基础模型，它是一个预训练的模型，判别器将基于这个模型的特征提取器来构建。
        # client_num：客户端的数量，这个参数决定了判别器输出层的大小，即客户端的类别数。
        super(Discriminator, self).__init__()
        # 调用父类 nn.Module 的构造函数，这是初始化模块的标准做法。
        try:
            in_features = base_model.classifier.in_features
        except:
            raise ValueError("base model has no classifier")
        # 尝试从 base_model 的分类器中获取输入特征的数量。
        # 如果 base_model 没有分类器（即没有 classifier 属性），则抛出一个 ValueError 异常。
        self.discriminator = nn.Sequential(
            nn.Linear(in_features, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, client_num, bias=False),
        )
        # 定义判别器的网络结构，使用 nn.Sequential 来按顺序包含以下层：
        # 第一个全连接层（nn.Linear），将输入特征从 in_features 映射到 512 维，不使用偏置（bias=False）。
        # 批量归一化层（nn.BatchNorm1d），对特征进行归一化处理。
        # ReLU 激活层（nn.ReLU），引入非线性。
        # 第二个全连接层，将 512 维特征再次映射到 512 维，同样不使用偏置。
        # 另一个批量归一化层。
        # 另一个 ReLU 激活层。
        # 最后一个全连接层，将 512 维特征映射到 client_num 维，表示客户端的类别。

    def forward(self, x):
        # 定义了 forward 方法，它是每个 nn.Module 子类必须实现的方法，用于指定模型的前向传播逻辑。
        x = self.discriminator(x)
        # 将输入 x 通过判别器网络进行前向传播。
        return x
        # 返回判别器网络的输出。


# DiscriminateDataset 类用于创建一个包含特征和索引的数据集，这对于某些需要额外信息（如样本来源索引）的机器学习任务非常有用。
# 例如，在对抗性训练或领域适应的场景中，可能需要根据样本的索引来训练一个判别器网络。
# 通过继承 PyTorch 的 Dataset 类，DiscriminateDataset 可以很容易地与 PyTorch 的数据加载和处理流程集成。
class DiscriminateDataset(Dataset):
    # 这段代码定义了一个名为 DiscriminateDataset 的类，它继承自 PyTorch 的 Dataset 类。
    # 此类用于在机器学习中创建一个数据集，它将特征和索引打包在一起，通常用于训练判别器网络。以下是对类及其方法的逐行解释：
    # 定义了一个名为 DiscriminateDataset 的新类，它继承自 PyTorch 的 Dataset 类。
    # 这表明 DiscriminateDataset 是一个自定义数据集类，可以与 PyTorch 的 DataLoader 配合使用。
    def __init__(self, feature, index):
        # 构造函数接收两个参数：
        # feature：一个包含特征的数据结构，这些特征将被用于训练。
        # index：一个包含索引的数据结构，这些索引通常对应于特征。
        # initiate this class
        self.feature = feature
        self.index = index
        # 在类的实例中保存传入的 feature 和 index。

    def __getitem__(self, idx):
        # 定义了 __getitem__ 方法，它根据索引 idx 返回数据集中的一项。这是 Dataset 类的一个必需方法，用于支持索引访问。
        single_feature = self.feature[idx]
        # 根据索引 idx 从特征列表中获取单个特征。
        single_index = self.index[idx]
        # 根据索引 idx 从索引列表中获取单个索引。
        return single_feature, single_index
        # 返回一个包含单个特征和对应索引的元组。

    def __len__(self):
        # 定义了 __len__ 方法，它返回数据集中的总项数。这是 Dataset 类的一个必需方法，用于确定数据集中有多少项。
        return len(self.index)
        # 返回索引列表的长度，这代表了数据集中的样本数量。


class ADCOLServer(FedAvgServer):
    # 这段代码定义了一个名为 ADCOLServer 的类，它继承自 FedAvgServer 类。
    # ADCOLServer 类实现了一个服务器，用于在 ADCOL（对抗性领域适应的联邦学习算法）中进行模型训练和判别器训练。以下是对类及其方法的逐行解释：
    # 定义了一个名为 ADCOLServer 的新类，它继承自 FedAvgServer 类。
    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        # 定义了一个静态方法 get_hyperparams，用于获取超参数。这个方法接收一个可选参数 args_list，返回一个 Namespace 对象。
        parser = ArgumentParser()
        # 创建一个 ArgumentParser 对象，用于解析命令行参数。
        parser.add_argument("--mu", type=float, default=0.5)
        # 添加一个命令行参数 --mu，它是一个浮点数，默认值为 0.5。
        parser.add_argument(
            "--dis_lr", type=float, default=0.01, help="learning rate for discriminator"
        )
        # 添加一个命令行参数 --dis_lr，它是一个浮点数，用于设置判别器的学习率，默认值为 0.01。
        parser.add_argument(
            "--dis_epoch",
            type=int,
            default=3,
            help="epochs for trainig discriminator. larger dis_epoch is recommende when mu is large",
        )
        # 添加一个命令行参数 --dis_epoch，它是一个整数，用于设置训练判别器的周期数，默认值为 3。
        return parser.parse_args(args_list)
        # 解析命令行参数，并返回一个 Namespace 对象。

    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "ADCOL",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        # 构造函数接收以下参数：
        # args：一个 NestedNamespace 对象，包含算法参数。
        # algo：算法名称，默认为 "ADCOL"。
        # unique_model：布尔值，指示是否为每个客户端使用独特的模型。
        # use_fedavg_client_cls：布尔值，指示是否使用联邦平均客户端类。
        # return_diff：布尔值，指示是否返回模型更新的差异。
        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
        # 调用父类 FedAvgServer 的构造函数。
        self.train_client_num = len(self.train_clients)
        # 设置训练客户端的数量。
        self.discriminator = Discriminator(
            base_model=self.model, client_num=len(self.train_clients)
        )
        # 创建一个 Discriminator 实例，用于训练判别器。
        self.init_trainer(
            ADCOLClient,
            discriminator=deepcopy(self.discriminator),
            client_num=self.client_num,
        )
        # 初始化训练器，使用 ADCOLClient 类和复制的判别器。
        self.feature_dataloader: DataLoader = None
        self.features = {}
        # 初始化特征数据加载器和特征字典。

    def train_one_round(self):
        # 定义了 train_one_round 方法，用于训练一轮。
        client_packages = self.trainer.train()
        # 调用训练器的 train 方法，获取客户端包裹。
        self.features = {}
        self.feature_dataloader = None
        # 重置特征字典和特征数据加载器。
        for cid in self.selected_clients:
            self.features[cid] = client_packages[cid]["features_list"]
        # 从客户端包裹中获取特征列表，并存储在特征字典中。
        self.aggregate(client_packages)
        # 聚合客户端模型更新。
        self.train_and_test_discriminator()
        # 训练和测试判别器。

    def package(self, client_id: int):
        # 定义了 package 方法，用于打包服务器到客户端的包裹。
        server_package = super().package(client_id)
        # 调用父类方法获取基本的服务器包裹。
        server_package["new_discriminator_params"] = deepcopy(
            self.discriminator.state_dict()
        )
        # 将判别器的新参数添加到服务器包裹中。
        return server_package
        # 返回服务器包裹。

    def train_and_test_discriminator(self):
        # 定义了 train_and_test_discriminator 方法，用于训练和测试判别器。
        self.generate_client_index()
        # 生成客户端索引。
        if (self.current_epoch + 1) % self.args.common.test_interval == 0:
            acc_before = self.test_discriminator()
        # 如果当前周期是测试周期，则测试判别器。
        self.train_discriminator()
        # 训练判别器。
        if (self.current_epoch + 1) % self.args.common.test_interval == 0:
            acc_after = self.test_discriminator()
            if self.verbose:
                self.logger.log(
                    f"The accuracy of discriminator: {acc_before*100 :.2f}% -> {acc_after*100 :.2f}%"
                )
        # 再次测试判别器，并记录准确率变化。
        self.discriminator.cpu()
        # 将判别器移回 CPU。

    def train_discriminator(self):
        # 定义了 train_discriminator 方法，用于训练判别器。
        self.discriminator.to(self.device)
        # 将判别器移动到设备上。
        self.discriminator.train()
        # 将判别器设置为训练模式。
        self.discriminator_optimizer = torch.optim.SGD(
            self.discriminator.parameters(), lr=self.args.adcol.dis_lr
        )
        # 创建判别器的优化器。
        loss_func = nn.CrossEntropyLoss().to(self.device)
        # 定义交叉熵损失函数。
        # train discriminator
        for _ in range(self.args.adcol.dis_epoch):
            # 遍历训练周期。
            for x, y in self.feature_dataloader:
                # 遍历特征数据加载器。
                x, y = x.to(self.device), y.to(self.device)
                # 将特征和标签移动到设备上。
                y = y.type(torch.float32)
                # 将标签转换为浮点数类型。
                y_pred = self.discriminator(x)
                # 获取判别器的预测结果。
                loss = loss_func(y_pred, y).mean()
                # 计算损失。
                self.discriminator_optimizer.zero_grad()
                loss.backward()
                self.discriminator_optimizer.step()
                # 执行反向传播和优化器步骤。

    def test_discriminator(self):
        # 定义了 test_discriminator 方法，用于测试判别器。
        # test discriminator
        self.discriminator.to(self.device)
        # 将判别器移动到设备上。
        self.discriminator.eval()
        # 将判别器设置为评估模式。
        if self.feature_dataloader:
            # 如果存在特征数据加载器，则继续测试。
            self.accuracy_list = []
            # 初始化准确率列表。
            for x, y in self.feature_dataloader:
                # 遍历特征数据加载器。
                x, y = x.to(self.device), y.to(self.device)
                # 将特征和标签移动到设备上。
                y_pred = self.discriminator(x)
                # 获取判别器的预测结果。
                y_pred = torch.argmax(y_pred, dim=1)
                # 获取预测结果中概率最高的类别索引。
                y = torch.argmax(y, dim=1)
                # 获取真实标签中概率最高的类别索引。
                correct = torch.sum(y_pred == y).item()
                # 计算正确预测的数量。
                self.accuracy_list.append(correct / self.args.common.batch_size)
                # 计算当前批次的准确率并添加到列表中。
            accuracy = sum(self.accuracy_list) / len(self.accuracy_list)
            # 计算所有批次的平均准确率。
            return accuracy
            # 返回判别器的平均准确率。

    # 这个方法的主要作用是准备判别器的训练数据，并将其封装成一个 DataLoader 对象，以便在训练判别器时使用。
    # 通过这种方式，服务器可以有效地管理和使用来自不同客户端的特征数据进行判别器的训练。
    def generate_client_index(self):
        # 这段代码定义了一个名为 generate_client_index 的方法，用于生成客户端索引并创建判别器训练数据集。
        # 以下是对代码的逐行解释：
        # generate client_index_list by self.features
        client_index_list = []
        feature_list = []
        # 初始化两个空列表，client_index_list 用于存储客户端索引的张量，feature_list 用于存储特征张量。
        for client, feature in self.features.items():
            # 遍历 self.features 字典，该字典存储了来自不同客户端的特征数据。
            feature = torch.cat(feature, 0)
            # 将当前客户端的特征列表 feature 沿着第一个维度（0 维）连接成一个连续的特征张量。
            index_tensor = torch.full(
                (feature.shape[0],), fill_value=client, dtype=torch.int64
            )
            # 为当前客户端生成一个索引张量，使用 torch.full 创建一个与特征张量形状相同的张量，并将所有值填充为客户端的索引 client。
            client_index_list.append(index_tensor)
            feature_list.append(feature)
            # 将生成的索引张量添加到 client_index_list 列表中，将特征张量添加到 feature_list 列表中。
        orgnized_features = torch.cat(feature_list, 0)
        # 将所有客户端的特征张量沿着第一个维度连接成一个完整的特征张量。
        orgnized_client_index = torch.cat(client_index_list).type(torch.int64)
        # 将所有客户端的索引张量沿着第一个维度连接成一个完整的索引张量，并确保其数据类型为 torch.int64。
        targets = torch.zeros(
            (orgnized_client_index.shape[0], len(self.train_clients)), dtype=torch.int64
        )
        # 创建一个目标张量，其形状为 (organzied_client_index的行数, 训练客户端的数量)，初始值为零。
        targets = targets.scatter(
            dim=1,
            index=orgnized_client_index.unsqueeze(-1),
            src=torch.ones((orgnized_client_index.shape[0], 1), dtype=torch.int64),
        ).type(torch.float32)
        # 使用 scatter 方法在目标张量中为每个客户端的索引位置填充 1，其他位置保持 0。然后转换目标张量的数据类型为 torch.float32。
        discriminator_training_dataset = DiscriminateDataset(orgnized_features, targets)
        # 使用 DiscriminateDataset 类创建判别器训练数据集，传入特征张量和目标张量。
        self.feature_dataloader = DataLoader(
            discriminator_training_dataset,
            batch_size=self.args.common.batch_size,
            shuffle=True,
        )
        # 创建一个 DataLoader 对象，用于加载判别器训练数据集，并设置批量大小和混洗数据。
