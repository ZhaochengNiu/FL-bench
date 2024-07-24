from collections import OrderedDict
# 从Python标准库的collections模块导入OrderedDict类。OrderedDict是一个字典子类，它记住了元素插入的顺序。
from functools import partial
# 从functools模块导入partial函数。partial函数用于创建一个函数的部分（即已经预设了某些参数的函数），这有助于减少函数调用时的参数数量。
from typing import Optional
# 从typing模块导入Optional类型。在类型注解中，Optional用于表示一个变量可能有值，也可能是None。

import torch
# 导入 PyTorch 库，这是一个广泛使用的开源机器学习库，特别适合处理基于GPU的张量计算。
import torch.nn as nn
# 从 PyTorch 库中导入torch.nn模块，并将其别名设置为nn。torch.nn模块包含构建神经网络所需的类和函数。
import torchvision.models as models
# 从torchvision 库中导入models模块，并将其别名设置为models。torchvision.models模块提供了多种预训练的模型架构。
from torch import Tensor
# 从torch模块中导入Tensor类。Tensor是PyTorch中的基本数据结构，用于表示多维数组。
from src.utils.constants import DATA_SHAPE, INPUT_CHANNELS, NUM_CLASSES
# 从项目源代码中的src/utils/constants.py文件导入DATA_SHAPE、INPUT_CHANNELS和NUM_CLASSES常量。
# 这些常量可能用于定义数据的形状、输入通道数和类别数量。
from src.utils.tools import NestedNamespace
# 从项目源代码中的src/utils/tools.py文件导入NestedNamespace类。
# NestedNamespace可能是一个自定义的工具类，用于处理嵌套的命名空间。


class DecoupledModel(nn.Module):
    # 这段代码定义了一个名为 DecoupledModel 的类，它继承自 PyTorch 的 nn.Module。
    # 这个类设计用来处理一些特定的神经网络模型操作，比如获取特征、检查模型组件等。下面是对代码的逐行解释：
    # 定义了一个名为 DecoupledModel 的类，它继承自 PyTorch 的 nn.Module 类。
    def __init__(self):
        super(DecoupledModel, self).__init__()
        # 类的构造函数，调用了父类 nn.Module 的构造函数。
        self.need_all_features_flag = False
        # 初始化一个标志位，用来指示是否需要收集所有特征。
        self.all_features = []
        # 初始化一个空列表，用来存储所有特征。
        self.base: nn.Module = None
        # 初始化 base 属性，它是一个 nn.Module 类型的变量，但初始值为 None。
        self.classifier: nn.Module = None
        # 初始化 classifier 属性，它也是一个 nn.Module 类型的变量，初始值为 None。
        self.dropout: list[nn.Module] = []
        # 初始化一个空列表，用来存储模型中的所有 Dropout 模块。

    def need_all_features(self):
        # 定义了一个方法 need_all_features，这个方法用来注册一个前向钩子，以便在模型的前向传播过程中收集特征。
        target_modules = [
            module
            for module in self.base.modules()
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)
        ]
        # 创建一个列表，包含所有是 nn.Conv2d 或 nn.Linear 类型的模块。

        def _get_feature_hook_fn(model, input, output):
            # 定义了一个内部函数，这个函数会在前向钩子被触发时调用。
            if self.need_all_features_flag:
                self.all_features.append(output.detach().clone())
            # 如果 need_all_features_flag 为 True，则将输出特征 output 从计算图中分离并克隆，然后添加到 all_features 列表中。

        for module in target_modules:
            module.register_forward_hook(_get_feature_hook_fn)
        # 为 target_modules 列表中的每个模块注册前向钩子 _get_feature_hook_fn。

    def check_and_preprocess(self, args: NestedNamespace):
        # 定义了一个方法 check_and_preprocess，它接收一个参数 args，这个参数是一个 NestedNamespace 类型的对象。
        if self.base is None or self.classifier is None:
            # 检查 base 和 classifier 是否已经被设置，如果没有，则抛出一个运行时错误。
            raise RuntimeError(
                "You need to re-write the base and classifier in your custom model class."
            )
        self.dropout = [
            module
            for module in list(self.base.modules()) + list(self.classifier.modules())
            if isinstance(module, nn.Dropout)
        ]
        # 更新 dropout 列表，包含 base 和 classifier 中所有的 Dropout 模块。
        if args.common.buffers == "global":
            # 如果参数 args.common.buffers 的值为 "global"，则对模型中的 BatchNorm2d 模块进行处理。
            for module in self.modules():
                # 遍历模型中的所有模块。
                if isinstance(module, torch.nn.BatchNorm2d):
                    # 如果模块是 BatchNorm2d 类型，则进行以下操作。
                    buffers_list = list(module.named_buffers())
                    # 获取 BatchNorm2d 模块的所有缓冲区，并将其转换为参数。
                    for name_buffer, buffer in buffers_list:
                        # transform buffer to parameter
                        # for showing out in model.parameters()
                        delattr(module, name_buffer)
                        module.register_parameter(
                            name_buffer,
                            torch.nn.Parameter(buffer.float(), requires_grad=False),
                        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.base(x))
        # 定义了模型的前向传播方法，它首先通过 base 模块，然后将输出传递给 classifier 模块。

    def get_last_features(self, x: Tensor, detach=True) -> Tensor:
        # 定义了一个方法 get_last_features，用来获取模型最后一个 base 模块的特征。
        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.eval()
                # 如果存在 Dropout 模块，则将它们设置为评估模式。

        func = (lambda x: x.detach().clone()) if detach else (lambda x: x)
        # 定义一个 lambda 函数，根据 detach 参数决定是返回分离并克隆的特征还是原始特征。
        out = self.base(x)
        # 通过 base 模块获取特征。

        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.train()
                # 如果存在 Dropout 模块，则将它们设置回训练模式。
        return func(out)
        # 返回处理后的特征。

    def get_all_features(self, x: Tensor) -> Optional[list[Tensor]]:
        # 定义了一个方法 get_all_features，用来获取模型所有层的特征。
        feature_list = None
        # 初始化一个变量 feature_list 用来存储所有特征。
        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.eval()
        # 如果存在 Dropout 模块，则将它们设置为评估模式。
        self.need_all_features_flag = True
        _ = self.base(x)
        self.need_all_features_flag = False
        # 设置 need_all_features_flag 为 True，以便在前向传播过程中收集所有特征，
        # 然后调用 base 模块，最后将标志位设置回 False。
        if len(self.all_features) > 0:
            feature_list = self.all_features
            self.all_features = []
        # 如果收集到了特征，则将它们存储到 feature_list 并清空 all_features 列表。
        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.train()
                # 如果存在 Dropout 模块，则将它们设置回训练模式。
        return feature_list
        # 返回所有特征的列表，如果没有特征被收集，则返回 None。


# CNN used in FedAvg
# 整体来看，FedAvgCNN 类定义了一个适用于多个数据集的通用 CNN 架构，通过传入不同的 dataset 参数来适应不同的输入尺寸和类别数。
# 这种设计使得模型可以灵活地应用于不同的联邦学习场景。
class FedAvgCNN(DecoupledModel):
    # 这段代码定义了一个名为 FedAvgCNN 的类，它继承自之前定义的 DecoupledModel 类。
    # FedAvgCNN 类是一个卷积神经网络（CNN），用于联邦学习中的模型平均（Federated Averaging，简称 FedAvg）。
    # 以下是对类及其方法和属性的逐行解释：
    # 定义了一个名为 FedAvgCNN 的类，继承自 DecoupledModel 类。
    feature_length = {
        "mnist": 1024,
        "medmnistS": 1024,
        "medmnistC": 1024,
        "medmnistA": 1024,
        "covid19": 196736,
        "fmnist": 1024,
        "emnist": 1024,
        "femnist": 1,
        "cifar10": 1600,
        "cinic10": 1600,
        "cifar100": 1600,
        "tiny_imagenet": 3200,
        "celeba": 133824,
        "svhn": 1600,
        "usps": 800,
    }
    # 定义了一个名为 feature_length 的类属性，它是一个字典，包含了不同数据集对应的特征长度。
    # 这些特征长度用于确定网络中全连接层的输入维度。

    def __init__(self, dataset: str):
        # 类的构造函数，接受一个参数 dataset，表示使用的是哪一个数据集。
        super(FedAvgCNN, self).__init__()
        # 调用父类 DecoupledModel 的构造函数。
        self.base = nn.Sequential(
            OrderedDict(
                conv1=nn.Conv2d(INPUT_CHANNELS[dataset], 32, 5),
                activation1=nn.ReLU(),
                pool1=nn.MaxPool2d(2),
                conv2=nn.Conv2d(32, 64, 5),
                activation2=nn.ReLU(),
                pool2=nn.MaxPool2d(2),
                flatten=nn.Flatten(),
                fc1=nn.Linear(self.feature_length[dataset], 512),
                activation3=nn.ReLU(),
            )
        )
        # 初始化 base 属性，它是一个 nn.Sequential 容器，包含了网络的卷积层、激活层、池化层和全连接层。OrderedDict 用于确保层的顺序。
        # conv1、conv2：卷积层，用于提取图像特征。
        # activation1、activation2、activation3：ReLU 激活函数，用于引入非线性。
        # pool1、pool2：最大池化层，用于降低特征维度。
        # flatten：Flatten 层，用于将多维特征图展平为一维特征向量。
        # fc1：第一个全连接层，将特征向量映射到更高维度的空间。
        self.classifier = nn.Linear(512, NUM_CLASSES[dataset])
        # 初始化 classifier 属性，它是一个全连接层，将 base 模块输出的 512 维特征向量映射到数据集中的类别数量（NUM_CLASSES[dataset]）。


# 整体来看，LeNet5 类定义了一个适用于多个数据集的通用 LeNet-5 架构，通过传入不同的 dataset 参数来适应不同的输入尺寸和类别数。
# 这种设计使得模型可以灵活地应用于不同的机器学习场景。
class LeNet5(DecoupledModel):
    # 这段代码定义了一个名为 LeNet5 的类，它继承自 DecoupledModel 类。
    # LeNet5 类实现了 LeNet-5 神经网络架构，这是一种经典的卷积神经网络，最初用于手写数字识别任务。以下是对类及其方法和属性的逐行解释：
    # 定义了一个名为 LeNet5 的类，继承自 DecoupledModel 类。
    feature_length = {
        "mnist": 256,
        "medmnistS": 256,
        "medmnistC": 256,
        "medmnistA": 256,
        "covid19": 49184,
        "fmnist": 256,
        "emnist": 256,
        "femnist": 256,
        "cifar10": 400,
        "cinic10": 400,
        "svhn": 400,
        "cifar100": 400,
        "celeba": 33456,
        "usps": 200,
        "tiny_imagenet": 2704,
    }
    # 定义了一个名为 feature_length 的类属性，它是一个字典，包含了不同数据集对应的特征长度。
    # 这些特征长度用于确定网络中第一个全连接层 fc1 的输入维度。

    def __init__(self, dataset: str) -> None:
        # 类的构造函数，接受一个参数 dataset，表示使用的是哪一个数据集。
        super(LeNet5, self).__init__()
        # 调用父类 DecoupledModel 的构造函数。
        self.base = nn.Sequential(
            OrderedDict(
                conv1=nn.Conv2d(INPUT_CHANNELS[dataset], 6, 5),
                bn1=nn.BatchNorm2d(6),
                activation1=nn.ReLU(),
                pool1=nn.MaxPool2d(2),
                conv2=nn.Conv2d(6, 16, 5),
                bn2=nn.BatchNorm2d(16),
                activation2=nn.ReLU(),
                pool2=nn.MaxPool2d(2),
                flatten=nn.Flatten(),
                fc1=nn.Linear(self.feature_length[dataset], 120),
                activation3=nn.ReLU(),
                fc2=nn.Linear(120, 84),
                activation4=nn.ReLU(),
            )
        )
        # 初始化 base 属性，它是一个 nn.Sequential 容器，包含了网络的卷积层、批量归一化层（BatchNorm）、
        # 激活层、池化层、展平层和全连接层。OrderedDict 用于确保层的顺序。
        # conv1、conv2：卷积层，用于提取图像特征。
        # bn1、bn2：批量归一化层，用于归一化卷积层的输出，有助于加速训练并提高模型稳定性。
        # activation1、activation2、activation3、activation4：ReLU 激活函数，用于引入非线性。
        # pool1、pool2：最大池化层，用于降低特征维度。
        # flatten：Flatten 层，用于将多维特征图展平为一维特征向量。
        # fc1、fc2：全连接层，将特征向量映射到更高维度的空间。
        self.classifier = nn.Linear(84, NUM_CLASSES[dataset])
        # 初始化 classifier 属性，它是一个全连接层，将 base 模块输出的 84 维特征向量映射到数据集中的类别数量（NUM_CLASSES[dataset]）。


class TwoNN(DecoupledModel):
    # 这段代码定义了一个名为 TwoNN 的类，它继承自 DecoupledModel 类。
    # TwoNN 类实现了一个简单的神经网络，通常用于分类任务。以下是对类及其方法和属性的逐行解释：
    # 定义了一个名为 TwoNN 的类，继承自 DecoupledModel 类。
    feature_length = {
        "mnist": 784,
        "medmnistS": 784,
        "medmnistC": 784,
        "medmnistA": 784,
        "fmnist": 784,
        "emnist": 784,
        "femnist": 784,
        "cifar10": 3072,
        "cinic10": 3072,
        "svhn": 3072,
        "cifar100": 3072,
        "usps": 1536,
        "synthetic": DATA_SHAPE["synthetic"],
    }
    # 定义了一个名为 feature_length 的类属性，它是一个字典，包含了不同数据集对应的特征长度。
    # '这些特征长度用于确定网络中第一个全连接层的输入维度。

    def __init__(self, dataset):
        # 类的构造函数，接受一个参数 dataset，表示使用的是哪一个数据集。
        super(TwoNN, self).__init__()
        # 调用父类 DecoupledModel 的构造函数。
        self.base = nn.Sequential(
            nn.Linear(self.feature_length[dataset], 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
        )
        # 初始化 base 属性，它是一个 nn.Sequential 容器，包含了两个全连接层和两个 ReLU 激活函数。
        # 第一个全连接层将输入特征映射到 200 维，第二个全连接层将 200 维特征再次映射到 200 维。
        # self.base = nn.Linear(features_length[dataset], 200)
        self.classifier = nn.Linear(200, NUM_CLASSES[dataset])
        # 初始化 classifier 属性，它是一个全连接层，将 base 模块输出的 200 维特征向量映射到数据集中的类别数量（NUM_CLASSES[dataset]）。

    def need_all_features(self):
        return
        # 这个方法被重写，但在 TwoNN 类中没有实现任何功能。这可能是因为 TwoNN 不需要收集所有特征，或者在当前的实现中不需要额外处理。

    def forward(self, x):
        # 定义了模型的前向传播方法。
        x = torch.flatten(x, start_dim=1)
        # 将输入 x 展平，从第二个维度开始展平（通常是图像的通道维度）。
        x = self.classifier(self.base(x))
        # 通过 base 模块和 classifier 模块进行前向传播。
        return x
        # 返回模型的输出。

    def get_last_features(self, x, detach=True):
        # 定义了一个方法 get_last_features，用于获取模型最后一个全连接层的特征。
        func = (lambda x: x.clone().detach()) if detach else (lambda x: x)
        # 定义一个 lambda 函数，根据 detach 参数决定是返回分离并克隆的特征还是原始特征。
        x = torch.flatten(x, start_dim=1)
        # 将输入 x 展平，从第二个维度开始展平。
        x = self.base(x)
        # 通过 base 模块进行前向传播。
        return func(x)
        # 返回处理后的特征。

    def get_all_features(self, x):
        # 定义了一个方法 get_all_features，但在 TwoNN 类中这个方法抛出了一个运行时错误。
        raise RuntimeError("2NN has 0 Conv layer, so is unable to get all features.")
        # 抛出一个运行时错误，说明 TwoNN 没有卷积层，因此无法获取所有特征。


# 整体来看，AlexNet 类定义了一个可以用于不同数据集的通用 AlexNet 架构，通过传入不同的 dataset 参数来适应不同的类别数。
# 此外，通过选择是否加载预训练权重，用户可以根据需要使用预训练的模型或者随机初始化的模型。
class AlexNet(DecoupledModel):
    # 定义了一个名为 AlexNet 的类，继承自 DecoupledModel 类。
    def __init__(self, dataset):
        # 类的构造函数，接受一个参数 dataset，表示使用的是哪一个数据集。
        super().__init__()
        # 调用父类 DecoupledModel 的构造函数。
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        # 定义了一个变量 pretrained 并设置为 True，用于决定是否加载预训练的参数。如果设置为 False，则不会加载预训练的权重。
        alexnet = models.alexnet(
            weights=models.AlexNet_Weights.DEFAULT if pretrained else None
        )
        # 创建一个 AlexNet 模型实例。如果 pretrained 为 True，则加载默认的预训练权重；如果为 False，则不加载任何权重。
        self.base = alexnet
        # 将创建的 AlexNet 模型实例赋值给 base 属性。在 DecoupledModel 类中，base 通常表示模型的特征提取部分。
        self.classifier = nn.Linear(
            alexnet.classifier[-1].in_features, NUM_CLASSES[dataset]
        )
        # 创建一个新的全连接层 classifier 并赋值给 self.classifier。
        # 这个全连接层的输入特征数量由 AlexNet 原始分类器最后一个层的输入特征数量决定，
        # 输出特征数量由数据集的类别数 NUM_CLASSES[dataset] 决定。
        self.base.classifier[-1] = nn.Identity()
        # 将 AlexNet 模型中最后一个分类器层替换为一个恒等映射层 nn.Identity()。
        # 这样做的目的是去除原始 AlexNet 分类器层的任何影响，使得 self.base 仅作为一个特征提取器。


# 整体来看，SqueezeNet 类定义了一个可以用于不同数据集的通用 SqueezeNet 架构，通过传入不同的 version 和 dataset 参数来适应不同的 SqueezeNet 版本和类别数。
# 此外，通过选择是否加载预训练权重，用户可以根据需要使用预训练的模型或者随机初始化的模型。
class SqueezeNet(DecoupledModel):
    # 这段代码定义了一个名为 SqueezeNet 的类，它继承自 DecoupledModel 类。
    # SqueezeNet 类实现了 SqueezeNet 架构，这是一种高效的卷积神经网络，特别设计用于减少参数数量而保持性能。
    # 以下是对类及其构造函数的逐行解释：
    # 定义了一个名为 SqueezeNet 的类，继承自 DecoupledModel 类。
    def __init__(self, version, dataset):
        # 类的构造函数，接受两个参数：version 表示 SqueezeNet 的版本（"0" 或 "1"），dataset 表示使用的是哪一个数据集。
        super().__init__()
        # 调用父类 DecoupledModel 的构造函数。
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        # 定义了一个变量 pretrained 并设置为 True，用于决定是否加载预训练的参数。如果设置为 False，则不会加载预训练的权重。
        archs = {
            "0": (models.squeezenet1_0, models.SqueezeNet1_0_Weights.DEFAULT),
            "1": (models.squeezenet1_1, models.SqueezeNet1_1_Weights.DEFAULT),
        }
        # 定义了一个字典 archs，用于根据版本号获取对应的 SqueezeNet 模型构造函数和预训练权重。
        squeezenet: models.SqueezeNet = archs[version][0](
            weights=archs[version][1] if pretrained else None
        )
        # 根据传入的 version 创建 SqueezeNet 模型实例。如果 pretrained 为 True，则加载对应的预训练权重；如果为 False，则不加载任何权重。
        self.base = squeezenet.features
        # 将 SqueezeNet 模型的 features 属性赋值给 self.base。在 DecoupledModel 类中，base 通常表示模型的特征提取部分。
        self.classifier = nn.Sequential(
            # 定义一个新的 nn.Sequential 容器，用于构建分类器部分的网络。
            nn.Dropout(),
            # 在分类器中添加一个 Dropout 层，用于正则化，减少过拟合。
            nn.Conv2d(
                squeezenet.classifier[1].in_channels,
                NUM_CLASSES[dataset],
                kernel_size=1,
            ),
            # 添加一个卷积层，将 SqueezeNet 分类器中第二个卷积层的输入通道数映射到数据集的类别数 NUM_CLASSES[dataset]，
            # 卷积核大小设置为 1x1。
            nn.ReLU(True),
            # 添加一个 ReLU 激活层，True 参数表示进行 inplace 操作，即在原地修改输入张量。
            nn.AdaptiveAvgPool2d((1, 1)),
            # 添加一个自适应平均池化层，将特征图的大小调整为 1x1。
            nn.Flatten(),
            # 最后添加一个 Flatten 层，将多维特征图展平为一维特征向量。
        )


# 整体来看，DenseNet 类定义了一个可以用于不同数据集的通用 DenseNet 架构，
# 通过传入不同的 version 和 dataset 参数来适应不同的 DenseNet 版本和类别数。
# 此外，通过选择是否加载预训练权重，用户可以根据需要使用预训练的模型或者随机初始化的模型。
class DenseNet(DecoupledModel):
    # 这段代码定义了一个名为 DenseNet 的类，它继承自 DecoupledModel 类。
    # DenseNet 类实现了 DenseNet（Densely Connected Convolutional Networks）架构，
    # 这是一种使用密集连接模式来构建卷积神经网络的方法，可以有效地提高网络的性能和参数效率。以下是对类及其构造函数的逐行解释：
    # 定义了一个名为 DenseNet 的类，继承自 DecoupledModel 类。
    def __init__(self, version, dataset):
        # 类的构造函数，接受两个参数：version 表示 DenseNet 的版本（如 "121", "161", "169", "201"），
        # dataset 表示使用的是哪一个数据集。
        super().__init__()
        # 调用父类 DecoupledModel 的构造函数。
        archs = {
            "121": (models.densenet121, models.DenseNet121_Weights.DEFAULT),
            "161": (models.densenet161, models.DenseNet161_Weights.DEFAULT),
            "169": (models.densenet169, models.DenseNet169_Weights.DEFAULT),
            "201": (models.densenet201, models.DenseNet201_Weights.DEFAULT),
        }
        # 定义了一个字典 archs，用于根据版本号获取对应的 DenseNet 模型构造函数和预训练权重。
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        # 定义了一个变量 pretrained 并设置为 True，用于决定是否加载预训练的参数。如果设置为 False，则不会加载预训练的权重。
        densenet: models.DenseNet = archs[version][0](
            weights=archs[version][1] if pretrained else None
        )
        # 根据传入的 version 创建 DenseNet 模型实例。如果 pretrained 为 True，则加载对应的预训练权重；如果为 False，则不加载任何权重。
        self.base = densenet
        # 将创建的 DenseNet 模型实例赋值给 self.base。在 DecoupledModel 类中，base 通常表示模型的特征提取部分。
        self.classifier = nn.Linear(
            densenet.classifier.in_features, NUM_CLASSES[dataset]
        )
        # 创建一个新的全连接层 classifier 并赋值给 self.classifier。
        # 这个全连接层的输入特征数量由 DenseNet 原始分类器的输入特征数量决定，输出特征数量由数据集的类别数 NUM_CLASSES[dataset] 决定。
        self.base.classifier = nn.Identity()
        # 将 DenseNet 模型中最后一个分类器层替换为一个恒等映射层 nn.Identity()。
        # 这样做的目的是去除原始 DenseNet 分类器层的任何影响，使得 self.base 仅作为一个特征提取器。


class ResNet(DecoupledModel):
    # 这段代码定义了一个名为 ResNet 的类，它继承自 DecoupledModel 类。
    # ResNet 类实现了 ResNet（残差网络）架构，这是一种在深度学习领域广泛使用的卷积神经网络，特别适用于图像识别任务。
    # 以下是对类及其构造函数的逐行解释：
    # 定义了一个名为 ResNet 的类，继承自 DecoupledModel 类。
    def __init__(self, version, dataset):
        # 类的构造函数，接受两个参数：version 表示 ResNet 的版本（如 "18", "34", "50", "101", "152"），dataset 表示使用的是哪一个数据集。
        super().__init__()
        # 调用父类 DecoupledModel 的构造函数。
        archs = {
            "18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
            "34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
            "50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
            "101": (models.resnet101, models.ResNet101_Weights.DEFAULT),
            "152": (models.resnet152, models.ResNet152_Weights.DEFAULT),
        }
        # 定义了一个字典 archs，用于根据版本号获取对应的 ResNet 模型构造函数和预训练权重。
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        # 定义了一个变量 pretrained 并设置为 True，用于决定是否加载预训练的参数。如果设置为 False，则不会加载预训练的权重。
        resnet: models.ResNet = archs[version][0](
            weights=archs[version][1] if pretrained else None
        )
        # 根据传入的 version 创建 ResNet 模型实例。如果 pretrained 为 True，则加载对应的预训练权重；如果为 False，则不加载任何权重。
        self.base = resnet
        # 将创建的 ResNet 模型实例赋值给 self.base。在 DecoupledModel 类中，base 通常表示模型的特征提取部分。
        self.classifier = nn.Linear(self.base.fc.in_features, NUM_CLASSES[dataset])
        # 将 ResNet 模型中最后一个全连接层（通常用于分类）替换为一个恒等映射层 nn.Identity()。
        # 这样做的目的是去除原始 ResNet 分类器层的任何影响，使得 self.base 仅作为一个特征提取器。
        self.base.fc = nn.Identity()
        # 将 ResNet 模型中最后一个全连接层（通常用于分类）替换为一个恒等映射层 nn.Identity()。
        # 这样做的目的是去除原始 ResNet 分类器层的任何影响，使得 self.base 仅作为一个特征提取器。


class MobileNet(DecoupledModel):
    # 定义了一个名为 MobileNet 的新类，它继承自 DecoupledModel 类。
    # 这表明 MobileNet 类将具备 DecoupledModel 类的所有功能和属性。
    def __init__(self, version, dataset):
        # 这是 MobileNet 类的构造函数，它接受两个参数：version 用于指定 MobileNet 的版本，dataset 用于指定数据集的类型。
        super().__init__()
        # 调用父类 DecoupledModel 的构造函数，这是类的初始化的第一步。
        archs = {
            "2": (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT),
            "3s": (
                models.mobilenet_v3_small,
                models.MobileNet_V3_Small_Weights.DEFAULT,
            ),
            "3l": (
                models.mobilenet_v3_large,
                models.MobileNet_V3_Large_Weights.DEFAULT,
            ),
        }
        # 创建一个字典 archs，其中的键是 MobileNet 版本号的字符串，值是一个元组，包含用于创建相应 MobileNet 模型的函数和预训练权重的类。
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        # 定义一个变量 pretrained 并初始化为 True，表示是否希望加载预训练的模型权重。
        # 如果设置为 False，则模型将不会加载预训练权重，而是使用随机初始化的权重。
        mobilenet = archs[version][0](weights=archs[version][1] if pretrained else None)
        # 根据提供的 version 参数，从 archs 字典中选择相应的 MobileNet 模型构造函数，并创建模型实例。
        # 如果 pretrained 为 True，则传递预训练权重；否则，传递 None。
        self.base = mobilenet
        # 将创建的 MobileNet 模型实例赋值给 self.base 属性。在 DecoupledModel 类中，base 属性通常用于存储模型的特征提取部分。
        self.classifier = nn.Linear(
            mobilenet.classifier[-1].in_features, NUM_CLASSES[dataset]
        )
        # 创建一个新的线性层（全连接层），用作分类器，并将其赋值给 self.classifier 属性。
        # 这个分类器的输入特征数量由 MobileNet 模型最后一个分类器层的输入特征数量决定，
        # 输出特征数量由 dataset 参数对应的类别数 NUM_CLASSES[dataset] 决定。
        self.base.classifier[-1] = nn.Identity()
        # 将 MobileNet 模型中最后一个分类器层替换为一个恒等映射层 nn.Identity()。
        # 这通常是为了将 self.base 作为特征提取器使用，而不是用于最终分类。


class EfficientNet(DecoupledModel):
    # 这段代码定义了一个名为 EfficientNet 的类，继承自 DecoupledModel 类。
    # EfficientNet 类实现了 EfficientNet 架构，这是一种高效的卷积神经网络，
    # 通过复合缩放方法（同时扩展深度、宽度和分辨率）来改进模型性能。以下是对类及其构造函数的逐行解释：
    # 定义了一个名为 EfficientNet 的新类，它继承自 DecoupledModel 类，意味着 EfficientNet 将具有解耦模型的属性和方法。
    def __init__(self, version, dataset):
        # 这是 EfficientNet 类的构造函数，它接受两个参数：version 用于指定 EfficientNet 的版本，dataset 用于指定所使用的数据集。
        super().__init__()
        # 调用父类 DecoupledModel 的构造函数，以便正确初始化继承的属性。
        archs = {
            "0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
            "1": (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT),
            "2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT),
            "3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT),
            "4": (models.efficientnet_b4, models.EfficientNet_B4_Weights.DEFAULT),
            "5": (models.efficientnet_b5, models.EfficientNet_B5_Weights.DEFAULT),
            "6": (models.efficientnet_b6, models.EfficientNet_B6_Weights.DEFAULT),
            "7": (models.efficientnet_b7, models.EfficientNet_B7_Weights.DEFAULT),
        }
        # 定义了一个字典 archs，其中包含不同版本 EfficientNet 模型的构造函数和对应的预训练权重。
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        # 设置一个标志变量 pretrained，用于控制是否加载预训练的模型权重。
        efficientnet: models.EfficientNet = archs[version][0](
            weights=archs[version][1] if pretrained else None
        )
        # 根据提供的 version 参数，使用 archs 字典中相应的构造函数创建一个 EfficientNet 模型实例。
        # 如果 pretrained 为 True，则加载预训练权重；否则，使用 None 表示不加载预训练权重。
        self.base = efficientnet
        # 将创建的 EfficientNet 模型实例赋值给 self.base 属性。在 DecoupledModel 类中，base 属性通常用于存储模型的特征提取部分。
        self.classifier = nn.Linear(
            efficientnet.classifier[-1].in_features, NUM_CLASSES[dataset]
        )
        # 创建一个新的线性层（全连接层），并将其赋值给 self.classifier 属性。
        # 这个分类器层的输入特征数量由 efficientnet 模型最后一个分类器层的输入特征数量决定，
        # 输出特征数量由 dataset 对应的类别数 NUM_CLASSES[dataset] 决定。
        self.base.classifier[-1] = nn.Identity()
        # 将 EfficientNet 模型中最后一个分类器层替换为一个恒等映射层 nn.Identity()。
        # 这通常是为了将 self.base 用作特征提取器，而不是用于最终分类。


class ShuffleNet(DecoupledModel):
    # 这段代码定义了一个名为 ShuffleNet 的类，继承自 DecoupledModel 类。
    # ShuffleNet 类实现了 ShuffleNetV2 架构，这是一种轻量级的卷积神经网络，特别适用于计算资源受限的环境。
    # 以下是对类及其构造函数的逐行解释：
    # 定义了一个名为 ShuffleNet 的新类，它继承自 DecoupledModel 类。
    def __init__(self, version, dataset):
        # 这是 ShuffleNet 类的构造函数，它接受两个参数：version 用于指定 ShuffleNet 的版本，dataset 用于指定所使用的数据集。
        super().__init__()
        # 调用父类 DecoupledModel 的构造函数，以便正确初始化继承的属性。
        archs = {
            "0_5": (
                models.shufflenet_v2_x0_5,
                models.ShuffleNet_V2_X0_5_Weights.DEFAULT,
            ),
            "1_0": (
                models.shufflenet_v2_x1_0,
                models.ShuffleNet_V2_X1_0_Weights.DEFAULT,
            ),
            "1_5": (
                models.shufflenet_v2_x1_5,
                models.ShuffleNet_V2_X1_5_Weights.DEFAULT,
            ),
            "2_0": (
                models.shufflenet_v2_x2_0,
                models.ShuffleNet_V2_X2_0_Weights.DEFAULT,
            ),
        }
        # 定义了一个字典 archs，其中包含不同版本 ShuffleNetV2 模型的构造函数和对应的预训练权重。
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        # 设置一个标志变量 pretrained，用于控制是否加载预训练的模型权重。
        shufflenet: models.ShuffleNetV2 = archs[version][0](
            weights=archs[version][1] if pretrained else None
        )
        # 根据提供的 version 参数，使用 archs 字典中相应的构造函数创建一个 ShuffleNetV2 模型实例。
        # 如果 pretrained 为 True，则加载预训练权重；否则，使用 None 表示不加载预训练权重。
        self.base = shufflenet
        # 将创建的 ShuffleNetV2 模型实例赋值给 self.base 属性。在 DecoupledModel 类中，base 属性通常用于存储模型的特征提取部分。
        self.classifier = nn.Linear(shufflenet.fc.in_features, NUM_CLASSES[dataset])
        # 创建一个新的线性层（全连接层），并将其赋值给 self.classifier 属性。
        # 这个分类器层的输入特征数量由 shufflenet 模型的全连接层 fc 的输入特征数量决定，
        # 输出特征数量由 dataset 对应的类别数 NUM_CLASSES[dataset] 决定。
        self.base.fc = nn.Identity()
        # 将 ShuffleNetV2 模型中最后一个全连接层 fc 替换为一个恒等映射层 nn.Identity()。
        # 这通常是为了将 self.base 用作特征提取器，而不是用于最终分类。


class VGG(DecoupledModel):
    # 这段代码定义了一个名为 VGG 的类，继承自 DecoupledModel 类。
    # VGG 类实现了 VGG（Visual Geometry Group）架构，这是一种经典的卷积神经网络，广泛应用于图像识别任务。以下是对类及其构造函数的逐行解释：
    # 定义了一个名为 VGG 的新类，它继承自 DecoupledModel 类。
    def __init__(self, version, dataset):
        # 这是 VGG 类的构造函数，它接受两个参数：
        # version 用于指定 VGG 网络的版本（如 "11", "13", "16", "19"），dataset 用于指定所使用的数据集。
        super().__init__()
        # 调用父类 DecoupledModel 的构造函数，以便正确初始化继承的属性。
        archs = {
            "11": (models.vgg11, models.VGG11_Weights.DEFAULT),
            "13": (models.vgg13, models.VGG13_Weights.DEFAULT),
            "16": (models.vgg16, models.VGG16_Weights.DEFAULT),
            "19": (models.vgg19, models.VGG19_Weights.DEFAULT),
        }
        # 定义了一个字典 archs，其中包含不同版本 VGG 模型的构造函数和对应的预训练权重。
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        # 设置一个标志变量 pretrained，用于控制是否加载预训练的模型权重。
        vgg: models.VGG = archs[version][0](
            weights=archs[version][1] if pretrained else None
        )
        # 根据提供的 version 参数，使用 archs 字典中相应的构造函数创建一个 VGG 模型实例。
        # 如果 pretrained 为 True，则加载预训练权重；否则，使用 None 表示不加载预训练权重。
        self.base = vgg
        # 将创建的 VGG 模型实例赋值给 self.base 属性。在 DecoupledModel 类中，base 属性通常用于存储模型的特征提取部分。
        self.classifier = nn.Linear(
            vgg.classifier[-1].in_features, NUM_CLASSES[dataset]
        )
        # 创建一个新的线性层（全连接层），并将其赋值给 self.classifier 属性。
        # 这个分类器层的输入特征数量由 vgg 模型的最后一个分类器层的输入特征数量决定，
        # 输出特征数量由 dataset 对应的类别数 NUM_CLASSES[dataset] 决定。
        self.base.classifier[-1] = nn.Identity()
        # 将 VGG 模型中最后一个分类器层替换为一个恒等映射层 nn.Identity()。
        # 这通常是为了将 self.base 用作特征提取器，而不是用于最终分类。


# NOTE: You can build your custom model here.
# What you only need to do is define the architecture in __init__().
# Don't need to consider anything else, which are handled by DecoupledModel well already.
# Run `python *.py -m custom` to use your custom model.
class CustomModel(DecoupledModel):
    def __init__(self, dataset):
        super().__init__()
        # You need to define:
        # 1. self.base (the feature extractor part)
        # 2. self.classifier (normally the final fully connected layer)
        # The default forwarding process is: out = self.classifier(self.base(input))
        pass


MODELS = {
    "custom": CustomModel,
    "lenet5": LeNet5,
    "avgcnn": FedAvgCNN,
    "alex": AlexNet,
    "2nn": TwoNN,
    "squeeze0": partial(SqueezeNet, version="0"),
    "squeeze1": partial(SqueezeNet, version="1"),
    "res18": partial(ResNet, version="18"),
    "res34": partial(ResNet, version="34"),
    "res50": partial(ResNet, version="50"),
    "res101": partial(ResNet, version="101"),
    "res152": partial(ResNet, version="152"),
    "dense121": partial(DenseNet, version="121"),
    "dense161": partial(DenseNet, version="161"),
    "dense169": partial(DenseNet, version="169"),
    "dense201": partial(DenseNet, version="201"),
    "mobile2": partial(MobileNet, version="2"),
    "mobile3s": partial(MobileNet, version="3s"),
    "mobile3l": partial(MobileNet, version="3l"),
    "efficient0": partial(EfficientNet, version="0"),
    "efficient1": partial(EfficientNet, version="1"),
    "efficient2": partial(EfficientNet, version="2"),
    "efficient3": partial(EfficientNet, version="3"),
    "efficient4": partial(EfficientNet, version="4"),
    "efficient5": partial(EfficientNet, version="5"),
    "efficient6": partial(EfficientNet, version="6"),
    "efficient7": partial(EfficientNet, version="7"),
    "shuffle0_5": partial(ShuffleNet, version="0_5"),
    "shuffle1_0": partial(ShuffleNet, version="1_0"),
    "shuffle1_5": partial(ShuffleNet, version="1_5"),
    "shuffle2_0": partial(ShuffleNet, version="2_0"),
    "vgg11": partial(VGG, version="11"),
    "vgg13": partial(VGG, version="13"),
    "vgg16": partial(VGG, version="16"),
    "vgg19": partial(VGG, version="19"),
}
