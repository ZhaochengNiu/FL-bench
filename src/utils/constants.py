import json
# 导入 json 模块，用于处理JSON数据格式，允许你轻松地编码和解码JSON数据。
import os
# 导入 os 模块，提供了一种方便的方式来使用操作系统依赖的功能，例如文件路径操作。
from enum import Enum
# 从 enum 模块导入 Enum 类，用于创建枚举类，这可以帮助你管理一组相关的常数。
from pathlib import Path
# 从 pathlib 模块导入 Path 类，提供了一种面向对象的文件系统路径操作方法。
from torch import optim
# 从 torch 库导入 optim 模块，它包含PyTorch的优化器，如SGD、Adam等，用于模型训练过程中更新参数。
FLBENCH_ROOT = Path(__file__).parent.parent.parent.absolute()
# 设置 FLBENCH_ROOT 变量，其值为当前脚本文件所在目录的上三级目录的绝对路径。这通常用于确定项目的根目录。
OUT_DIR = FLBENCH_ROOT / "out"
# 设置 OUT_DIR 变量，其值为 FLBENCH_ROOT 下的 "out" 子目录的路径。这通常用于存储输出文件或日志。
TEMP_DIR = FLBENCH_ROOT / "temp"
# 设置 TEMP_DIR 变量，其值为 FLBENCH_ROOT 下的 "temp" 子目录的路径。这通常用于存储临时文件。


class MODE(Enum):
    # 这段代码定义了一个名为 MODE 的枚举类，它继承自 Python 标准库中的 Enum 类。
    # 枚举类是一种特殊的类，用于为一组相关常数定义一个共同的类型，并提供命名管理。以下是对代码的逐行解释：
    # 定义了一个名为 MODE 的枚举类。
    SERIAL = 0
    # 在 MODE 枚举类中定义了一个名为 SERIAL 的成员，其值为 0。这表示串行模式，可能用于单线程或单进程的执行环境。
    PARALLEL = 1
    # 在 MODE 枚举类中定义了一个名为 PARALLEL 的成员，其值为 1。这表示并行模式，可能用于多线程或多进程的执行环境。

    # 枚举类的好处包括：
    # 类型安全：确保只能使用预定义的值。
    # 可读性：使用名称而不是数字或字符串，提高代码的可读性。
    # 易于维护：集中管理一组相关的常数，便于维护和更新。
    # 使用枚举类可以让代码更加清晰和易于理解，特别是在处理具有预定义选项的设置时。
    # 在这个例子中，MODE 枚举类用于控制程序是应该以串行还是并行的方式运行。

# 这个字典提供了一套完整的默认配置，可以用于启动和运行实验，同时也可以根据需要对这些参数进行修改和扩展。
# 通过集中管理这些参数，可以方便地进行实验配置和复现。
DEFAULT_COMMON_ARGS = {
    "dataset": "mnist",
    # "dataset": "mnist": 指定使用的默认数据集名称，这里是 "mnist"。
    "seed": 42,
    # "seed": 42: 设置随机种子为 42，用于确保实验的可重复性。
    "model": "lenet5",
    # "model": "lenet5": 指定使用的默认模型，这里是 "lenet5"。
    "join_ratio": 0.1,
    # "join_ratio": 0.1: 指定客户端加入通信的比率。
    "global_epoch": 100,
    # "global_epoch": 100: 指定全局训练周期数。
    "local_epoch": 5,
    # "local_epoch": 5: 指定每个客户端在每轮联邦学习中的本地训练周期数。
    "finetune_epoch": 0,
    # "finetune_epoch": 0: 指定微调周期数，这里设置为 0，表示不进行微调。
    "batch_size": 32,
    # "batch_size": 32: 指定每个客户端训练时的批次大小。
    "test_interval": 100,
    # "test_interval": 100: 指定测试间隔，即每多少个全局周期进行一次测试。
    "straggler_ratio": 0,
    # "straggler_ratio": 0: 指定落后者（训练速度慢的客户端）的比率。
    "straggler_min_local_epoch": 0,
    # "straggler_min_local_epoch": 0: 指定落后者客户端的最小本地周期数。
    "external_model_params_file": None,
    # "external_model_params_file": None: 指定外部模型参数文件的路径，这里为 None，表示不使用外部模型参数。
    "buffers": "local",
    # "buffers": "local": 指定缓冲区的存储位置，"local" 表示在本地存储。
    "optimizer": {
        "name": "sgd",  # [sgd, adam, adamw, rmsprop, adagrad]
        # "name": "sgd": 指定优化器的名称，这里是 "sgd"。
        "lr": 0.01,
        # "lr": 0.01: 指定学习率。
        "dampening": 0,  # SGD,
        # "dampening": 0: 指定 SGD 优化器的阻尼项。
        "weight_decay": 0,
        # "weight_decay": 0: 指定权重衰减。
        "momentum": 0,  # SGD, RMSprop,
        # "momentum": 0: 指定 SGD 和 RMSprop 优化器的动量。
        "alpha": 0.99,  # RMSprop,
        # "alpha": 0.99: 指定 RMSprop 优化器的 alpha 参数。
        "nesterov": False,  # SGD,
        # "nesterov": False: 指定是否使用 Nesterov 动量。
        "betas": [0.9, 0.999],  # Adam, AdamW,
        # "betas": [0.9, 0.999]: 指定 Adam 和 AdamW 优化器的 beta 参数。
        "amsgrad": False,  # Adam, AdamW
        # "amsgrad": False: 指定是否使用 Adam 优化器的 AMSGrad 变体。
    },
    # "optimizer": {...}: 包含优化器配置的字典，有多个键用于配置不同类型的优化器：
    "eval_test": True,
    # "eval_test": True: 指定是否在测试集上进行评估。
    "eval_val": False,
    # "eval_val": False: 指定是否在验证集上进行评估。
    "eval_train": False,
    # "eval_train": False: 指定是否在训练集上进行评估。
    "verbose_gap": 10,
    # "verbose_gap": 10: 指定详细输出的信息间隔。
    "visible": False,
    # "visible": False: 指定是否显示详细的调试信息。
    "use_cuda": True,
    # "use_cuda": True: 指定是否使用 CUDA（GPU加速）。
    "save_log": True,
    # "save_log": True: 指定是否保存日志。
    "save_model": False,
    # "save_model": False: 指定是否保存模型参数。
    "save_fig": True,
    # "save_fig": True: 指定是否保存图表。
    "save_metrics": True,
    # "save_metrics": True: 指定是否保存度量指标。
    "delete_useless_run": True,
    # "delete_useless_run": True: 指定是否删除无用的运行结果。
}

# 这些默认参数为并行计算提供了一个基本的配置模板，可以根据实际的硬件资源和计算需求进行调整。
# 例如，如果你在一个有多个 GPU 的高性能计算环境中，你可能会设置 "num_gpus" 和 "num_cpus" 以充分利用这些资源。
# 同样，根据任务的并行度和资源的可用性，"num_workers" 也可以相应地增加或减少。
DEFAULT_PARALLEL_ARGS = {
    # 这段代码定义了一个名为 DEFAULT_PARALLEL_ARGS 的字典，它包含了用于配置并行计算环境的默认参数。
    # 这些参数通常用于设置分布式计算资源，如使用 Ray 这类分布式框架时的配置。以下是对每个参数的解释：
    "ray_cluster_addr": None,
    # "ray_cluster_addr": None: 指定 Ray 集群的地址。如果设置为 None，则 Ray 将在本地启动并尝试发现其他节点。
    # 在分布式环境中，你可能需要指定集群的地址以连接到正确的集群。
    "num_gpus": None,
    # "num_gpus": None: 指定每个工作节点可分配的 GPU 数量。如果设置为 None，则表示不限制 GPU 的使用，或者在当前环境中不使用 GPU。
    "num_cpus": None,
    # "num_cpus": None: 指定每个工作节点可分配的 CPU 核心数量。如果设置为 None，则表示不限制 CPU 核心的使用。
    "num_workers": 2,
    # "num_workers": 2: 指定工作节点（或工作进程）的数量。在这个默认配置中，设置为 2 表示将有两个工作节点参与并行计算。
}

# 这个字典为不同的数据集提供了一个统一的接口来获取它们的输入通道数，这在构建神经网络模型时非常有用，因为输入层的配置需要根据图像的通道数来确定。
# 通过使用 INPUT_CHANNELS 字典，可以方便地为不同的数据集创建适当配置的模型。
INPUT_CHANNELS = {
    "mnist": 1,
    # "mnist": 1: MNIST 数据集是灰度图像，因此输入通道数为 1。
    "medmnistS": 1,
    "medmnistC": 1,
    "medmnistA": 1,
    # "medmnistS": 1, "medmnistC": 1, "medmnistA": 1: 这些数据集也是灰度图像，每个像素点只有一个颜色值，输入通道数同样为 1。
    "covid19": 3,
    # "covid19": 3: COVID-19 数据集包含彩色图像，因此输入通道数为 3（红色、绿色、蓝色）。
    "fmnist": 1,
    # "fmnist": 1: Fashion-MNIST 数据集是灰度图像，输入通道数为 1。
    "emnist": 1,
    # "emnist": 1: EMNIST 数据集是扩展的 MNIST 数据集，也是灰度图像，输入通道数为 1。
    "femnist": 1,
    # "femnist": 1: FEMNIST 数据集同样是灰度图像，输入通道数为 1。
    "cifar10": 3,
    # "cifar10": 3: CIFAR-10 数据集包含彩色图像，输入通道数为 3。
    "cinic10": 3,
    # "cinic10": 3: CINIC-10 数据集也是彩色图像，输入通道数为 3。
    "svhn": 3,
    # "svhn": 3: SVHN 数据集是彩色图像，输入通道数为 3。
    "cifar100": 3,
    # "cifar100": 3: CIFAR-100 数据集包含彩色图像，输入通道数为 3。
    "celeba": 3,
    # "celeba": 3: CelebA 数据集包含彩色图像，输入通道数为 3。
    "usps": 1,
    # "usps": 1: USPS 数据集是灰度图像，输入通道数为 1。
    "tiny_imagenet": 3,
    # "tiny_imagenet": 3: Tiny ImageNet 数据集包含彩色图像，输入通道数为 3。
    "domain": 3,
    # "domain": 3: 领域泛化数据集，这里假设为彩色图像，输入通道数为 3。
}


# 这两个函数的目的是简化从特定数据集目录中加载配置参数的过程。通过检查文件的存在性并加载 JSON 数据，它们使得配置数据的获取变得简单和直接。
# 如果相应的配置文件不存在，函数会返回一个空字典，这可以作为默认行为，避免程序因缺少配置而出错。
# 这种模式在处理可选配置时非常有用，提供了灵活性和健壮性。
def _get_domainnet_args():
    if os.path.isfile(FLBENCH_ROOT / "data" / "domain" / "metadata.json"):
        # 检查在 FLBENCH_ROOT 下的 data/domain 目录中是否存在名为 metadata.json 的文件。
        with open(FLBENCH_ROOT / "data" / "domain" / "metadata.json", "r") as f:
            # 检查在 FLBENCH_ROOT 下的 data/domain 目录中是否存在名为 metadata.json 的文件。
            metadata = json.load(f)
            # 检查在 FLBENCH_ROOT 下的 data/domain 目录中是否存在名为 metadata.json 的文件。
        return metadata
        # 返回加载的 metadata 数据。
    else:
        return {}
        # 如果 metadata.json 文件不存在，则返回一个空字典。


def _get_synthetic_args():
    if os.path.isfile(FLBENCH_ROOT / "data" / "synthetic" / "args.json"):
        # 检查在 FLBENCH_ROOT 下的 data/synthetic 目录中是否存在名为 args.json 的文件。
        with open(FLBENCH_ROOT / "data" / "synthetic" / "args.json", "r") as f:
            # 如果文件存在，使用 with 语句打开文件，以只读模式（"r"）
            metadata = json.load(f)
            # 使用 json.load 函数从文件中加载 JSON 数据，并将其存储在变量 metadata 中。
        return metadata
        # 返回加载的 metadata 数据。
    else:
        return {}
        # 如果 args.json 文件不存在，则返回一个空字典。


# 这段代码定义了一个名为 DATA_SHAPE 的字典，它映射了不同数据集名称到对应的数据形状。
# 数据形状通常指图像数据的通道数、高度和宽度，或者在某些情况下，仅指特征维度。以下是对字典中每个条目的解释：
# (C, H, W)
DATA_SHAPE = {
    "mnist": (1, 28, 28),
    # "mnist": (1, 28, 28): MNIST 数据集的图像是灰度图，具有 1 个通道和 28x28 像素的尺寸。
    "medmnistS": (1, 28, 28),
    "medmnistC": (1, 28, 28),
    "medmnistA": (1, 28, 28),
    # "medmnistS": (1, 28, 28), "medmnistC": (1, 28, 28), "medmnistA": (1, 28, 28): 这些数据集的图像同样是灰度图，具有与 MNIST 相同的尺寸。
    "fmnist": (1, 28, 28),
    # "fmnist": (1, 28, 28): Fashion-MNIST 数据集也是灰度图，尺寸与 MNIST 相同。
    "svhn": (3, 32, 32),
    # "svhn": (3, 32, 32): SVHN 数据集的图像是彩色图，具有 3 个通道和 32x32 像素的尺寸。
    "emnist": 62,
    # "emnist": 62: EMNIST 数据集的特征维度是 62，这里没有提供图像尺寸，可能是因为它是文本数据。
    "femnist": 62,
    # "femnist": 62: FEMNIST 数据集的特征维度与 EMNIST 相同。
    "cifar10": (3, 32, 32),
    # "cifar10": (3, 32, 32): CIFAR-10 数据集的图像是彩色图，具有 3 个通道和 32x32 像素的尺寸。
    "cinic10": (3, 32, 32),
    # "cinic10": (3, 32, 32): CINIC-10 数据集与 CIFAR-10 具有相同的图像尺寸和通道数。
    "cifar100": (3, 32, 32),
    # "cifar100": (3, 32, 32): CIFAR-100 数据集与 CIFAR-10 和 CINIC-10 具有相同的图像尺寸和通道数。
    "covid19": (3, 244, 224),
    # "covid19": (3, 244, 224): COVID-19 数据集的图像是彩色图，具有 3 个通道和 244x224 像素的尺寸。
    "usps": (1, 16, 16),
    # "usps": (1, 16, 16): USPS 数据集的图像是灰度图，具有 1 个通道和 16x16 像素的尺寸。
    "celeba": (3, 218, 178),
    # "celeba": (3, 218, 178): CelebA 数据集的图像是彩色图，具有 3 个通道和 218x178 像素的尺寸。
    "tiny_imagenet": (3, 64, 64),
    # "tiny_imagenet": (3, 64, 64): Tiny ImageNet 数据集的图像是彩色图，具有 3 个通道和 64x64 像素的尺寸。
    "synthetic": _get_synthetic_args().get("dimension", 0),
    # Synthetic 数据集的形状通过调用 _get_synthetic_args 函数并尝试获取 "dimension" 键的值来确定。如果没有提供，则默认为 0。
    "domain": (3, *(_get_domainnet_args().get("image_size", (0, 0)))),
    # Domain 数据集的形状通过调用 _get_domainnet_args 函数并尝试获取 "image_size" 键的值来确定。
    # 这个值应该是一个元组，表示图像的高度和宽度。如果未提供，则默认为 (0, 0)。注意，这里使用了星号 * 来解包元组，
    # 以便将其作为独立的参数传递给元组。
}

# 这段代码定义了一个名为 NUM_CLASSES 的字典，用于存储不同数据集的类别数量。
# 这对于构建和配置机器学习模型，特别是分类模型时，确定输出层的神经元数量非常重要。以下是对字典中每个条目的详细解释：
NUM_CLASSES = {
    "mnist": 10,
    # "mnist": 10：MNIST 数据集包含10个类别，对应于手写数字0到9。
    "medmnistS": 11,
    "medmnistC": 11,
    "medmnistA": 11,
    # "medmnistS": 11，"medmnistC": 11，"medmnistA": 11：这些医学MNIST变体数据集包含11个类别，可能对应于不同的医学图像类别。
    "fmnist": 10,
    # "fmnist": 10：Fashion-MNIST 数据集包含10个类别，代表不同的时尚物品类别。
    "svhn": 10,
    # "svhn": 10：Street View House Numbers (SVHN) 数据集包含10个数字类别，与MNIST类似，但图像更复杂。
    "emnist": 62,
    # "emnist": 62：扩展的MNIST (EMNIST) 数据集包含62个字母类别，包括大写和小写英文字母。
    "femnist": 62,
    # "femnist": 62：女性手写数字 (FEMNIST) 数据集与 EMNIST 类似，也包含62个类别。
    "cifar10": 10,
    # "cifar10": 10：CIFAR-10 数据集包含10个类别，代表不同的动物和车辆。
    "cinic10": 10,
    # "cinic10": 10：CINIC-10 数据集与 CIFAR-10 类似，也包含10个类别。
    "cifar100": 100,
    # "cifar100": 100：CIFAR-100 数据集包含100个类别，是 CIFAR-10 的扩展版本，包含更多类别。
    "covid19": 4,
    # "covid19": 4：COVID-19 数据集在这个上下文中可能包含4个类别，可能对应于不同的COVID-19相关的图像类别。
    "usps": 10,
    # "usps": 10：USPS 数据集包含10个类别，代表手写数字0到9。
    "celeba": 2,
    # "celeba": 2：CelebA 数据集在这个上下文中可能被用作二分类问题，例如区分两个不同的属性或类别。
    "tiny_imagenet": 200,
    # "tiny_imagenet": 200：Tiny ImageNet 数据集包含200个类别，是 ImageNet 的一个较小版本，用于图像分类任务。
    "synthetic": _get_synthetic_args().get("class_num", 0),
    # Synthetic 数据集的类别数量通过调用 _get_synthetic_args 函数并尝试获取 "class_num" 键的值来确定。如果没有提供，则默认为0。
    "domain": _get_domainnet_args().get("class_num", 0),
    # Domain 数据集的类别数量通过调用 _get_domainnet_args 函数并尝试获取 "class_num" 键的值来确定。如果没有提供，则默认为0。
}

# 这段代码定义了一个名为 DATA_MEAN 的字典，它为不同的数据集提供了平均值（均值）数据。
# 在图像处理和机器学习中，数据的均值通常用于数据预处理步骤，比如用于图像的归一化。以下是对字典中每个条目的解释：
# 这些均值数据通常用于图像数据的预处理，通过从每个通道中减去这些均值，可以使数据的分布更加接近标准正态分布，有助于模型的训练和收敛。
# 每个数据集的均值可能是基于其训练集统计得出的。使用这些预处理步骤可以提高模型在不同数据集上的性能和泛化能力。
DATA_MEAN = {
    "mnist": [0.1307],
    # "mnist": [0.1307]：MNIST 数据集的均值为 0.1307，这是所有像素值的平均值，由于是灰度图像，因此只有一个值。
    "cifar10": [0.4914, 0.4822, 0.4465],
    # "cifar10": [0.4914, 0.4822, 0.4465]：CIFAR-10 数据集的均值是一个列表，包含三个颜色通道（红、绿、蓝）的平均值。
    "cifar100": [0.5071, 0.4865, 0.4409],
    # "cifar100": [0.5071, 0.4865, 0.4409]：CIFAR-100 数据集的均值，与 CIFAR-10 类似，也是三个颜色通道的平均值。
    "emnist": [0.1736],
    # "emnist": [0.1736]：EMNIST 数据集的均值，这是灰度图像数据集的均值。
    "fmnist": [0.286],
    # "fmnist": [0.286]：Fashion-MNIST 数据集的均值。
    "femnist": [0.9637],
    # "femnist": [0.9637]：FEMNIST 数据集的均值。
    "medmnist": [124.9587],
    # "medmnist": [124.9587]：Medical MNIST 数据集的均值，这里只有一个值，可能因为数据集是灰度图像。
    "medmnistA": [118.7546],
    "medmnistC": [124.424],
    # "medmnistA": [118.7546] 和 "medmnistC": [124.424]：这些是 Medical MNIST 数据集的不同变体的均值。
    "covid19": [125.0866, 125.1043, 125.1088],
    # "covid19": [125.0866, 125.1043, 125.1088]：COVID-19 数据集的均值，包含三个颜色通道的平均值。
    "celeba": [128.7247, 108.0617, 97.2517],
    # "celeba": [128.7247, 108.0617, 97.2517]：CelebA 数据集的均值，包含三个颜色通道的平均值。
    "synthetic": [0.0],
    # "synthetic": [0.0]：合成数据集的均值为 0.0，这可能表明合成数据在生成时已经居中归一化。
    "svhn": [0.4377, 0.4438, 0.4728],
    # "svhn": [0.4377, 0.4438, 0.4728]：SVHN 数据集的均值，包含三个颜色通道的平均值。
    "tiny_imagenet": [122.5119, 114.2915, 101.388],
    # "tiny_imagenet": [122.5119, 114.2915, 101.388]：Tiny ImageNet 数据集的均值，包含三个颜色通道的平均值。
    "cinic10": [0.47889522, 0.47227842, 0.43047404],
    # "cinic10": [0.47889522, 0.47227842, 0.43047404]：CINIC-10 数据集的均值，包含三个颜色通道的平均值。
    "domain": [0.485, 0.456, 0.406],
    # "domain": [0.485, 0.456, 0.406]：领域泛化数据集的均值，包含三个颜色通道的平均值。
}

# 这段代码定义了一个名为 DATA_STD 的字典，它为不同的数据集提供了标准差（Standard Deviation，STD）数据。在图像处理和机器学习中，
# 数据的标准差通常与均值一起用于数据预处理步骤，如归一化，以便将数据缩放到一个统一的尺度。以下是对字典中每个条目的解释：
# 使用这些标准差数据，可以对图像数据进行标准化处理，通常的公式为：(x - mean) / std。
# 这种标准化有助于让模型更好地学习，因为输入数据的分布更加统一，通常接近标准正态分布。每个数据集的标准差可能是基于其训练集统计得出的。
DATA_STD = {
    "mnist": [0.3015],
    # MNIST 数据集的标准差为 0.3015，这是所有像素值的标准差，由于是灰度图像，因此只有一个值。
    "cifar10": [0.2023, 0.1994, 0.201],
    # CIFAR-10 数据集的标准差是一个列表，包含三个颜色通道（红、绿、蓝）的标准差。
    "cifar100": [0.2009, 0.1984, 0.2023],
    # CIFAR-100 数据集的标准差，与 CIFAR-10 类似，也是三个颜色通道的标准差。
    "emnist": [0.3248],
    # EMNIST 数据集的标准差，这是灰度图像数据集的标准差。
    "fmnist": [0.3205],
    # Fashion-MNIST 数据集的标准差。
    "femnist": [0.155],
    # FEMNIST 数据集的标准差。
    "medmnist": [57.5856],
    "medmnistA": [62.3489],
    "medmnistC": [58.8092],
    # "medmnistA": [62.3489] 和 "medmnistC": [58.8092]：这些是 Medical MNIST 数据集的不同变体的标准差。
    "covid19": [56.6888, 56.6933, 56.6979],
    # COVID-19 数据集的标准差，包含三个颜色通道的标准差。
    "celeba": [67.6496, 62.2519, 61.163],
    # CelebA 数据集的标准差，包含三个颜色通道的标准差。
    "synthetic": [1.0],
    # 合成数据集的标准差为 1.0，这可能表明合成数据在生成时已经标准化。
    "svhn": [0.1201, 0.1231, 0.1052],
    # SVHN 数据集的标准差，包含三个颜色通道的标准差。
    "tiny_imagenet": [58.7048, 57.7551, 57.6717],
    # Tiny ImageNet 数据集的标准差，包含三个颜色通道的标准差。
    "cinic10": [0.24205776, 0.23828046, 0.25874835],
    # CINIC-10 数据集的标准差，包含三个颜色通道的标准差。
    "domain": [0.229, 0.224, 0.225],
    # 领域泛化数据集的标准差，包含三个颜色通道的标准差。
}

# 这段代码定义了一个名为 OPTIMIZERS 的字典，它映射了不同的优化器名称到 PyTorch 的 optim 模块中对应的优化器类。
# 以下是对字典中每个条目的解释：
OPTIMIZERS = {
    "sgd": optim.SGD,
    # "sgd": optim.SGD：SGD 是随机梯度下降（Stochastic Gradient Descent）的缩写，这是一种常用的优化算法，适用于各种机器学习任务。
    # 在 PyTorch 中，optim.SGD 是实现 SGD 优化器的类。
    "adam": optim.Adam,
    # "adam": optim.Adam：Adam 是一种自适应学习率优化算法，它结合了 RMSprop 和 Momentum 的优点。
    # optim.Adam 是 PyTorch 中实现 Adam 优化器的类。
    "adamw": optim.AdamW,
    # "adamw": optim.AdamW：AdamW 是 Adam 优化器的一个变体，它在原始 Adam 算法的基础上对权重进行了 L2 正则化。
    # 在 PyTorch 中，optim.AdamW 是实现 AdamW 优化器的类。
    "rmsprop": optim.RMSprop,
    # "rmsprop": optim.RMSprop：RMSprop 是一种自适应学习率优化算法，它使用指数衰减平均值来调整学习率。
    # optim.RMSprop 是 PyTorch 中实现 RMSprop 优化器的类。
    "adagrad": optim.Adagrad,
    # "adagrad": optim.Adagrad：Adagrad 是一种为每个参数自适应调整学习率的优化算法，它适用于处理稀疏数据。
    # optim.Adagrad 是 PyTorch 中实现 Adagrad 优化器的类。
}


# 这段代码定义了一个名为 LR_SCHEDULERS 的字典，
# 它映射了不同的学习率调度器名称到 PyTorch 的 optim.lr_scheduler 模块中对应的学习率调度器类。以下是对字典中每个条目的解释：
LR_SCHEDULERS = {
    "step": optim.lr_scheduler.StepLR,
    # 步进学习率调度器（StepLR）在每个给定的步数后将学习率降低一定的因子。这有助于在训练过程中逐渐减小学习率，从而提高模型的收敛精度。
    "cosine": optim.lr_scheduler.CosineAnnealingLR,
    # 余弦退火学习率调度器（CosineAnnealingLR）根据余弦退火策略调整学习率，其变化类似于余弦函数。
    # 这种调度器可以在训练的后期逐渐减小学习率，有助于模型在接近最优解时更细致地搜索参数空间。
    "constant": optim.lr_scheduler.ConstantLR,
    # 恒定学习率调度器（ConstantLR）保持学习率在整个训练过程中不变。这在某些情况下可以提供稳定的训练效果。
    "plateau": optim.lr_scheduler.ReduceLROnPlateau,
    # 当验证集的性能停止提升时，平台学习率调度器（ReduceLROnPlateau）会降低学习率。
    # 这种方法可以帮助模型在训练过程中避免过拟合，并在性能不再提升时通过减小学习率来细化搜索。
}
