import json
# 导入 json 模块，用于处理JSON数据格式，允许你轻松地编码和解码JSON数据。
import os
# 导入 os 模块，提供了一种方便的方式来使用操作系统依赖的功能，例如文件路径操作。
import random
# 导入 random 模块，包含用于生成随机数的函数，例如 random.randint() 用于获取一个随机整数。
from argparse import Namespace
# 从 argparse 模块导入 Namespace 类，通常用于存储命令行参数解析的结果。
from collections import OrderedDict
# 从 collections 模块导入 OrderedDict 类，它是一个字典子类，能够记住键的顺序。
from pathlib import Path
# 从 pathlib 模块导入 Path 类，提供了一种面向对象的文件系统路径操作方法。
from typing import Callable, Iterator, Sequence, Union
# 从 typing 模块导入几个用于类型注解的类：
# Callable：注解一个可调用的对象。
# Iterator：注解一个迭代器对象。
# Sequence：注解序列类型，如列表、元组等。
# Union：注解允许多种类型中的一个。
import numpy as np
# 导入 numpy 库，并将其简称为 np。NumPy 是一个用于科学计算的Python库，提供了大量的数学函数和对多维数组的支持。
import pynvml
# 导入 pynvml 模块，用于NVIDIA GPU的管理库，可以获取GPU的状态和性能指标
import torch
# 导入 torch 库，PyTorch是一个开源的机器学习库，广泛用于计算机视觉和自然语言处理等深度学习领域。
from rich.console import Console
# 从 rich 库导入 Console 类，rich 是一个用于在终端输出富文本的库。
from torch.utils.data import DataLoader
# 从 torch.utils.data 模块导入 DataLoader 类，用于创建数据加载器，可以迭代地加载数据集。
from src.utils.constants import DEFAULT_COMMON_ARGS, DEFAULT_PARALLEL_ARGS
# 从项目源代码的 utils.constants 模块导入 DEFAULT_COMMON_ARGS 和 DEFAULT_PARALLEL_ARGS，这些可能是用于设置程序的默认参数。
from src.utils.metrics import Metrics
# 从项目源代码的 utils.metrics 模块导入 Metrics 类，这个类可能用于计算和跟踪机器学习模型的性能指标。


# 通过调用这个函数并传入一个整数种子，可以在不同的运行环境中生成相同的随机数序列，从而确保实验结果的可重复性。
# 这对于调试、比较不同模型和算法的性能非常重要。
def fix_random_seed(seed: int) -> None:
    # 这段代码定义了一个名为 fix_random_seed 的函数，其主要作用是设置随机种子，以确保实验的可重复性。以下是对函数中每个步骤的解释：
    # 定义了 fix_random_seed 函数，它接收一个整数类型的参数 seed，用于设置随机种子。函数返回类型为 None，表示这个函数不返回任何值。
    """Fix the random seed of FL training.

    Args:
        seed (int): Any number you like as the random seed.
    """
    # 函数的文档字符串说明了其作用：固定联邦学习（FL）训练的随机种子。
    os.environ["PYTHONHASHSEED"] = str(seed)
    # 设置环境变量 PYTHONHASHSEED，这可以影响 Python 中字典的哈希随机性。将 seed 转换为字符串并赋值给环境变量。
    random.seed(seed)
    # 为 Python 的 random 模块设置随机种子，这会影响 random 模块生成的随机数。
    np.random.seed(seed)
    # 为 NumPy 的随机数生成器设置随机种子，这会影响 NumPy 生成的随机数。
    torch.random.manual_seed(seed)
    # 为 PyTorch 设置随机种子，这会影响 PyTorch 张量的随机数生成。
    if torch.cuda.is_available():
        # 检查 CUDA（GPU加速）是否可用。
        torch.cuda.empty_cache()
        # 如果 CUDA 可用，清空 CUDA 的缓存，以确保每次运行时 GPU 状态的一致性。
        torch.cuda.manual_seed_all(seed)
        # 为 CUDA 设置随机种子，确保在 GPU 上生成的随机数也是可重复的。
    torch.backends.cudnn.deterministic = True
    # 设置 PyTorch 的后端库 cuDNN（CUDA Deep Neural Network library）为确定性模式，这可以减少训练过程中的随机性。
    torch.backends.cudnn.benchmark = False
    # 关闭 cuDNN 的基准测试模式，因为在确定性模式下，基准测试模式可能会影响结果的一致性。


# 这个函数通过比较不同 GPU 的空闲内存，选择具有最多空闲内存的 GPU 用于运行实验。
# 这有助于确保实验能够在具有足够内存的设备上运行，从而避免内存不足导致的问题。通过这种方式，可以更有效地利用可用的计算资源。
def get_optimal_cuda_device(use_cuda: bool) -> torch.device:
    # 这段代码定义了一个名为 get_optimal_cuda_device 的函数，
    # 其主要作用是动态选择具有最多可用内存的 CUDA 设备（GPU），以便运行联邦学习（FL）实验。以下是对函数中每个步骤的解释：
    # 定义了 get_optimal_cuda_device 函数，它接收一个布尔类型的参数 use_cuda，用于决定是否使用 CUDA。函数返回类型为 torch.device。
    """Dynamically select CUDA device (has the most memory) for running FL
    experiment.

    Args:
        use_cuda (bool): `True` for using CUDA; `False` for using CPU only.

    Returns:
        torch.device: The selected CUDA device.
    """
    if not torch.cuda.is_available() or not use_cuda:
        # 如果 CUDA 不可用或者 use_cuda 参数为 False，则返回 CPU 设备。
        return torch.device("cpu")
        # 返回 CPU 设备。
    pynvml.nvmlInit()
    # 初始化 NVIDIA Management Library（pynvml），这是一个用于管理 NVIDIA GPU 的库
    gpu_memory = []
    # 初始化一个空列表 gpu_memory，用于存储每个 GPU 的可用内存。
    if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        gpu_ids = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
        assert max(gpu_ids) < torch.cuda.device_count()
    else:
        gpu_ids = range(torch.cuda.device_count())
    # 检查环境变量 CUDA_VISIBLE_DEVICES 是否设置，如果设置了，则获取其中定义的 GPU ID 列表；如果没有设置，则使用所有可用的 GPU。
    for i in gpu_ids:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory.append(memory_info.free)
    # 遍历所有 GPU，获取每个 GPU 的内存信息，并记录它们的空闲内存。
    gpu_memory = np.array(gpu_memory)
    # 将 gpu_memory 列表转换为 NumPy 数组，以便进行数值计算。
    best_gpu_id = np.argmax(gpu_memory)
    # 使用 np.argmax 函数找到具有最大空闲内存的 GPU 的索引。
    return torch.device(f"cuda:{best_gpu_id}")
    # 返回具有最大空闲内存的 CUDA 设备。


# 这个 vectorize 函数提供了一种灵活的方式来处理不同类型的输入源，并将其中的张量展平并连接成一个单一的张量。
# 这在机器学习中非常有用，特别是在需要将模型参数或梯度向量化时。通过这种方式，可以方便地对模型参数进行操作，如优化、更新或传输。
def vectorize(
    src: OrderedDict[str, torch.Tensor] | list[torch.Tensor] | torch.nn.Module,
    detach=True,
) -> torch.Tensor:
    # 这段代码定义了一个名为 vectorize 的函数，其目的是将一个或多个张量（tensors）向量化（也称为扁平化或展平）。
    # 以下是对函数中每个步骤的解释：
    # 定义了 vectorize 函数，它接收两个参数：
    # src：可以是 OrderedDict 包含张量、张量的列表、torch.nn.Module 对象或迭代器。
    # detach：布尔值，默认为 True，表示是否调用 .detach() 方法并克隆张量以避免对原始计算图的修改。 函数返回一个展平后的张量。
    """Vectorize(Flatten) and concatenate all tensors in `src`.

    Args:
        `src`: The source of tensors.
        `detach`: Set as `True` to return `tensor.detach().clone()`. Defaults to `True`.

    Returns:
        The vectorized tensor.
    """
    func = (lambda x: x.detach().clone()) if detach else (lambda x: x)
    # 定义了一个 lambda 函数 func，根据 detach 参数的值，选择返回分离并克隆的张量或原始张量。
    if isinstance(src, list):
        # 检查 src 是否为列表类型。
        return torch.cat([func(param).flatten() for param in src])
        # 如果是列表，遍历列表中的每个张量，应用 func 函数，展平张量，并使用 torch.cat 连接所有展平的张量。
    elif isinstance(src, OrderedDict) or isinstance(src, dict):
        # 检查 src 是否为 OrderedDict 或普通字典。
        return torch.cat([func(param).flatten() for param in src.values()])
        # 如果是字典，遍历字典中的值（假设为张量），应用 func 函数，展平张量，并连接所有展平的张量。
    elif isinstance(src, torch.nn.Module):
        # 检查 src 是否为 torch.nn.Module 类型。
        return torch.cat([func(param).flatten() for param in src.state_dict().values()])
        # 如果是 torch.nn.Module 对象，遍历模块的状态字典中的每个参数，应用 func 函数，展平参数，并连接所有展平的参数。
    elif isinstance(src, Iterator):
        # 检查 src 是否为迭代器类型。
        return torch.cat([func(param).flatten() for param in src])
        # 如果是迭代器，遍历迭代器中的每个元素，应用 func 函数，展平元素，并连接所有展平的元素。


# 这个 evaluate_model 函数提供了一个简洁的方式来评估模型的性能，它封装了模型评估的常见步骤，
# 包括设置模型为评估模式、处理数据、计算损失和更新性能指标。通过返回 Metrics 对象，用户可以方便地获取模型在特定数据集上的性能指标。
@torch.no_grad()
# 使用 torch.no_grad() 装饰器来禁用梯度计算，这通常用于评估模型，因为在评估过程中不需要进行梯度更新。
def evalutate_model(
    # 这段代码定义了一个名为 evaluate_model 的函数，用于评估指定的模型在给定数据加载器上的性能，并返回一个包含评估指标的 Metrics 对象。
    # 以下是对函数及其参数和步骤的详细解释：
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion=torch.nn.CrossEntropyLoss(reduction="sum"),
    device=torch.device("cpu"),
) -> Metrics:
    # 定义了 evaluate_model 函数，它接收以下参数：
    # model：要评估的模型，类型为 torch.nn.Module。
    # dataloader：用于提供数据的 DataLoader 对象。
    # criterion：可选参数，用于计算损失的准则，如果不提供，则默认使用 torch.nn.CrossEntropyLoss 且 reduction="sum"。
    # device：可选参数，用于指定计算发生的设备，默认为 CPU。
    # 函数返回一个 Metrics 对象。
    """For evaluating the `model` over `dataloader` and return metrics.

    Args:
        model (torch.nn.Module): Target model.
        dataloader (DataLoader): Target dataloader.
        criterion (optional): The metric criterion. Defaults to torch.nn.CrossEntropyLoss(reduction="sum").
        device (torch.device, optional): The device that holds the computation. Defaults to torch.device("cpu").

    Returns:
        Metrics: The metrics objective.
    """
    model.eval()
    # 将模型设置为评估模式，这会关闭模型中的某些特定层（如 Dropout 和 Batch Normalization）的训练行为。
    model.to(device)
    # 将模型移动到指定的设备上。
    metrics = Metrics()
    # 创建一个 Metrics 对象来存储评估结果。
    for x, y in dataloader:
        # 遍历 dataloader 提供的数据，其中 x 是特征，y 是标签。
        x, y = x.to(device), y.to(device)
        # 将数据和标签移动到指定的设备上。
        logits = model(x)
        # 通过模型传递数据 x，得到未经激活的输出 logits。
        loss = criterion(logits, y).item()
        # 使用指定的损失准则计算 logits 和标签 y 之间的损失，并获取损失的标量值。
        pred = torch.argmax(logits, -1)
        # 计算 logits 的最大概率值对应的预测标签。
        metrics.update(Metrics(loss, pred, y))
        # 更新 Metrics 对象，传入损失、预测和实际标签。
    return metrics
    # 返回包含评估指标的 Metrics 对象。


# parse_args 函数提供了一个集中的方式来处理参数解析，它考虑了默认参数、配置文件和命令行输入，确保了参数的一致性和正确性。
# 通过这种方式，用户可以灵活地配置实验参数，同时保持代码的清晰和可维护性。
def parse_args(
    config_file_args: dict | None,
    method_name: str,
    get_method_args_func: Callable[[Sequence[str] | None], Namespace] | None,
    method_args_list: list[str],
) -> Namespace:
    # 这段代码定义了一个名为 parse_args 的函数，它的作用是从默认参数字典、配置文件和命令行界面（CLI）中提取参数，并生成最终的参数集合。
    # 以下是对函数及其参数和步骤的详细解释：
    #
    # 定义了 parse_args 函数，它接收以下参数：
    # config_file_args：从用户定义的 .yml 文件加载的参数字典，如果没有指定，则为 None。
    # method_name：联邦学习方法（FL method）的名称。
    # get_method_args_func：一个可调用函数，用于解析指定联邦学习方法的特定参数。
    # method_args_list：在 CLI 上为指定的联邦学习方法设置的参数列表。
    # 函数返回一个 NestedNamespace 对象，包含最终的参数。
    """Purge arguments from default args dict, config file and CLI and produce
    the final arguments.
    函数的文档字符串说明了其作用：从默认参数字典、配置文件和 CLI 中清除参数，并生成最终参数。
    Args:
        config_file_args (Union[dict, None]): Argument dictionary loaded from user-defined `.yml` file. `None` for unspecifying.
        method_name (str): The FL method's name.
        get_method_args_func (Union[ Callable[[Union[Sequence[str], None]], Namespace], None ]): The callable function of parsing FL method `method_name`'s spec arguments.
        method_args_list (list[str]): FL method `method_name`'s specified arguments set on CLI.

    Returns:
        NestedNamespace: The final argument namespace.
    """
    ARGS = dict(
        mode="serial", common=DEFAULT_COMMON_ARGS, parallel=DEFAULT_PARALLEL_ARGS
    )
    # 初始化 ARGS 字典，包含模式（默认为 "serial"）、通用参数和并行参数。
    if config_file_args is not None:
        # 如果提供了 config_file_args，则根据配置文件中的参数更新 ARGS。
        if "common" in config_file_args.keys():
            ARGS["common"].update(config_file_args["common"])
        if "parallel" in config_file_args.keys():
            ARGS["parallel"].update(config_file_args["parallel"])
        if "mode" in config_file_args.keys():
            ARGS["mode"] = config_file_args["mode"]

    if get_method_args_func is not None:
        # 如果提供了 get_method_args_func 函数，则获取默认方法参数、配置文件中的方法参数和 CLI 方法参数，并根据优先级合并它们。
        default_method_args = get_method_args_func([]).__dict__
        # 调用 get_method_args_func 函数（不带参数）以获取默认方法参数，并将其转换为字典。
        config_file_method_args = {}
        if config_file_args is not None:
            config_file_method_args = config_file_args.get(method_name, {})
            # 从 config_file_args 中获取与 method_name 对应的参数，如果没有，则为空字典。
        cli_method_args = get_method_args_func(method_args_list).__dict__
        # 调用 get_method_args_func 函数（带参数 method_args_list）以获取 CLI 方法参数，并将其转换为字典。
        # extract arguments set explicitly set in CLI
        for key in default_method_args.keys():
            if default_method_args[key] == cli_method_args[key]:
                cli_method_args.pop(key)
                # 从 CLI 方法参数中移除那些与默认方法参数相同的条目。
        # For the same argument, the value setting priority is CLI > config file > defalut value
        method_args = default_method_args
        for key in default_method_args.keys():
            if key in cli_method_args.keys():
                method_args[key] = cli_method_args[key]
            elif key in config_file_method_args.keys():
                method_args[key] = config_file_method_args[key]
        # 根据优先级（CLI > 配置文件 > 默认值）更新方法参数。
        ARGS[method_name] = method_args
        # 将合并后的方法参数存储在 ARGS 字典中，以 method_name 为键。
    assert ARGS["mode"] in ["serial", "parallel"], f"Unrecongnized mode: {ARGS['mode']}"
    # 确保 ARGS 中的模式是 "serial" 或 "parallel"，如果不是，则断言失败。
    if ARGS["mode"] == "parallel":
        if ARGS["parallel"]["num_workers"] < 2:
            print(
                f"num_workers is less than 2: {ARGS['parallel']['num_workers']} and mode is fallback to serial."
            )
            ARGS["mode"] = "serial"
            del ARGS["parallel"]
            # 如果模式是 "parallel" 且 num_workers 小于 2，则打印消息并回退到 "serial" 模式。
    return NestedNamespace(ARGS)
    # 返回一个 NestedNamespace 对象，包含最终的参数集合。

class Logger:
    # 这段代码定义了一个名为 Logger 的类，旨在解决 rich 库中进度条和日志函数之间的不兼容性问题。以下是对类及其方法的逐行解释：
    # 定义了一个名为 Logger 的新类。
    def __init__(
        self, stdout: Console, enable_log: bool, logfile_path: Union[Path, str]
    ):
        # 构造函数接收以下参数：
        # stdout (Console): rich.console.Console 对象，用于在标准输出（stdout）上打印信息。
        # enable_log (bool): 一个标志，指示日志功能是否激活。
        # logfile_path (Union[Path, str]): 日志文件的路径，可以是 Path 对象或字符串。
        """This class is for solving the incompatibility between the progress
        bar and log function in library `rich`.
        类的文档字符串说明了其目的：解决 rich 库中进度条和日志函数之间的不兼容性。
        Args:
            stdout (Console): The `rich.console.Console` for printing info onto stdout.
            enable_log (bool): Flag indicates whether log function is actived.
            logfile_path (Union[Path, str]): The path of log file.
        """

        self.stdout = stdout
        # 将 stdout 对象保存为实例变量。
        self.logfile_output_stream = None
        # 初始化 logfile_output_stream 为 None，这将用于存储日志文件的输出流。
        self.enable_log = enable_log
        # 根据传入的 enable_log 参数，保存日志功能是否激活的状态。
        if self.enable_log:
            # 如果日志功能被激活（enable_log 为 True），执行以下操作：
            self.logfile_output_stream = open(logfile_path, "w")
            # 打开指定路径的日志文件，并准备写入。
            self.logfile_logger = Console(
                file=self.logfile_output_stream,
                record=True,
                log_path=False,
                log_time=False,
                soft_wrap=True,
                tab_size=4,
            )
            # 创建一个 Console 对象，用于写入日志文件。参数配置如下：
            # file: 设置为刚打开的日志文件的输出流。
            # record: 设置为 True，启用记录功能。
            # log_path: 设置为 False，不在日志中记录路径。
            # log_time: 设置为 False，不在日志中记录时间。
            # soft_wrap: 设置为 True，启用软换行。
            # tab_size: 设置制表符宽度为 4。

    def log(self, *args, **kwargs):
        # 定义了一个 log 方法，用于记录日志信息。它接收任意数量的位置参数和关键字参数。
        self.stdout.log(*args, **kwargs)
        # 使用 stdout 对象的 log 方法将信息记录到标准输出。
        if self.enable_log:
            # 如果日志功能被激活，执行以下操作：
            self.logfile_logger.log(*args, **kwargs)
            # 使用 logfile_logger 对象的 log 方法将相同的信息记录到日志文件。

    def close(self):
        # 定义了一个 close 方法，用于关闭日志文件。
        if self.logfile_output_stream:
            # 如果 logfile_output_stream 不是 None，即日志文件已被打开，执行以下操作：
            self.logfile_output_stream.close()
            # 关闭日志文件的输出流。


# 这个 NestedNamespace 类的目的是提供一个可以处理嵌套字典结构的命令行参数存储解决方案。
# 它允许用户以一种层次化的方式组织和访问参数，同时保持了与 argparse.Namespace 的兼容性。
# 通过重写 __str__ 方法，它还提供了一种方便的方式来将参数的当前状态输出为 JSON 格式的字符串，这在调试和记录参数配置时非常有用。
class NestedNamespace(Namespace):
    # 这段代码定义了一个名为 NestedNamespace 的类，它继承自 argparse.Namespace 类，并添加了对嵌套字典的支持。
    # 以下是对类及其方法的逐行解释：
    # 定义了一个名为 NestedNamespace 的新类，它继承自 argparse.Namespace 类，通常用于存储命令行参数解析的结果。
    def __init__(self, args_dict: dict):
        # 构造函数接收一个字典 args_dict 作为参数，该字典包含要存储的参数。
        super().__init__(
            **{
                key: self._nested_namespace(value) if isinstance(value, dict) else value
                for key, value in args_dict.items()
            }
        )
        # 调用父类 Namespace 的构造函数，使用解包的字典来初始化。
        # 如果字典中的值本身也是一个字典，则使用 _nested_namespace 方法递归地将其转换为 NestedNamespace 实例。

    def _nested_namespace(self, dictionary):
        # 定义了一个私有方法 _nested_namespace，它接收一个字典作为参数。
        return NestedNamespace(dictionary)
        # 如果传入的是字典，则创建一个新的 NestedNamespace 实例并返回。

    def to_dict(self):
        # 定义了一个 to_dict 方法，用于将 NestedNamespace 实例转换回字典。
        return {
            key: (value.to_dict() if isinstance(value, NestedNamespace) else value)
            for key, value in self.__dict__.items()
        }
        # 遍历实例的 __dict__ 属性（包含所有的属性和值），
        # 如果值是 NestedNamespace 类型，则递归调用 to_dict 方法；否则直接使用该值。最终返回一个字典。

    def __str__(self):
        # 定义了 __str__ 方法，用于返回 NestedNamespace 实例的字符串表示。
        return json.dumps(self.to_dict(), indent=4, sort_keys=False)
        # 使用 json.dumps 方法将 to_dict 方法返回的字典转换为格式化的 JSON 字符串，
        # 其中 indent=4 表示美化输出，sort_keys=False 表示不排序键。
