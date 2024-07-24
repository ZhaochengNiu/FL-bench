import importlib
# importlib 提供了动态导入模块的功能，即可以在运行时导入一个模块。
import os
# os 模块提供了许多与操作系统交互的功能，比如文件和目录的操作、路径操作等。
import sys
# sys 模块提供了访问和与 Python 解释器进行交互的功能，比如访问命令行参数、标准输入输出、模块路径等。
import inspect
# inspect 模块提供了获取实时对象信息的功能，比如获取模块、类、方法、函数、回溯、帧对象和代码对象的信息。
from pathlib import Path
# Path 类提供了一种面向对象的文件系统路径操作方式，比 os.path 更加直观和方便。
import yaml
# yaml 模块用于处理 YAML 文件，YAML 是一种数据序列化格式，常用于配置文件和数据交换。
import pynvml
# pynvml 是 NVIDIA 管理库（NVML）的 Python 绑定，用于监控和管理 NVIDIA GPU 设备的状态和信息。

from src.server.fedavg import FedAvgServer
# FedAvgServer 类可能是实现联邦平均算法的服务器端逻辑。

FLBENCH_ROOT = Path(__file__).parent.absolute()
# 获取当前脚本文件所在目录的绝对路径，并将其赋值给 FLBENCH_ROOT 变量。
# __file__ 是当前脚本的路径，parent 获取其父目录，absolute() 转换为绝对路径。
# 这样可以确保 FLBENCH_ROOT 是一个绝对路径。
if FLBENCH_ROOT not in sys.path:
    sys.path.append(FLBENCH_ROOT.as_posix())
    # 功能：检查 FLBENCH_ROOT 是否在 sys.path 中，如果不在，则将其添加进去。
    # 作用：sys.path 是一个包含解释器查找模块路径的列表，通过将 FLBENCH_ROOT 添加到 sys.path 中，可以确保在该路径下的模块可以被导入。
    # as_posix() 方法将路径对象转换为 POSIX 风格的字符串路径。

from src.utils.tools import parse_args
# parse_args 函数可能用于解析命令行参数或配置文件参数，这对于配置和启动程序是很常见的。


# 这段代码的主要目的是解析命令行参数和配置文件参数，初始化服务器，然后根据参数启动服务器。
# 它还处理了分布式计算的初始化，特别是当使用 Ray 进行并行计算时。
if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 检查命令行参数的数量。sys.argv 是一个包含命令行参数的列表。
        raise RuntimeError(
            "No method is specified. Run like `python main.py <method> [config_file_relative_path] [cli_method_args ...]`,",
            "e.g., python main.py fedavg config/template.yml`",
        )
    # 如果参数数量少于2，引发一个 RuntimeError 异常，并提供如何运行脚本的示例。
    method_name = sys.argv[1]
    # 获取第一个命令行参数，即方法名称。
    config_file_path = None
    cli_method_args = []
    # 初始化变量，用于存储配置文件路径和命令行方法参数。
    if len(sys.argv) > 2:
        # 检查是否有更多的命令行参数。
        if ".yaml" in sys.argv[2] or ".yml" in sys.argv[2]:  # ***.yml or ***.yaml
            # 检查第二个参数是否包含 ".yaml" 或 ".yml" 后缀，这通常表示一个YAML配置文件。
            config_file_path = Path(sys.argv[2]).absolute()
            cli_method_args = sys.argv[3:]
            # 如果第二个参数是配置文件，则获取其绝对路径，并获取其余的命令行参数。
        else:
            cli_method_args = sys.argv[2:]
            # 如果第二个参数不是配置文件，则将所有剩余的参数视为命令行方法参数。
    try:
        fl_method_server_module = importlib.import_module(f"src.server.{method_name}")
        # 尝试导入名为 src.server.{method_name} 的模块，其中 {method_name} 是从命令行参数中获取的方法名称。
    except:
        raise ImportError(f"Can't import `src.server.{method_name}`.")
    # 如果导入失败，引发一个 ImportError 异常。
    module_attributes = inspect.getmembers(fl_method_server_module)
    # 获取模块的所有属性。
    server_class = [
        attribute
        for attribute in module_attributes
        if attribute[0].lower() == method_name + "server"
    ][0][1]
    # 在模块属性中查找以方法名称加 "server" 后缀命名的类，并获取这个类。
    get_method_hyperparams_func = getattr(server_class, f"get_hyperparams", None)
    # 尝试获取 server_class 中名为 get_hyperparams 的方法，如果不存在则返回 None。
    config_file_args = None
    # 初始化变量，用于存储从配置文件中读取的参数。
    if config_file_path is not None and os.path.isfile(config_file_path):
        # 检查配置文件路径是否存在且为文件。
        with open(config_file_path, "r") as f:
            try:
                config_file_args = yaml.safe_load(f)
            except:
                raise TypeError(
                    f"Config file's type should be yaml, now is {config_file_path}"
                )
        # 读取配置文件并尝试将其内容解析为YAML格式。如果失败，引发一个 TypeError 异常。
    ARGS = parse_args(
        config_file_args, method_name, get_method_hyperparams_func, cli_method_args
    )
    # 调用 parse_args 函数来解析命令行参数和配置文件参数，并将结果存储在 ARGS 中。
    # target method is not inherited from FedAvgServer
    if server_class.__bases__[0] != FedAvgServer and server_class != FedAvgServer:
        # 检查 server_class 是否继承自 FedAvgServer。
        parent_server_class = server_class.__bases__[0]
        # 获取父类。
        if hasattr(parent_server_class, "get_hyperparams"):
            get_parent_method_hyperparams_func = getattr(
                parent_server_class, f"get_hyperparams", None
            )
            # 尝试获取父类中的 get_hyperparams 方法。
            # class name: ***Server, only want ***
            parent_method_name = parent_server_class.__name__.lower()[:-6]
            # 获取父方法的名称，去掉 "Server" 后缀。
            # extract the hyperparams of parent method
            PARENT_ARGS = parse_args(
                config_file_args,
                parent_method_name,
                get_parent_method_hyperparams_func,
                cli_method_args,
            )
            # 调用 parse_args 函数来解析父类的参数。
            setattr(ARGS, parent_method_name, getattr(PARENT_ARGS, parent_method_name))
            # 将父类的参数设置到 ARGS 中。
    if ARGS.mode == "parallel":
        # 检查 ARGS 中的模式是否为 "parallel"。
        import ray
        # 导入 Ray 库，用于分布式计算。
        num_available_gpus = ARGS.parallel.num_gpus
        num_available_cpus = ARGS.parallel.num_cpus
        # 获取 ARGS 中指定的 GPU 和 CPU 数量。
        if num_available_gpus is None:
            # 检查是否未指定 GPU 数量。
            pynvml.nvmlInit()
            # 初始化 NVML 库。
            num_total_gpus = pynvml.nvmlDeviceGetCount()
            # 获取总 GPU 数量。
            if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
                # 检查环境变量 CUDA_VISIBLE_DEVICES 是否设置。
                num_available_gpus = min(
                    len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")), num_total_gpus
                )
                # 根据 CUDA_VISIBLE_DEVICES 环境变量设置可用的 GPU 数量。
            else:
                num_available_gpus = num_total_gpus
                # 否则，使用所有 GPU。
        if num_available_cpus is None:
            num_available_cpus = os.cpu_count()
            # 如果未指定 CPU 数量，则使用所有可用的 CPU。
        try:
            ray.init(
                address=ARGS.parallel.ray_cluster_addr,
                namespace=method_name,
                num_cpus=num_available_cpus,
                num_gpus=num_available_gpus,
                ignore_reinit_error=True,
            )
            # 初始化 Ray，设置地址、命名空间、CPU 和 GPU 数量。
        except ValueError:
            # 捕获 ValueError 异常。
            # have existing cluster
            # then no pass num_cpus and num_gpus
            ray.init(
                address=ARGS.parallel.ray_cluster_addr,
                namespace=method_name,
                ignore_reinit_error=True,
            )
            # 如果已经存在 Ray 集群，则重新初始化 Ray，但不指定 CPU 和 GPU 数量。
        cluster_resources = ray.cluster_resources()
        # 获取 Ray 集群的资源信息。
        ARGS.parallel.num_cpus = cluster_resources["CPU"]
        ARGS.parallel.num_gpus = cluster_resources["GPU"]
        # 更新 ARGS 中的 CPU 和 GPU 数量。
    server = server_class(args=ARGS)
    # 创建服务器实例。
    server.run()
    # 启动服务器。
