from torch.nn import BatchNorm2d, Conv2d, Linear
# 从 PyTorch 的神经网络模块导入 BatchNorm2d（二维批量归一化层）、Conv2d（二维卷积层）和 Linear（线性层）。
from src.client.fedavg import FedAvgClient
# 从项目源代码的 src.client.fedavg 模块导入 FedAvgClient 类。


# 这段代码定义了一个名为 LGFedAvgClient 的类，它继承自 FedAvgClient 类，并为联邦学习中的本地化广义联邦平均（LGFedAvg）算法实现客户端逻辑。
# 以下是对类及其构造函数的逐行解释：
class LGFedAvgClient(FedAvgClient):
    # 定义 LGFedAvgClient 类，继承自 FedAvgClient 类。
    def __init__(self, **commons):
        # 构造函数接收任意数量的关键字参数 commons。
        super().__init__(**commons)
        # 调用父类 FedAvgClient 的构造函数。
        self.personal_params_name = []
        # 初始化一个空列表 self.personal_params_name，用于存储客户端个性化参数的名称。
        trainable_layers = [
            (name, module)
            for name, module in self.model.named_modules()
            if isinstance(module, Conv2d)
            or isinstance(module, BatchNorm2d)
            or isinstance(module, Linear)
        ]
        # 创建一个列表 trainable_layers，包含模型中所有可训练层的名称和模块。可训练层包括二维卷积层、二维批量归一化层和线性层。
        personal_layers = trainable_layers[self.args.lgfedavg.num_global_layers :]
        # 根据 self.args.lgfedavg.num_global_layers 的值，
        # 获取 trainable_layers 中从该索引开始之后的所有层，这些层被视为客户端个性化层。
        for module_name, module in personal_layers:
            for param_name, _ in module.named_parameters():
                self.personal_params_name.append(f"{module_name}.{param_name}")
        # 遍历客户端个性化层，将每个层的参数名称添加到 self.personal_params_name 列表中。
        self.init_personal_params_dict = {
            name: param.clone().detach()
            for name, param in self.model.state_dict(keep_vars=True).items()
            if (not param.requires_grad) or (name in self.personal_params_name)
        }
        # 创建一个字典 self.init_personal_params_dict，包含客户端个性化参数的初始值。
        # 这个字典是通过克隆并分离（clone().detach()）模型状态字典中的参数来初始化的，
        # 其中只包含不需要梯度的参数或在 self.personal_params_name 列表中的参数。
