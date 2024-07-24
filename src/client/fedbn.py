from torch.nn import BatchNorm2d
# 从PyTorch的神经网络模块torch.nn中导入BatchNorm2d类。
# 这个类用于创建二维批量归一化层，这是一种在深度学习中常用的层，用于规范化输入数据。
from src.client.fedavg import FedAvgClient
# 从项目源代码的src.client.fedavg模块导入FedAvgClient类。
# 这个类是联邦平均（FedAvg）算法中客户端的基础实现，联邦平均算法是一种流行的联邦学习算法。


class FedBNClient(FedAvgClient):
    # 定义了一个名为FedBNClient的新类，它继承自FedAvgClient类。
    # 这意味着FedBNClient将继承FedAvgClient的所有属性和方法，并可以添加或修改特定的功能。
    def __init__(self, **commons):
        # 这是FedBNClient类的构造函数，它接受任意数量的关键字参数（**commons），这些参数将被传递给父类的构造函数。
        super().__init__(**commons)
        # 调用父类FedAvgClient的构造函数，并将关键字参数**commons传递给它。
        # 这确保了FedBNClient类的实例在初始化时能够正确地设置所有继承的属性。
        self.personal_params_name.extend(
            name for name in self.model.state_dict().keys() if "bn" in name
        )
        # 扩展self.personal_params_name列表，添加模型中所有包含"bn"的参数名称。
        # 这通常用于区分模型中哪些参数是批量归一化层的参数。
        # remove duplicates
        self.personal_params_name = list(set(self.personal_params_name))
        # 由于参数名称可能存在重复，这里使用集合（set）来去除重复的名称，然后将结果转换回列表。
        # 这样，self.personal_params_name中就只包含唯一的参数名称。
