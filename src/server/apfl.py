from argparse import ArgumentParser, Namespace
# 从 argparse 模块导入 ArgumentParser 和 Namespace 类。ArgumentParser 用于解析命令行参数，而 Namespace 用于存储解析后的参数。
from copy import deepcopy
# 从 copy 模块导入 deepcopy 函数，该函数用于创建对象的深拷贝，这对于复制包含可变类型的对象（如列表或字典）非常有用。
import torch
# 导入 PyTorch 库，这是一个流行的开源机器学习库，广泛用于计算机视觉和自然语言处理等深度学习任务。
from src.client.apfl import APFLClient
# 从项目的 src.client.apfl 模块导入 APFLClient 类。这个类可能是实现了某种特定客户端逻辑的联邦学习客户端。
from src.server.fedavg import FedAvgServer
# 从项目的 src.server.fedavg 模块导入 FedAvgServer 类。这个类实现了联邦平均（FedAvg）算法的服务器逻辑，用于联邦学习。
from src.utils.tools import NestedNamespace
# 从项目的 src.utils.tools 模块导入 NestedNamespace 类。
# 这个类可能是一个工具类，用于处理或存储参数，可能与 argparse.Namespace 有关但提供了额外的功能或便利性。


class APFLServer(FedAvgServer):
    # 这段代码定义了一个名为 APFLServer 的类，继承自 FedAvgServer 类。
    # APFLServer 类实现了异步个性化联邦学习（Asynchronous Personalized Federated Learning，简称 APFL）算法的服务器端逻辑。
    # 以下是对类及其方法的逐行解释：
    # 定义了一个名为 APFLServer 的新类，它继承自 FedAvgServer 类。
    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        # 定义了一个静态方法 get_hyperparams，用于获取超参数。这个方法接收一个可选参数 args_list，返回一个 Namespace 对象。
        parser = ArgumentParser()
        # 创建一个 ArgumentParser 对象，用于解析命令行参数。
        parser.add_argument("--alpha", type=float, default=0.5)
        # 添加一个命令行参数 --alpha，它是一个浮点数，默认值为 0.5。
        parser.add_argument("--adaptive_alpha", type=int, default=1)
        # 添加一个命令行参数 --adaptive_alpha，它是一个整数，默认值为 1，表示是否使用自适应的 alpha 值。
        return parser.parse_args(args_list)
        # 解析命令行参数，并返回一个 Namespace 对象。

    def __init__(

        self,
        args: NestedNamespace,
        algo: str = "APFL",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        # 构造函数接收以下参数：
        # args：一个 NestedNamespace 对象，包含算法参数。
        # algo：算法名称，默认为 "APFL"。
        # unique_model：布尔值，指示是否为每个客户端使用独特的模型。
        # use_fedavg_client_cls：布尔值，指示是否使用联邦平均客户端类。
        # return_diff：布尔值，指示是否返回模型更新的差异。

        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
        # 调用父类 FedAvgServer 的构造函数。
        self.init_trainer(APFLClient)
        # 初始化训练器，使用 APFLClient 类。
        self.client_local_model_params = {
            i: deepcopy(self.model.state_dict()) for i in self.train_clients
        }
        # 为每个训练客户端创建本地模型参数的深拷贝。
        self.client_alphas = {
            i: torch.tensor(self.args.apfl.alpha) for i in self.train_clients
        }
        # 为每个训练客户端设置 alpha 值。

    def package(self, client_id: int):
        # 定义了 package 方法，用于打包服务器到客户端的包裹。
        server_package = super().package(client_id)
        # 调用父类方法获取基本的服务器包裹。
        server_package["alpha"] = self.client_alphas[client_id]
        # 将客户端的 alpha 值添加到服务器包裹中。
        server_package["local_model_params"] = self.client_local_model_params[client_id]
        # 将客户端的本地模型参数添加到服务器包裹中。
        return server_package
        # 返回服务器包裹。

    def train_one_round(self):
        # 定义了一个名为 train_one_round 的方法，该方法是 APFLServer 类的一部分，用于执行一轮训练过程。
        client_packages = self.trainer.train()
        # 调用 trainer 对象的 train 方法来开始训练过程。self.trainer 可能是一个负责管理客户端训练的 Trainer 类的实例。
        # 此方法可能涉及异步地在多个客户端上进行训练，并收集训练结果。
        for client_id in self.selected_clients:
            # 遍历 self.selected_clients 中的每个客户端 ID。
            # self.selected_clients 是一个列表，包含了当前训练轮次中被选中参与训练的客户端 ID。
            self.client_local_model_params[client_id] = client_packages[client_id][
                "local_model_params"
            ]
            # 对于每个客户端 ID，从 client_packages 字典中获取该客户端的本地模型参数 "local_model_params"，
            # 并更新 self.client_local_model_params 字典中对应的条目。这将保留每个客户端最新的本地模型参数。
            self.client_alphas[client_id] = client_packages[client_id]["alpha"]
            # 同样地，从 client_packages 字典中获取每个客户端的 "alpha" 值，并更新 self.client_alphas 字典中对应的条目。
            # alpha 值可能用于控制每个客户端在联邦学习中的个性化学习率。
        self.aggregate(client_packages)
        # 调用 aggregate 方法来合并或聚合所有客户端的训练结果。这个方法使用 client_packages 字典作为输入，
        # 该字典包含了所有客户端的训练输出。聚合过程可能涉及计算模型参数的平均值或应用其他联邦学习算法特定的逻辑。
