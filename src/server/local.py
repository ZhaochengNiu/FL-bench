from src.server.fedavg import FedAvgServer
from src.utils.tools import NestedNamespace
# FedAvgServer 类用于实现联邦平均算法的服务器端逻辑，
# NestedNamespace 类用于处理命令行参数。

class LocalServer(FedAvgServer):
    # 这段代码定义了一个名为 LocalServer 的类，继承自 FedAvgServer 类。
    # LocalServer 类实现了一个服务器端逻辑，用于处理本地训练的联邦学习场景。以下是对类及其方法的逐行解释：
    # 定义 LocalServer 类，继承自 FedAvgServer 类。
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "Local-only",
        unique_model=True,
        use_fedavg_client_cls=True,
        return_diff=False,
    ):
        # 构造函数接收以下参数：
        # args：一个 NestedNamespace 对象，包含算法参数。
        # algo：算法名称，默认为 "Local-only"。
        # unique_model：布尔值，指示是否为每个客户端使用独特的模型，默认为 True。
        # use_fedavg_client_cls：布尔值，指示是否使用联邦平均客户端类，默认为 True。
        # return_diff：布尔值，指示是否返回模型更新的差异，默认为 False。
        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
        # 调用父类 FedAvgServer 的构造函数。

    # LocalServer 类扩展了 FedAvgServer 类的功能，以适应本地训练的场景。在这种设置中，每个客户端可能有一个独特的模型副本，
    # 服务器负责聚合这些本地更新的模型参数。这种个性化的方法可能有助于提高模型在不同客户端上的泛化能力。
    def train_one_round(self):
        # 定义 train_one_round 方法，用于训练一轮。
        client_packages = self.trainer.train()
        # 调用训练器的 train 方法来开始训练过程。这可能涉及在客户端上进行本地训练。
        for client_id, package in client_packages.items():
            # 遍历训练器返回的客户端包裹。client_packages 是一个字典，其中包含客户端 ID 和对应的训练结果包裹。
            self.clients_personal_model_params[client_id].update(
                package["regular_model_params"]
            )
            # 对于每个客户端，更新服务器上存储的客户端个性化模型参数。regular_model_params 可能是基础模型参数的更新。
            self.clients_personal_model_params[client_id].update(
                package["personal_model_params"]
            )
            # 进一步更新客户端个性化模型参数，这可能包含特定于客户端的额外个性化。
