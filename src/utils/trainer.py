from collections import OrderedDict, deque
# OrderedDict：从collections模块导入OrderedDict类。OrderedDict是字典的一个子类，它记住了元素插入的顺序，这在需要保持键的顺序时非常有用。
# deque：从collections模块导入deque类。deque是“double-ended queue”的缩写，即双端队列，是一个高效的数据结构，支持在两端快速地添加和删除元素。
from typing import Any, Callable
# Any：从typing模块导入Any类型注解，用于表示没有任何类型限制的类型，可以是任何类型。
# Callable：从typing模块导入Callable类型注解，用于注解一个可调用的对象，例如函数、方法或者实现了__call__方法的类实例。
import ray
# 导入ray库，Ray是一个用于构建分布式应用程序的框架，它提供了一种简单的方法来并行处理大量的任务。
import ray.actor
# 从ray库中导入actor模块。在Ray中，actor是一种特殊的对象，用于创建状态ful的并发任务，类似于其他并发模型中的线程或进程。
from src.utils.constants import MODE
# 从项目目录下的src.utils.constants模块导入MODE常量。这通常是一个项目特定的配置或常量，可能用于控制程序的行为模式或运行时参数。


# 这个类的设计允许它在不同的运行模式下（串行或并行）以不同的方式进行训练和测试。
# 在串行模式下，它使用单个客户端实例；在并行模式下，它使用多个通过 Ray 框架管理的客户端实例。
# 这种设计提供了灵活性，使得同一代码库可以用于单线程和多线程环境
class FLbenchTrainer:
    # 这段代码定义了一个名为 FLbenchTrainer 的类，它用于训练和测试联邦学习（Federated Learning，简称 FL）模型。
    # 以下是对类及其构造函数和方法的逐行解释：
    # 定义了一个名为 FLbenchTrainer 的类。

    def __init__(
        self, server, client_cls, mode: str, num_workers: int, init_args: dict
    ):
        # 类的构造函数，接收以下参数：
        # server：服务器对象，可能包含服务器的配置和方法。
        # client_cls：客户端类，定义了客户端的行为。
        # mode：运行模式，可以是串行（"SERIAL"）或并行（"PARALLEL"）。
        # num_workers：工作线程数，用于并行模式。
        # init_args：初始化参数字典，包含创建客户端时需要的参数。
        self.server = server
        # 将传入的 server 对象保存为实例变量。
        self.client_cls = client_cls
        # 将传入的 client_cls 类保存为实例变量。
        self.mode = mode
        # 将传入的 mode 保存为实例变量。
        self.num_workers = num_workers
        # 将传入的 num_workers 保存为实例变量。
        if self.mode == MODE.SERIAL:
            # 检查运行模式是否为串行（"SERIAL"）。
            self.worker = client_cls(**init_args)
            # 如果是串行模式，创建一个客户端实例。
        elif self.mode == MODE.PARALLEL:
            # 检查运行模式是否为并行（"PARALLEL"）。
            ray_client = ray.remote(client_cls).options(
                num_cpus=self.server.args.parallel.num_cpus / self.num_workers,
                num_gpus=self.server.args.parallel.num_gpus / self.num_workers,
            )
            # 为并行模式配置 Ray 客户端，指定每个工作线程的 CPU 和 GPU 数量。
            self.workers: list[ray.actor.ActorHandle] = [
                ray_client.remote(**init_args) for _ in range(self.num_workers)
            ]
            # 创建多个 Ray 客户端实例，并将它们保存为一个列表。
        else:
            raise ValueError(f"Unrecongnized running mode.")
            # 如果运行模式既不是串行也不是并行，抛出一个 ValueError 异常。
        if self.mode == MODE.SERIAL:
            # 再次检查运行模式是否为串行。
            self.train = self._serial_train
            self.test = self._serial_test
            self.exec = self._serial_exec
            # 如果是串行模式，将训练、测试和执行方法分别设置为串行版本的实现。
        else:
            # 如果是并行模式。
            self.train = self._parallel_train
            self.test = self._parallel_test
            self.exec = self._parallel_exec
            # 将训练、测试和执行方法分别设置为并行版本的实现。

    # 这个方法的主要作用是在串行训练模式下，逐个与客户端进行通信，更新模型参数、优化器状态和学习率调度器状态，并记录训练过程中的度量指标。
    # 通过这种方式，FLbenchTrainer 类能够在串行模式下有效地管理客户端的训练过程。
    def _serial_train(self):
        # 这段代码定义了一个名为 _serial_train 的方法，它是 FLbenchTrainer 类的一个私有方法，用于在串行模式下进行客户端的训练。
        # 以下是对这个方法的逐行解释：
        # 定义了一个名为 _serial_train 的方法，它没有接收任何参数。
        client_packages = OrderedDict()
        # 创建一个 OrderedDict 来存储客户端的包裹（packages），这些包裹可能包含训练结果、模型参数等。
        for client_id in self.server.selected_clients:
            # 遍历服务器选择的客户端列表。
            server_package = self.server.package(client_id)
            # 为当前客户端创建一个包裹，这个包裹可能包含模型参数、梯度等信息。
            client_package = self.worker.train(server_package)
            # 使用客户端实例调用 train 方法，传入服务器包裹，并获取客户端训练后的包裹。
            client_packages[client_id] = client_package
            # 将客户端的包裹按照客户端 ID 存储到 client_packages 字典中。
            if self.server.verbose:
                # 检查服务器是否处于详细模式（verbose），如果是，则记录训练信息。
                self.server.logger.log(
                    *client_package["eval_results"]["message"], sep="\n"
                )
                # 记录客户端训练评估结果的消息。
            self.server.client_metrics[client_id][self.server.current_epoch] = (
                client_package["eval_results"]
            )
            # 更新服务器上对应客户端和当前周期的度量指标。
            self.server.clients_personal_model_params[client_id].update(
                client_package["personal_model_params"]
            )
            # 更新服务器上对应客户端的个性化模型参数。
            self.server.client_optimizer_states[client_id].update(
                client_package["optimizer_state"]
            )
            # 更新服务器上对应客户端的优化器状态。
            self.server.client_lr_scheduler_states[client_id].update(
                client_package["lr_scheduler_state"]
            )
            # 更新服务器上对应客户端的学习率调度器状态。
        return client_packages
        # 返回包含所有客户端训练包裹的有序字典。

    # 这个方法的主要作用是在并行训练模式下，使用 Ray 框架同时与多个客户端进行通信和训练。
    # 通过异步执行和等待任务完成，它能够高效地管理多个客户端的训练过程，并更新服务器上的相关状态和度量指标。
    def _parallel_train(self):
        # 这段代码定义了一个名为 _parallel_train 的方法，它是 FLbenchTrainer 类的一个私有方法，用于在并行模式下进行客户端的训练。
        # 以下是对这个方法的逐行解释：
        # 定义了一个名为 _parallel_train 的方法，它没有接收任何参数。
        clients = self.server.selected_clients
        # 获取服务器选择的客户端列表。
        i = 0
        # 初始化一个计数器 i，用于遍历客户端列表。
        futures = []
        # 初始化一个列表 futures，用于存储异步任务的引用。
        idle_workers = deque(range(self.num_workers))
        # 创建一个双端队列 idle_workers，包含从 0 到 self.num_workers - 1 的整数，表示工作线程的索引。
        map = {}
        # 初始化一个字典 map，用于存储异步任务与其对应的客户端 ID 和工作线程 ID 的映射。
        client_packages = OrderedDict()
        # 创建一个 OrderedDict 来存储客户端的包裹（packages）。
        while i < len(clients) or len(futures) > 0:
            # 使用一个 while 循环，直到所有客户端都被分配任务并且所有任务都完成。
            while i < len(clients) and len(idle_workers) > 0:
                # 嵌套的 while 循环，用于分配任务给空闲的工作线程。
                worker_id = idle_workers.popleft()
                # 从 idle_workers 队列的左侧弹出一个元素，作为工作线程的索引。
                server_package = ray.put(self.server.package(clients[i]))
                # 将服务器包裹序列化并准备进行分布式操作。
                future = self.workers[worker_id].train.remote(server_package)
                # 为所选客户端分配任务到对应工作线程，并将任务的异步引用存储在 future 中。
                map[future] = (clients[i], worker_id)
                # 将 future 与对应的客户端 ID 和工作线程 ID 存储在 map 字典中。
                futures.append(future)
                # 将 future 添加到 futures 列表中。
                i += 1
                # 增加计数器 i。
            if len(futures) > 0:
                # 检查是否有未完成的异步任务。
                all_finished, futures = ray.wait(futures)
                # 使用 Ray 的 wait 函数等待异步任务完成，all_finished 包含已经完成的任务，futures 包含未完成的任务。
                for finished in all_finished:
                    # 遍历所有已完成的异步任务。
                    client_id, worker_id = map[finished]
                    # 从 map 字典中获取与完成的任务对应的客户端 ID 和工作线程 ID。
                    client_package = ray.get(finished)
                    # 使用 Ray 的 get 函数获取完成的任务结果。
                    idle_workers.append(worker_id)
                    # 将工作线程 ID 添加回 idle_workers 队列，表示工作线程现在空闲。
                    client_packages[client_id] = client_package
                    # 将客户端的包裹存储到 client_packages 字典中。
                    if self.server.verbose:
                        # 检查服务器是否处于详细模式（verbose），如果是，则记录训练信息。
                        self.server.logger.log(
                            *client_package["eval_results"]["message"], sep="\n"
                        )
                        # 记录客户端训练评估结果的消息。
                    self.server.client_metrics[client_id][self.server.current_epoch] = (
                        client_package["eval_results"]
                    )
                    # 更新服务器上对应客户端和当前周期的度量指标。
                    self.server.clients_personal_model_params[client_id].update(
                        client_package["personal_model_params"]
                    )
                    # 更新服务器上对应客户端的个性化模型参数。
                    self.server.client_optimizer_states[client_id].update(
                        client_package["optimizer_state"]
                    )
                    # 更新服务器上对应客户端的优化器状态。
                    self.server.client_lr_scheduler_states[client_id].update(
                        client_package["lr_scheduler_state"]
                    )
                    # 更新服务器上对应客户端的学习率调度器状态。
        return client_packages
        # 返回包含所有客户端训练包裹的有序字典。

    # 这个方法的主要作用是在串行测试模式下，逐个与客户端进行通信，获取模型在不同阶段和数据分割上的性能度量指标，
    # 并将这些度量指标累加到 results 字典中。这有助于评估模型在训练过程中的性能变化。
    def _serial_test(self, clients: list[int], results: dict):
        # 这段代码定义了一个名为 _serial_test 的方法，它是 FLbenchTrainer 类的一个私有方法，用于在串行模式下对客户端进行测试。
        # 以下是对这个方法的逐行解释：
        # 定义了一个名为 _serial_test 的方法，它接收两个参数：
        # clients：一个整数列表，包含要测试的客户端的 ID。
        # results：一个字典，用于存储测试结果。
        for client_id in clients:
            # 遍历 clients 列表中的每个客户端 ID。
            server_package = self.server.package(client_id)
            # 为当前客户端 ID 创建一个包裹，这个包裹可能包含测试所需的模型参数或其他信息。
            metrics = self.worker.test(server_package)
            # 调用客户端实例的 test 方法，并传入服务器包裹。获取测试后的度量指标。
            for stage in ["before", "after"]:
                # 遍历测试可能涉及的阶段，这里的 "before" 和 "after" 可能指的是模型更新前后的测试。
                for split in ["train", "val", "test"]:
                    # 遍历数据集的不同分割，包括训练集（"train"）、验证集（"val"）和测试集（"test"）。
                    results[stage][split].update(metrics[stage][split])
                    # 使用当前客户端在特定阶段和数据分割的测试度量指标更新 results 字典中相应的条目。

    # 这个方法的主要作用是在并行测试模式下，使用 Ray 框架同时与多个客户端进行通信和测试。
    # 通过异步执行和等待任务完成，它能够高效地管理多个客户端的测试过程，并更新服务器上的相关度量指标。
    def _parallel_test(self, clients: list[int], results: dict):
        # 这段代码定义了一个名为 _parallel_test 的方法，它是 FLbenchTrainer 类的一个私有方法，用于在并行模式下对客户端进行测试。
        # 以下是对这个方法的逐行解释：
        # 定义了一个名为 _parallel_test 的方法，接收两个参数：
        # clients：一个整数列表，包含要测试的客户端的 ID。
        # results：一个字典，用于存储测试结果。
        i = 0
        # 初始化一个计数器 i，用于遍历客户端列表。
        futures = []
        # 初始化一个列表 futures，用于存储异步任务的引用。
        idle_workers = deque(range(self.num_workers))
        # 创建一个双端队列 idle_workers，包含从 0 到 self.num_workers - 1 的整数，表示工作线程的索引。
        map = {}  # {future: (client_id, worker_id)}
        # 初始化一个字典 map，用于存储异步任务与其对应的客户端 ID 和工作线程 ID 的映射。
        while i < len(clients) or len(futures) > 0:
            # 使用一个 while 循环，直到所有客户端都被分配任务并且所有任务都完成。
            while i < len(clients) and len(idle_workers) > 0:
                # 嵌套的 while 循环，用于分配测试任务给空闲的工作线程。
                server_package = ray.put(self.server.package(clients[i]))
                # 将服务器包裹序列化并准备进行分布式操作。
                worker_id = idle_workers.popleft()
                # 从 idle_workers 队列的左侧弹出一个元素，作为工作线程的索引。
                future = self.workers[worker_id].test.remote(server_package)
                # 为所选客户端分配测试任务到对应工作线程，并将任务的异步引用存储在 future 中。
                map[future] = (clients[i], worker_id)
                # 将 future 与对应的客户端 ID 和工作线程 ID 存储在 map 字典中。
                futures.append(future)
                # 将 future 添加到 futures 列表中。
                i += 1
                # 增加计数器 i。
            if len(futures) > 0:
                # 检查是否有未完成的异步任务。
                all_finished, futures = ray.wait(futures)
                # 使用 Ray 的 wait 函数等待异步任务完成，all_finished 包含已经完成的任务，futures 包含未完成的任务。
                for finished in all_finished:
                    # 遍历所有已完成的异步任务。
                    metrics = ray.get(finished)
                    # 使用 Ray 的 get 函数获取完成的任务结果。
                    _, worker_id = map[finished]
                    # 从 map 字典中获取与完成的任务对应的工作线程 ID。
                    idle_workers.append(worker_id)
                    # 将工作线程 ID 添加回 idle_workers 队列，表示工作线程现在空闲。
                    for stage in ["before", "after"]:
                        for split in ["train", "val", "test"]:
                            results[stage][split].update(metrics[stage][split])
                    # 遍历测试可能涉及的阶段和数据分割，使用当前客户端的测试度量指标更新 results 字典中相应的条目。

    def _serial_exec(
        # 这段代码定义了一个名为 _serial_exec 的方法，它是 FLbenchTrainer 类的一个私有方法，用于在串行模式下执行指定的函数。
        # 以下是对这个方法的逐行解释：
        self,
        func_name: str,
        clients: list[int],
        package_func: Callable[[int], dict[str, Any]] = None,
    ):
        # 定义了一个名为 _serial_exec 的方法，接收以下参数：
        # self：类实例的引用。
        # func_name：一个字符串，表示要执行的函数名称。
        # clients：一个整数列表，包含要执行函数的客户端的 ID。
        # package_func：一个可选的可调用对象，用于创建客户端包裹。如果未提供，则默认使用服务器的 package 方法。
        if package_func is None:
            package_func = getattr(self.server, "package")
            # 如果未提供 package_func，则使用 getattr 函数从服务器实例中获取名为 package 的方法。
        client_packages = OrderedDict()
        # 创建一个 OrderedDict，用于存储客户端的包裹。
        for client_id in clients:
            # 遍历 clients 列表中的每个客户端 ID。
            server_package = package_func(client_id)
            # 使用 package_func 为当前客户端 ID 创建一个包裹。
            package = getattr(self.worker, func_name)(server_package)
            # 使用 getattr 函数获取 worker 实例中名为 func_name 的方法，并用 server_package 调用它。
            client_packages[client_id] = package
            # 将返回的包裹存储到 client_packages 字典中，以客户端 ID 为键。
        return client_packages
        # 返回包含所有客户端包裹的有序字典。

    def _parallel_exec(
        self,
        func_name: str,
        clients: list[int],
        package_func: Callable[[int], dict[str, Any]] = None,
    ):
        # 定义了一个名为 _parallel_exec 的方法，接收以下参数：
        # self：类实例的引用。
        # func_name：一个字符串，表示要执行的函数名称。
        # clients：一个整数列表，包含要执行函数的客户端的 ID。
        # package_func：一个可选的可调用对象，用于创建客户端包裹。如果未提供，则默认使用服务器的 package 方法。
        if package_func is None:
            package_func = getattr(self.server, "package")
        # 如果未提供 package_func，则使用 getattr 函数从服务器实例中获取名为 package 的方法。
        client_packages = OrderedDict()
        # 创建一个 OrderedDict，用于存储客户端的包裹。
        i = 0
        # 初始化一个计数器 i，用于遍历客户端列表。
        futures = []
        # 初始化一个列表 futures，用于存储异步任务的引用。
        idle_workers = deque(range(self.num_workers))
        # 创建一个双端队列 idle_workers，包含从 0 到 self.num_workers - 1 的整数，表示工作线程的索引。
        map = {}  # {future: (client_id, worker_id)}
        # 初始化一个字典 map，用于存储异步任务与其对应的客户端 ID 和工作线程 ID 的映射。
        while i < len(clients) or len(futures) > 0:
            # 使用一个 while 循环，直到所有客户端都被分配任务并且所有任务都完成。
            while i < len(clients) and len(idle_workers) > 0:
                # 嵌套的 while 循环，用于分配任务给空闲的工作线程。
                server_package = ray.put(package_func(clients[i]))
                # 使用 package_func 为当前客户端 ID 创建一个包裹，并使用 Ray 的 put 方法序列化。
                worker_id = idle_workers.popleft()
                # 从 idle_workers 队列的左侧弹出一个元素，作为工作线程的索引。
                future = getattr(self.workers[worker_id], func_name).remote(
                    server_package
                )
                # 使用 getattr 获取工作线程实例中名为 func_name 的方法，并用 server_package 调用它。
                # 将任务的异步引用存储在 future 中。
                map[future] = (clients[i], worker_id)
                # 将 future 与对应的客户端 ID 和工作线程 ID 存储在 map 字典中。
                futures.append(future)
                # 将 future 添加到 futures 列表中。
                i += 1
                # 增加计数器 i。
            if len(futures) > 0:
                # 检查是否有未完成的异步任务。
                all_finished, futures = ray.wait(futures)
                # 使用 Ray 的 wait 函数等待异步任务完成，all_finished 包含已经完成的任务，futures 包含未完成的任务。
                for finished in all_finished:
                    # 遍历所有已完成的异步任务。
                    package = ray.get(finished)
                    # 使用 Ray 的 get 函数获取完成的任务结果。
                    client_id, worker_id = map[finished]
                    # 从 map 字典中获取与完成的任务对应的客户端 ID 和工作线程 ID。
                    idle_workers.append(worker_id)
                    # 将工作线程 ID 添加回 idle_workers 队列，表示工作线程现在空闲。
                    client_packages[client_id] = package
                    # 将返回的包裹存储到 client_packages 字典中，以客户端 ID 为键。
        return client_packages
        # 返回包含所有客户端包裹的有序字典。
