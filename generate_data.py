import json
# 导入 json 模块，用于处理JSON（JavaScript Object Notation）数据格式，允许你轻松地编码和解码JSON数据。
import os
# 导入 os 模块，提供了一种方便的方式来使用操作系统依赖的功能，例如文件路径操作。
import pickle
# 导入 pickle 模块，用于序列化和反序列化Python对象，可以将对象“pickle”（序列化）到文件中，也可以从文件中“unpickle”（反序列化）。
import random
# 导入 random 模块，包含用于生成随机数的函数，例如 random.randint() 用于获取一个随机整数。
from copy import deepcopy
# 从 copy 模块中导入 deepcopy 函数，用于创建对象的深拷贝，这意味着递归复制对象中的所有内容，而不仅仅是对象本身。
from collections import Counter
# 从 collections 模块中导入 Counter 类，它是一个字典子类，用于计数可哈希对象，通常用于统计数据集中元素的出现次数。
from argparse import ArgumentParser
# 从 argparse 模块中导入 ArgumentParser 类，用于解析命令行参数和选项。
from pathlib import Path
# 从 pathlib 模块中导入 Path 类，提供了一种面向对象的文件系统路径操作方法。
import numpy as np
# 导入 numpy 库，并将其简称为 np。NumPy 是一个用于科学计算的Python库，提供了大量的数学函数和对多维数组的支持。
from src.utils.tools import fix_random_seed
# 从 src.utils.tools 模块中导入 fix_random_seed 函数。这个函数可能用于固定随机种子，以确保实验结果的可重复性。

from data.utils.process import (
    # 从 data.utils.process 模块中导入多个函数：
    exclude_domain,
    # exclude_domain：可能用于排除数据集中的特定领域。
    plot_distribution,
    # plot_distribution：可能用于绘制数据分布的图表。
    prune_args,
    # prune_args：可能用于剪枝或清理参数。
    generate_synthetic_data,
    # generate_synthetic_data：可能用于生成合成数据。
    process_celeba,
    # process_celeba：可能用于处理 CelebA 数据集。
    process_femnist,
    # process_femnist：可能用于处理 FEMNIST 数据集。
)

from data.utils.schemes import (
    dirichlet,
    iid_partition,
    randomly_assign_classes,
    allocate_shards,
    semantic_partition,
)
# 从 data.utils.schemes 模块中导入多个函数或类：
# dirichlet：可能用于从狄利克雷分布中采样。
# iid_partition：可能用于进行独立同分布的分区。
# randomly_assign_classes：可能用于随机分配类别。
# allocate_shards：可能用于分配数据分片。
# semantic_partition：可能用于进行语义分区。

from data.utils.datasets import DATASETS, BaseDataset
# 从 data.utils.datasets 模块中导入 DATASETS 和 BaseDataset：
# DATASETS：可能是一个包含可用数据集名称的集合或字典。
# BaseDataset：可能是一个基类，用于定义数据集的基本操作和属性。

CURRENT_DIR = Path(__file__).parent.absolute()
# 设置 CURRENT_DIR 变量，其值为当前脚本文件所在目录的绝对路径。这通常用于在脚本中引用相对路径时，能够转换为绝对路径。


def main(args):
    dataset_root = CURRENT_DIR / "data" / args.dataset
    # 设置数据集的根目录路径。
    fix_random_seed(args.seed)
    # 使用传入的种子值固定随机种子，以确保结果的可重复性。
    if not os.path.isdir(dataset_root):
        os.mkdir(dataset_root)
        # 如果数据集根目录不存在，则创建它。
    client_num = args.client_num
    partition = {"separation": None, "data_indices": [[] for _ in range(client_num)]}
    # 初始化客户端数量和分区信息。
    # x: num of samples,
    # y: label distribution
    stats = {i: {"x": 0, "y": {}} for i in range(args.client_num)}
    # 初始化每个客户端的统计信息。
    dataset: BaseDataset = None
    # 初始化数据集对象，类型为 BaseDataset。
    # # 根据数据集类型调用相应的数据处理函数。
    if args.dataset == "femnist":
        dataset = process_femnist(args, partition, stats)
        partition["val"] = []
    elif args.dataset == "celeba":
        dataset = process_celeba(args, partition, stats)
        partition["val"] = []
    elif args.dataset == "synthetic":
        dataset = generate_synthetic_data(args, partition, stats)
    else:  # MEDMNIST, COVID, MNIST, CIFAR10, ...
        # NOTE: If `args.ood_domains`` is not empty, then FL-bench will map all labels (class space) to the domain space
        # and partition data according to the new `targets` array.
        dataset = DATASETS[args.dataset](dataset_root, args)
        # 对于其他数据集，使用 DATASETS 字典创建数据集对象。
        targets = np.array(dataset.targets, dtype=np.int32)
        # 将数据集的目标转换为 NumPy 数组。
        target_indices = np.arange(len(targets), dtype=np.int32)
        valid_label_set = set(range(len(dataset.classes)))
        # 创建有效的标签集合。
        if args.dataset in ["domain"] and args.ood_domains:
            # 检查当前处理的数据集是否为 "domain" 类型，并且是否指定了 OOD 领域列表。
            metadata = json.load(open(dataset_root / "metadata.json", "r"))
            # 从 metadata.json 文件中加载元数据。这个文件包含了数据集的相关信息，例如领域映射和索引范围。
            valid_label_set, targets, client_num = exclude_domain(
                client_num=client_num,
                domain_map=metadata["domain_map"],
                targets=targets,
                domain_indices_bound=metadata["domain_indices_bound"],
                ood_domains=set(args.ood_domains),
                partition=partition,
                stats=stats,
            )
            # 调用 exclude_domain 函数，该函数用于处理数据集中的领域排除逻辑。这个函数接收以下参数：
            # client_num: 客户端的数量。
            # domain_map: 领域映射，将领域标签映射到类标签。
            # targets: 数据集中的类标签数组。
            # domain_indices_bound: 领域索引范围，定义了每个领域的起始和结束索引。
            # ood_domains: 一个集合，包含需要排除的 OOD 领域。
            # partition: 当前的数据分区信息。
            # stats: 当前客户端的统计信息。
            # 函数执行完成后，返回三个值：更新后的 valid_label_set（有效的标签集合），targets（更新后的类标签数组），和 client_num（更新后的客户端数量）。
            #
            # 这个 exclude_domain 函数的作用是：
            # 根据 ood_domains 集合中指定的领域，从数据集中排除这些领域的样本。
            # 更新 valid_label_set，以确保排除了与 OOD 领域相关的类标签。
            # 更新 targets 数组，以反映排除 OOD 样本后的类标签。
            # 可能还会更新 client_num，例如，如果某些客户端完全由 OOD 样本组成，它们可能会被删除，从而减少客户端总数。
            # 更新 partition 和 stats，以确保它们反映了数据集的新状态。

        iid_data_partition = deepcopy(partition)
        # 使用 deepcopy 函数深拷贝当前的分区信息 partition，以便在进行 IID 分区时不影响原始分区信息。
        iid_stats = deepcopy(stats)
        # 使用 deepcopy 函数深拷贝当前的统计信息 stats，以便在进行 IID 分区时不影响原始统计信息。
        if 0 < args.iid <= 1.0:  # iid partition
            # 检查 args.iid 参数是否在 (0, 1] 范围内。如果是，表示需要进行 IID 分区。
            sampled_indices = np.array(
                random.sample(
                    target_indices.tolist(), int(len(target_indices) * args.iid)
                )
            )
            # 从 target_indices 中随机采样 args.iid 比例的索引，并将这些索引转换为 NumPy 数组。
            # 这里 random.sample 函数从列表中随机选择指定数量的元素，tolist() 将 NumPy 数组转换为列表。
            # if args.iid < 1.0, then residual indices will be processed by another partition method
            target_indices = np.array(
                list(set(target_indices) - set(sampled_indices)), dtype=np.int32
            )
            # 计算剩余的索引（即没有被随机采样的索引），并将它们转换回 NumPy 数组。这里使用了集合的差集操作来获取未被采样的索引。
            iid_partition(
                targets=targets[sampled_indices],
                target_indices=sampled_indices,
                label_set=valid_label_set,
                client_num=client_num,
                partition=iid_data_partition,
                stats=iid_stats,
            )
            # 调用 iid_partition 函数进行 IID 分区。这个函数接收以下参数：
            # targets: 采样索引对应的目标标签数组。
            # target_indices: 采样的索引数组。
            # label_set: 有效的标签集合。
            # client_num: 客户端的数量。
            # partition: 用于存储 IID 分区结果的分区信息字典。
            # stats: 用于存储 IID 分区统计结果的统计信息字典。
            #
            # 函数的主要作用是：
            # 将采样的索引按照 IID 方式分配给不同的客户端。
            # 更新 iid_data_partition 和 iid_stats，以反映 IID 分区的结果。
        if len(target_indices) > 0 and args.alpha > 0:  # Dirichlet(alpha)
            # 这个条件判断首先检查 target_indices 数组是否包含任何索引（即长度大于0），然后检查 args.alpha 是否大于0。
            # args.alpha 是狄利克雷分布的参数，用于控制数据分区的非均匀性。如果这两个条件都满足，将执行狄利克雷分布的分区方法。
            dirichlet(
                targets=targets[target_indices],
                target_indices=target_indices,
                label_set=valid_label_set,
                client_num=client_num,
                alpha=args.alpha,
                least_samples=args.least_samples,
                partition=partition,
                stats=stats,
            )
            # 调用 dirichlet 函数，根据狄利克雷分布进行数据分区。这个函数接收以下参数：
            # targets: 通过 target_indices 索引筛选出的目标标签数组。
            # target_indices: 要进行分区的数据点索引数组。
            # label_set: 有效的标签集合，包含数据集中所有可能的类别。
            # client_num: 客户端的数量。
            # alpha: 狄利克雷分布的浓度参数，args.alpha 由用户指定。
            # least_samples: 每个客户端至少拥有的样本数量，args.least_samples 由用户指定。
            # partition: 用于存储分区结果的数据结构。
            # stats: 用于存储分区统计信息的数据结构。
            #
            # 函数的主要作用是：
            # 使用狄利克雷分布来决定每个客户端拥有的每个类别的样本数量，以达到非IID的数据分布。
            # 根据 alpha 参数和 least_samples 参数，调整每个客户端的样本分布，以满足特定的数据分布需求。
            # 更新 partition 字典和 stats 字典，以反映分区的结果和统计信息。
        elif len(target_indices) > 0 and args.classes != 0:  # randomly assign classes
            # 这个条件判断首先检查 target_indices 数组是否包含任何索引（即长度大于0），然后检查 args.classes 是否不为0。
            # 如果这两个条件都满足，将执行随机分配类别的分区方法。
            args.classes = max(1, min(args.classes, len(dataset.classes)))
            # 确保 args.classes 的值在合理的范围内。
            # 将其设置为1和数据集类别总数之间的最小值，确保分配的类别数既不少于1，也不多于数据集中的类别总数。
            randomly_assign_classes(
                targets=targets[target_indices],
                target_indices=target_indices,
                label_set=valid_label_set,
                client_num=client_num,
                class_num=args.classes,
                partition=partition,
                stats=stats,
            )
            # 调用 randomly_assign_classes 函数，随机分配类别进行数据分区。这个函数接收以下参数：
            # targets: 通过 target_indices 索引筛选出的目标标签数组。
            # target_indices: 要进行分区的数据点索引数组。
            # label_set: 有效的标签集合，包含数据集中所有可能的类别。
            # client_num: 客户端的数量。
            # class_num: 每个客户端拥有的类别数，由 args.classes 指定。
            # partition: 用于存储分区结果的数据结构。
            # stats: 用于存储分区统计信息的数据结构。
            #
            # 函数的主要作用是：
            # 随机分配每个客户端应拥有的类别，以实现非IID的数据分布。
            # 确保每个客户端至少拥有 args.classes 指定的类别数。
            # 更新 partition 字典和 stats 字典，以反映分区的结果和统计信息。
        elif len(target_indices) > 0 and args.shards > 0:  # allocate shards
            # 这个条件判断检查两个条件：target_indices 数组是否包含任何索引（长度大于0），以及 args.shards 是否大于0。
            # 如果这两个条件都满足，将执行基于分片的数据分配。
            allocate_shards(
                targets=targets[target_indices],
                target_indices=target_indices,
                label_set=valid_label_set,
                client_num=client_num,
                shard_num=args.shards,
                partition=partition,
                stats=stats,
            )
            # 调用 allocate_shards 函数，按照指定的分片数进行数据分配。这个函数接收以下参数：
            # targets: 通过 target_indices 索引筛选出的目标标签数组。
            # target_indices: 要进行分区的数据点索引数组。
            # label_set: 有效的标签集合，包含数据集中所有可能的类别。
            # client_num: 客户端的数量。
            # shard_num: 要分配的分片数量，由 args.shards 指定。
            # partition: 用于存储分区结果的数据结构。
            # stats: 用于存储分区统计信息的数据结构。
            #
            # 函数的主要作用是：
            # 根据 shard_num 将数据集分割成等数量的分片。
            # 将这些分片分配给不同的客户端，可能按照某种策略（例如，确保每个客户端获得相似数量的样本）。
            # 更新 partition 字典和 stats 字典，以反映分片分配的结果和统计信息。
        elif len(target_indices) > 0 and args.semantic:
            # 这个条件判断检查两个条件：target_indices 数组是否包含任何索引（长度大于0），
            # 以及 args.semantic 是否为真。如果这两个条件都满足，将执行基于语义的分区方法。
            semantic_partition(
                dataset=dataset,
                targets=targets[target_indices],
                target_indices=target_indices,
                label_set=valid_label_set,
                efficient_net_type=args.efficient_net_type,
                client_num=client_num,
                pca_components=args.pca_components,
                gmm_max_iter=args.gmm_max_iter,
                gmm_init_params=args.gmm_init_params,
                seed=args.seed,
                use_cuda=args.use_cuda,
                partition=partition,
                stats=stats,
            )
            # 调用 semantic_partition 函数，根据数据的语义特征进行数据分区。这个函数接收以下参数：
            # dataset: 数据集对象，可能包含原始数据和相关的元数据。
            # targets: 通过 target_indices 索引筛选出的目标标签数组。
            # target_indices: 要进行分区的数据点索引数组。
            # label_set: 有效的标签集合，包含数据集中所有可能的类别。
            # efficient_net_type: 指定使用的 EfficientNet 模型的类型或版本，由 args.efficient_net_type 指定。
            # client_num: 客户端的数量。
            # pca_components: 主成分分析（PCA）保留的成分数量，由 args.pca_components 指定。
            # gmm_max_iter: 高斯混合模型（GMM）的最大迭代次数，由 args.gmm_max_iter 指定。
            # gmm_init_params: GMM 初始化参数的选择方式，可以是 "random" 或 "kmeans"，由 args.gmm_init_params 指定。
            # seed: 随机种子，用于确保结果的可重复性。
            # use_cuda: 是否使用 CUDA（GPU加速），由 args.use_cuda 指定。
            # partition: 用于存储分区结果的数据结构。
            # stats: 用于存储分区统计信息的数据结构。
            #
            # 函数的主要作用是：
            # 利用语义特征对数据进行更细致的分区，可能涉及到使用深度学习模型提取特征、进行聚类等操作。
            # 根据语义特征将数据分配给不同的客户端，以达到特定的分区目标。
            # 更新 partition 字典和 stats 字典，以反映语义分区的结果和统计信息。
        elif (
            len(target_indices) > 0
            and args.dataset in ["domain"]
            and args.ood_domains is None
        ):
            # 这个 elif 条件分支检查三个条件是否同时满足：
            # target_indices 数组是否包含任何索引（长度大于0）。
            # args.dataset 的值是否为 "domain"，指定了特定的数据集。
            # args.ood_domains 是否为 None，表示没有指定 Out-Of-Distribution（OOD）领域。
            # 如果这些条件都满足，执行以下操作:
            with open(dataset_root / "original_partition.pkl", "rb") as f:
                # 使用 with 语句打开文件 "original_partition.pkl"，该文件位于由 dataset_root 指定的目录。文件以二进制读取模式打开。
                partition = {}
                partition["data_indices"] = pickle.load(f)
                partition["separation"] = None
                # 从文件中加载分区数据，使用 pickle.load 方法反序列化数据到 partition 字典的 "data_indices" 键。
                # 将 "separation" 键设置为 None。
                args.client_num = len(partition["data_indices"])
                # 更新 args.client_num 为分区中客户端数量的值，即 partition["data_indices"] 列表的长度。
        elif len(target_indices) > 0:
            # 另一个 elif 条件分支检查 target_indices 数组是否包含任何索引（长度大于0）。
            # 如果满足此条件，但没有满足前一个 elif 分支的特定条件，则执行以下操作：
            raise RuntimeError(
                "Part of data indices are ignored. Please set arbitrary one arg from"
                " [--alpha, --classes, --shards, --semantic] for partitioning."
            )
            # 引发一个 RuntimeError 异常，提示用户部分数据索引被忽略了，并要求用户设置一个参数来执行分区。这些参数包括：
            # --alpha：用于狄利克雷分布的参数。
            # --classes：用于随机分配类别的参数。
            # --shards：用于分配分片的参数。
            # --semantic：用于基于语义的分区的参数。
            # 这个异常表明，如果用户想要执行数据分区，需要指定至少一个分区策略参数。
            # 如果没有任何分区参数被设置，代码将无法继续执行，因为不知道如何分配数据。
    # merge the iid and niid partition results
    # 这是注释，说明接下来的代码将合并 IID 和 NIID 的分区结果。
    if 0 < args.iid < 1.0:
        # 这个条件判断检查 args.iid 是否在 (0, 1) 范围内。
        # 如果是，表示系统需要合并 IID 和 Non-IID 分区的结果。args.iid 代表了数据集中 IID 部分的比例。
        num_samples = []
        # 初始化一个空列表 num_samples，用于存储每个客户端拥有的样本数量。
        for i in range(args.client_num):
            # 使用一个 for 循环，遍历所有的客户端，args.client_num 是客户端的数量。
            partition["data_indices"][i] = np.concatenate(
                [partition["data_indices"][i], iid_data_partition["data_indices"][i]]
            ).astype(np.int32)
            # 对于每个客户端，使用 np.concatenate 合并原始的 Non-IID 分区索引 partition["data_indices"][i]
            # 和 IID 分区索引 iid_data_partition["data_indices"][i]，并将结果转换为 32 位整数类型。
            stats[i]["x"] += iid_stats[i]["x"]
            # 将 IID 分区统计中的样本数量 iid_stats[i]["x"] 添加到 Non-IID 分区统计中的样本数量 stats[i]["x"]。
            stats[i]["y"] = {
                cls: stats[i]["y"].get(cls, 0) + iid_stats[i]["y"].get(cls, 0)
                for cls in dataset.classes
            }
            # 更新每个客户端的类别统计信息 stats[i]["y"]。
            # 这里使用字典推导式遍历数据集中的所有类别 dataset.classes，对于每个类别，合并 Non-IID 和 IID 分区统计中的类别计数。
            num_samples.append(stats[i]["x"])
            # 将更新后的样本数量 stats[i]["x"] 追加到 num_samples 列表。
        num_samples = np.array(num_samples)
        # 将 num_samples 列表转换为 NumPy 数组，以便于进行数值计算。
        stats["samples_per_client"] = {
            "mean": num_samples.mean().item(),
            "stddev": num_samples.std().item(),
        }
        # 计算所有客户端样本数量的平均值和标准差，并将它们存储在 stats["samples_per_client"] 中。
        # num_samples.mean().item() 计算平均值并将其转换为一个标量，num_samples.std().item() 计算标准差并将其转换为一个标量。
    if partition["separation"] is None:
        # 根据指定的分割策略（基于用户或基于样本）来分配训练集、验证集和测试集的客户端。
        if args.split == "user":
            # 这个条件判断检查 args.split 参数是否等于 "user"args.split 参数是否等于 "user"。
            # 如果是，则表示分割策略是基于用户（即每个用户的数据完全独立）。
            test_clients_num = int(args.client_num * args.test_ratio)
            # 计算测试集客户端的数量，根据 args.client_num（总客户端数）和 args.test_ratio（测试集比例）计算得出。
            val_clients_num = int(args.client_num * args.val_ratio)
            # 计算验证集客户端的数量
            train_clients_num = args.client_num - test_clients_num - val_clients_num
            # 计算训练集客户端的数量，即总客户端数减去测试集和验证集客户端数。
            clients_4_train = list(range(train_clients_num))
            # 生成训练集客户端的索引列表，索引从 0 到 train_clients_num - 1。
            clients_4_val = list(
                range(train_clients_num, train_clients_num + val_clients_num)
            )
            # 生成验证集客户端的索引列表，索引从 train_clients_num 到 train_clients_num + val_clients_num。
            clients_4_test = list(
                range(train_clients_num + val_clients_num, args.client_num)
            )
            # 生成测试集客户端的索引列表，索引从 train_clients_num + val_clients_num 到 args.client_num - 1。
        elif args.split == "sample":
            # 这个条件判断检查 args.split 参数表示分割策略是基于样本（即数据集的样本被随机分配到不同的客户端）。
            clients_4_train = list(range(args.client_num))
            clients_4_val = clients_4_train
            clients_4_test = clients_4_train
            # 在基于样本的分割策略中，所有客户端都参与训练、验证和测试，因此这里简单地将所有客户端的索引列表赋值给
            # clients_4_train、clients_4_val 和 clients_4_test。
        partition["separation"] = {
            # 这行代码初始化一个字典，用于存储分区的分离信息，并将其赋值给 partition 字典的 "separation" 键。
            "train": clients_4_train,
            # "train": 这个键对应于训练集的客户端索引列表。clients_4_train 是一个列表，包含了用于训练的客户端的索引。
            "val": clients_4_val,
            # "val": 这个键对应于验证集的客户端索引列表。clients_4_val 是一个列表，包含了用于验证的客户端的索引。
            "test": clients_4_test,
            # "test": 这个键对应于测试集的客户端索引列表。clients_4_test 是一个列表，包含了用于测试的客户端的索引。
            "total": args.client_num,
            # "total": 这个键对应于总的客户端数量。args.client_num 是客户端总数。
        }
        # 整体来看，这段代码的作用是：
        # 定义分离策略：根据 args.split 的值（"user" 或 "sample"），确定如何分配训练集、验证集和测试集的客户端。
        # 计算客户端索引：根据客户端总数和比例（args.test_ratio 和 args.val_ratio），计算出训练、验证和测试集的客户端索引。
        # 存储分离信息：将这些索引存储在 partition["separation"] 字典中，以便后续使用。
    if args.dataset not in ["femnist", "celeba"]:
        # 这段代码的目的是根据不同的分割策略，将数据集分配到各个客户端的相应集合（训练集、验证集、测试集）中，
        # 以便在联邦学习或其他分布式学习场景中使用。
        # 这个条件判断确保了只有在处理非 "femnist" 和非 "celeba" 数据集时，才执行以下的数据分割逻辑。
        if args.split == "sample":
            # 当指定的分割方式为 "sample" 时，表示数据集是基于样本级别的分割。
            for client_id in partition["separation"]["train"]:
                # 遍历所有被分配给训练集的客户端索引。
                indices = partition["data_indices"][client_id]
                # 获取当前客户端的所有数据索引。
                np.random.shuffle(indices)
                # 随机打乱索引，以确保数据的随机性。
                testset_size = int(len(indices) * args.test_ratio)
                # 根据指定的测试集比例 args.test_ratio 计算出测试集的大小。
                valset_size = int(len(indices) * args.val_ratio)
                # 根据指定的验证集比例 args.val_ratio 计算出验证集的大小。
                trainset, valset, testset = (
                    indices[testset_size + valset_size :],
                    indices[testset_size : testset_size + valset_size],
                    indices[:testset_size],
                )
                # 根据打乱后的索引和计算出的集大小，分割出训练集、验证集和测试集。
                partition["data_indices"][client_id] = {
                    "train": trainset,
                    "val": valset,
                    "test": testset,
                }
                # 更新客户端的数据分区信息，分别存储训练集、验证集和测试集的索引。
        elif args.split == "user":
            # 当指定的分割方式为 "user" 时，表示数据集是基于用户级别的分割。
            for client_id in partition["separation"]["train"]:
                # 遍历所有被分配给训练集的客户端索引。
                indices = partition["data_indices"][client_id]
                # 获取当前客户端的所有数据索引。
                partition["data_indices"][client_id] = {
                    "train": indices,
                    "val": np.array([], dtype=np.int64),
                    "test": np.array([], dtype=np.int64),
                }
                # 对于 "user" 分割方式，每个客户端的数据被完全分配到训练集中，而验证集和测试集为空。
            for client_id in partition["separation"]["val"]:
                # 遍历所有被分配给验证集的客户端索引。
                indices = partition["data_indices"][client_id]
                partition["data_indices"][client_id] = {
                    "train": np.array([], dtype=np.int64),
                    "val": indices,
                    "test": np.array([], dtype=np.int64),
                }
                # 对于 "user" 分割方式，每个客户端的数据被完全分配到验证集中。
            for client_id in partition["separation"]["test"]:
                indices = partition["data_indices"][client_id]
                # 遍历所有被分配给测试集的客户端索引。
                partition["data_indices"][client_id] = {
                    "train": np.array([], dtype=np.int64),
                    "val": np.array([], dtype=np.int64),
                    "test": indices,
                }
                # 对于 "user" 分割方式，每个客户端的数据被完全分配到测试集中。

    if args.dataset in ["domain"]:
        # 检查当前数据集是否为 "domain" 类型。
        class_targets = np.array(dataset.targets, dtype=np.int32)
        # 将数据集中的目标（通常是类别标签）转换为 NumPy 数组，并指定数据类型为 np.int32。
        metadata = json.load(open(dataset_root / "metadata.json", "r"))
        # 从 metadata.json 文件中加载元数据，这个文件包含了数据集的额外信息，例如领域映射和索引范围。
        def _idx_2_domain_label(index):
            # 定义一个函数 _idx_2_domain_label，它将索引转换为对应的领域标签。
            for domain, bound in metadata["domain_indices_bound"].items():
                if bound["begin"] <= index < bound["end"]:
                    return metadata["domain_map"][domain]
                    # 函数内部遍历元数据中的 domain_indices_bound 字典，确定索引属于哪个领域，并返回该领域的标签。
        domain_targets = np.vectorize(_idx_2_domain_label)(
            np.arange(len(class_targets), dtype=np.int64)
        )
        # 使用 NumPy 的 vectorize 函数将 _idx_2_domain_label 函数应用于 class_targets 数组的长度范围内的每个索引，生成领域标签数组。
        for client_id in range(args.client_num):
            # 遍历所有客户端。
            indices = np.concatenate(
                [
                    partition["data_indices"][client_id]["train"],
                    partition["data_indices"][client_id]["val"],
                    partition["data_indices"][client_id]["test"],
                ]
            ).astype(np.int64)
            # 对于每个客户端，合并训练集、验证集和测试集的索引，并转换为 np.int64 类型的 NumPy 数组。
            stats[client_id] = {
                "x": len(indices),
                "class space": Counter(class_targets[indices].tolist()),
                "domain space": Counter(domain_targets[indices].tolist()),
            }
            # 为每个客户端计算统计信息，包括：
            # "x": 客户端拥有的样本总数。
            # "class space": 客户端样本的类别分布，使用 collections.Counter 计算。
            # "domain space": 客户端样本的领域分布。
        stats["domain_map"] = metadata["domain_map"]
        # 将元数据中的领域映射信息存储在统计信息字典的 "domain_map" 键中。
    # plot
    # 这段代码的目的是为数据集的每个类别或领域在不同客户端的分布情况提供一个可视化的表示，这有助于理解数据在联邦学习环境中的分布特性。
    # 这段代码负责根据命令行参数 args.plot_distribution 的值来决定是否绘制数据分布图。
    # 如果需要绘制，它会根据数据集的类型（"domain" 或其他）来绘制类别分布或领域分布图。以下是对代码的逐行解释：
    if args.plot_distribution:
        # 检查命令行参数 args.plot_distribution 是否为真，如果为真，则执行绘制分布图的代码块。
        if args.dataset in ["domain"]:
            # 检查数据集是否为 "domain" 类型。
            # class distribution
            counts = np.zeros((len(dataset.classes), args.client_num), dtype=np.int64)
            # 创建一个形状为 (len(dataset.classes), args.client_num) 的零矩阵，用于存储每个类别在每个客户端的样本计数。
            client_ids = range(args.client_num)
            # 创建一个客户端索引的范围。
            for i, client_id in enumerate(client_ids):
                # 遍历客户端索引。
                for j, cnt in stats[client_id]["class space"].items():
                    counts[j][i] = cnt
                # 对于每个客户端，更新类别分布矩阵 counts，其中 j 是类别索引，cnt 是该类别的样本计数。
            plot_distribution(
                client_num=args.client_num,
                label_counts=counts,
                save_path=f"{dataset_root}/class_distribution.png",
            )
            # 调用 plot_distribution 函数来绘制类别分布图，并将结果保存到指定路径。
            # domain distribution
            counts = np.zeros(
                (len(metadata["domain_map"]), args.client_num), dtype=np.int64
            )
            # 创建一个形状为 (len(metadata["domain_map"], args.client_num) 的零矩阵，用于存储每个领域在每个客户端的样本计数。
            client_ids = range(args.client_num)
            for i, client_id in enumerate(client_ids):
                # 再次遍历客户端索引。
                for j, cnt in stats[client_id]["domain space"].items():
                    counts[j][i] = cnt
                    # 对于每个客户端，更新领域分布矩阵 counts，其中 j 是领域索引，cnt 是该领域的样本计数。
            plot_distribution(
                client_num=args.client_num,
                label_counts=counts,
                save_path=f"{dataset_root}/domain_distribution.png",
            )
            # 调用 plot_distribution 函数来绘制领域分布图，并将结果保存到指定路径。
        else:
            # 如果数据集不是 "domain" 类型，执行这个 else 分支。
            counts = np.zeros((len(dataset.classes), args.client_num), dtype=np.int64)
            # 创建一个零矩阵，用于存储每个类别在每个客户端的样本计数。
            client_ids = range(args.client_num)
            # 创建一个客户端索引的范围。
            for i, client_id in enumerate(client_ids):
                # 遍历客户端索引。
                for j, cnt in stats[client_id]["y"].items():
                    counts[j][i] = cnt
                    # 对于每个客户端，更新类别分布矩阵 counts。
            plot_distribution(
                client_num=args.client_num,
                label_counts=counts,
                save_path=f"{dataset_root}/class_distribution.png",
            )
            # 调用 plot_distribution 函数来绘制类别分布图，并将结果保存到指定路径。
    with open(dataset_root / "partition.pkl", "wb") as f:
        # 使用 with 语句打开文件 "partition.pkl"，位于由 dataset_root 指定的目录。文件以二进制写入模式打开（"wb"）。
        pickle.dump(partition, f)
        # 使用 pickle.dump 函数将 partition 字典序列化并写入到文件中。
        # partition 字典包含了数据分区的详细信息，如客户端的数据索引和分离信息。
    with open(dataset_root / "all_stats.json", "w") as f:
        # 使用 with 语句打开文件 "all_stats.json"，同样位于由 dataset_root 指定的目录。文件以文本写入模式打开（"w"）。
        json.dump(stats, f, indent=4)
        # 使用 json.dump 函数将 stats 字典转换为 JSON 格式的字符串，并写入到文件中。
        # indent=4 参数用于使输出的 JSON 文件具有可读性，即每个层级缩进4个空格。
    with open(dataset_root / "args.json", "w") as f:
        # 使用 with 语句打开文件 "args.json"，位于由 dataset_root 指定的目录。文件以文本写入模式打开。
        json.dump(prune_args(args), f, indent=4)
        # 使用 json.dump 函数将 prune_args(args) 函数的返回值（处理后的命令行参数）转换为 JSON 格式的字符串，并写入到文件中。
        # indent=4 参数同样用于增加输出的可读性。


if __name__ == "__main__":
    parser = ArgumentParser()
    # 创建一个 ArgumentParser 对象，用于解析命令行参数。
    parser.add_argument(
        "-d", "--dataset", type=str, choices=DATASETS.keys(), required=True
    )
    # 添加一个必需的命令行参数 --dataset（或简写为 -d），它接收一个字符串，其值必须在 DATASETS 字典的键中选择。
    parser.add_argument("--iid", type=float, default=0.0)
    # 添加一个可选的命令行参数 --iid，它接收一个浮点数，默认值为 0.0。
    parser.add_argument("-cn", "--client_num", type=int, default=20)
    # 添加一个可选的命令行参数 --client_num（或简写为 -cn），它接收一个整数，默认值为 20。
    parser.add_argument("--seed", type=int, default=42)
    # 添加一个可选的命令行参数 --seed，它接收一个整数，默认值为 42。
    parser.add_argument(
        "-sp", "--split", type=str, choices=["sample", "user"], default="sample"
    )
    # 添加一个可选的命令行参数 --split（或简写为 -sp），它接收一个字符串，其值必须在 ["sample", "user"] 中选择，默认值为 "sample"。
    parser.add_argument("-vr", "--val_ratio", type=float, default=0.0)
    # 添加一个可选的命令行参数 --val_ratio（或简写为 -vr），它接收一个浮点数，默认值为 0.0。
    parser.add_argument("-tr", "--test_ratio", type=float, default=0.25)
    # 添加一个可选的命令行参数 --test_ratio（或简写为 -tr），它接收一个浮点数，默认值为 0.25。
    parser.add_argument("-pd", "--plot_distribution", type=int, default=1)
    # 添加一个可选的命令行参数 --plot_distribution（或简写为 -pd），它接收一个整数，默认值为 1。
    # Randomly assign classes
    parser.add_argument("-c", "--classes", type=int, default=0)
    # 添加一个可选的命令行参数 --classes（或简写为 -c），它接收一个整数，默认值为 0。
    # Shards
    parser.add_argument("-s", "--shards", type=int, default=0)
    # 添加一个可选的命令行参数 --shards（或简写为 -s），它接收一个整数，默认值为 0。
    # Dirichlet
    parser.add_argument("-a", "--alpha", type=float, default=0)
    parser.add_argument("-ls", "--least_samples", type=int, default=40)
    # 添加两个与狄利克雷分布相关的可选命令行参数 --alpha（或简写为 -a）和 --least_samples（或简写为 -ls），
    # 分别接收浮点数和整数，默认值分别为 0 和 40。
    # For synthetic data only
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--dimension", type=int, default=60)
    # 添加三个仅用于合成数据的可选命令行参数 --gamma、 --beta 和 --dimension，分别接收浮点数和整数，默认值分别为 0.5、 0.5 和 60。
    # For CIFAR-100 only
    parser.add_argument("--super_class", type=int, default=0)
    # 添加一个仅用于CIFAR-100数据集的可选命令行参数 --super_class，它接收一个整数，默认值为 0。
    # For EMNIST only
    parser.add_argument(
        "--emnist_split",
        type=str,
        choices=["byclass", "bymerge", "letters", "balanced", "digits", "mnist"],
        default="byclass",
    )
    # 添加一个仅用于EMNIST数据集的可选命令行参数 --emnist_split，它接收一个字符串，
    # 其值必须在 ["byclass", "bymerge", "letters", "balanced", "digits", "mnist"] 中选择，默认值为 "byclass"。
    # For domain generalization datasets only
    parser.add_argument("--ood_domains", nargs="+", default=None)
    # 添加一个仅用于领域泛化数据集的可选命令行参数 --ood_domains，它接收一个或多个字符串值，默认为 None。
    # For semantic partition only
    parser.add_argument("-sm", "--semantic", type=int, default=0)
    parser.add_argument("--efficient_net_type", type=int, default=0)
    parser.add_argument("--gmm_max_iter", type=int, default=100)
    parser.add_argument(
        "--gmm_init_params", type=str, choices=["random", "kmeans"], default="kmeans"
    )
    parser.add_argument("--pca_components", type=int, default=256)
    parser.add_argument("--use_cuda", type=int, default=1)
    # 添加一系列与语义分区相关的可选命令行参数：
    # --semantic（或简写为 -sm）：接收一个整数，默认值为 0。
    # --efficient_net_type：接收一个整数，默认值为 0。
    # --gmm_max_iter：接收一个整数，默认值为 100。
    # --gmm_init_params：接收一个字符串，其值必须在 ["random", "kmeans"] 中选择，默认值为 "kmeans"。
    # --pca_components：接收一个整数，默认值为 256。
    # --use_cuda：接收一个整数，默认值为 1。
    args = parser.parse_args()
    # 解析命令行参数，并将结果存储在 args 中。
    main(args)
    # 调用 main 函数，并将解析后的参数 args 传递给它。
