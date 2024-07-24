import numpy as np
# 这行代码导入了NumPy库，并将这个库命名为 np。NumPy是一个用于科学计算的Python库，提供了大量的数学函数和对多维数组的支持。
import torch
# 这行代码导入了PyTorch库。PyTorch是一个开源的机器学习库，广泛用于计算机视觉和自然语言处理等深度学习领域。
# 它提供了强大的GPU加速的张量计算能力。
from sklearn import metrics
# 这行代码从scikit-learn库中导入了 metrics 模块。scikit-learn是一个流行的机器学习库，提供了许多用于分类、回归、聚类和降维的算法。
# metrics 模块包含了用于评估模型性能的各种指标和工具，例如准确率、精确度、召回率、F1分数等。


# 这个函数是类型安全的，它会根据输入的类型选择正确的转换方式。如果输入的类型既不是 torch.Tensor 也不是 list 或 np.ndarray，
# 函数将报错，提醒用户输入的类型不支持转换。这种设计有助于调试和确保数据类型的一致性。
def to_numpy(x):
    # 这段代码定义了一个名为 to_numpy 的函数，其目的是将输入的变量转换为 NumPy 数组。以下是对函数的逐行解释：
    # 定义了一个名为 to_numpy 的函数，它接受一个参数 x。
    if isinstance(x, torch.Tensor):
        # 检查输入的 x 是否是 PyTorch 的 Tensor 类型。
        return x.cpu().numpy()
        # 如果是 PyTorch 的 Tensor 类型，首先调用 .cpu() 方法将其转移到 CPU（如果它不在 CPU 上的话），
        # 然后调用 .numpy() 方法将其转换为 NumPy 数组并返回。
    elif isinstance(x, list):
        # 如果 x 不是 Tensor 类型，接下来检查它是否是 Python 的列表类型。
        return np.array(x)
        # 如果是列表类型，使用 NumPy 的 array 函数将列表转换为 NumPy 数组并返回。
    elif isinstance(x, np.ndarray):
        # 然后检查 x 是否已经是 NumPy 数组类型。
        return x
        # 如果 x 已经是 NumPy 数组，直接返回它，因为没有转换的必要。
    else:
        # 如果 x 不是上述任何类型，执行 else 代码块。
        raise TypeError(f"Unsupported type: {type(x)}")
        # 抛出一个 TypeError 异常，提示不支持的类型，并显示 x 的类型。


# 这个 Metrics 类可以用于跟踪和计算模型在训练或评估过程中的性能指标，如损失值、精确度、召回率和准确率等。
# 注意，这个类依赖于之前定义的 to_numpy 函数来确保输入数据是 NumPy 数组格式。
class Metrics:
    # 定义了一个名为 Metrics 的类。
    def __init__(self, loss=None, predicts=None, targets=None):
        # 类的构造函数，接受三个可选参数：loss（损失值，可以是数字或 None）、predicts（预测结果列表，可以是列表或 None）、
        # targets（真实目标列表，可以是列表或 None）。如果这些参数未提供，则默认值分别为 0.0、空列表和空列表。
        self._loss = loss if loss is not None else 0.0
        # 初始化 _loss 属性，如果 loss 参数提供则使用该值，否则初始化为 0.0。
        self._targets = targets if targets is not None else []
        # 初始化 _targets 属性，如果 targets 参数提供则使用该列表，否则初始化为空列表。
        self._predicts = predicts if predicts is not None else []
        # 初始化 _predicts 属性，如果 predicts 参数提供则使用该列表，否则初始化为空列表。

    def update(self, other):
        # 定义了一个 update 方法，用于将另一个 Metrics 实例的统计数据合并到当前实例中。
        if other is not None:
            # 检查传入的 other 对象是否不为 None。
            self._predicts.extend(to_numpy(other._predicts))
            # 将 other 实例的预测结果转换为 NumPy 数组，并扩展当前实例的预测结果列表。
            self._targets.extend(to_numpy(other._targets))
            # 将 other 实例的目标结果转换为 NumPy 数组，并扩展当前实例的目标结果列表。
            self._loss += other._loss
            # 将 other 实例的损失值加到当前实例的损失值上。

    def _calculate(self, metric, **kwargs):
        # 定义了一个私有方法 _calculate，用于计算给定的评估指标。它接受一个指标函数 metric 和一些关键字参数 kwargs。
        return metric(self._targets, self._predicts, **kwargs)
        # 调用传入的指标函数 metric，传入目标和预测结果，以及其他关键字参数，并返回计算结果。

    @property
    def loss(self):
        # 定义了一个只读属性 loss，计算平均损失值。
        if len(self._targets) > 0:
            return self._loss / len(self._targets)
        else:
            return 0
        # 如果目标列表不为空，则返回总损失除以目标数量的平均损失；如果为空，则返回 0。

    @property
    def macro_precision(self):
        # 定义了一个只读属性 macro_precision，计算宏平均精确度。
        score = self._calculate(
            metrics.precision_score, average="macro", zero_division=0
        )
        return score * 100
        # 使用 _calculate 方法和 precision_score 函数计算宏平均精确度，并将结果转换为百分比。

    @property
    def macro_recall(self):
        # 定义了一个只读属性 macro_recall，计算宏平均召回率。
        score = self._calculate(metrics.recall_score, average="macro", zero_division=0)
        return score * 100
        # 使用 _calculate 方法和 recall_score 函数计算宏平均召回率，并将结果转换为百分比。

    @property
    def micro_precision(self):
        # 定义了一个只读属性 micro_precision，计算微平均精确度。
        score = self._calculate(
            metrics.precision_score, average="micro", zero_division=0
        )
        return score * 100
        # 使用 _calculate 方法和 precision_score 函数计算微平均精确度，并将结果转换为百分比。

    @property
    def micro_recall(self):
        # 定义了一个只读属性 micro_recall，计算微平均召回率。
        score = self._calculate(metrics.recall_score, average="micro", zero_division=0)
        return score * 100
        # 使用 _calculate 方法和 recall_score 函数计算微平均召回率，并将结果转换为百分比。

    @property
    def accuracy(self):
        # 定义了一个只读属性 accuracy，计算准确率。
        if self.size == 0:
            return 0
        score = self._calculate(metrics.accuracy_score)
        return score * 100
        # 如果数据集大小为 0，则返回 0；否则，使用 _calculate 方法和 accuracy_score 函数计算准确率，并将结果转换为百分比。

    @property
    def corrects(self):
        # 定义了一个只读属性 corrects，计算非归一化的准确率。
        return self._calculate(metrics.accuracy_score, normalize=False)
        # 使用 _calculate 方法和 accuracy_score 函数计算非归一化的准确率。

    @property
    def size(self):
        # 定义了一个只读属性 size，返回目标列表的长度。
        return len(self._targets)
        # 返回目标列表的长度，即数据集的大小。
