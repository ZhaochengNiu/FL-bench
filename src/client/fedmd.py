from typing import List, OrderedDict

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from fedavg import FedAvgClient
from data.utils.datasets import DATASETS
from src.utils.tools import Logger, NestedNamespace, trainable_params
from src.utils.constants import DATA_MEAN, DATA_STD, FLBENCH_ROOT
from src.utils.models import DecoupledModel


class FedMDClient(FedAvgClient):
    def __init__(self, model: DecoupledModel, args: NestedNamespace, logger: Logger, device: torch.device):
        super(FedMDClient, self).__init__(model, args, logger, device)

        # --------- you can define your custom data preprocessing strategy here ------------
        test_data_transform = transforms.Compose(
            [
                transforms.Normalize(
                    DATA_MEAN[self.args.common.dataset],
                    DATA_STD[self.args.common.dataset],
                )
            ]
            if self.args.common.dataset in DATA_MEAN
            and self.args.common.dataset in DATA_STD
            else []
        )
        test_target_transform = transforms.Compose([])
        train_data_transform = transforms.Compose(
            [
                transforms.Normalize(
                    DATA_MEAN[self.args.common.dataset],
                    DATA_STD[self.args.common.dataset],
                )
            ]
            if self.args.common.dataset in DATA_MEAN
            and self.args.common.dataset in DATA_STD
            else []
        )
        train_target_transform = transforms.Compose([])
        # --------------------------------------------------------------------------------

        self.public_dataset = DATASETS[self.args.fedmd.public_dataset](
            root=FLBENCH_ROOT / "data" / args.fedmd.public_dataset,
            args=None,
            train_data_transform=train_data_transform,
            train_target_transform=train_target_transform,
            test_data_transform=test_data_transform,
            test_target_transform=test_target_transform,
        )
        self.public_dataset_loader = DataLoader(
            self.public_dataset, self.args.fedmd.public_batch_size, shuffle=True
        )
        self.iter_public_loader = iter(self.public_dataset_loader)
        self.public_data: List[torch.Tensor] = []
        self.public_targets: List[torch.Tensor] = []
        self.consensus: List[torch.Tensor] = []
        self.mse_criterion = torch.nn.MSELoss()

    def load_public_data_batches(self):
        for _ in range(self.args.fedmd.public_batch_num):
            try:
                x, y = next(self.iter_public_loader)
                if len(x) <= 1:
                    x, y = next(self.iter_public_loader)

            except StopIteration:
                self.iter_public_loader = iter(self.public_dataset_loader)
                x, y = next(self.iter_public_loader)
            self.public_data.append(x.to(self.device))
            self.public_targets.append(y.to(self.device))

    def train(
        self,
        client_id: int,
        local_epoch: int,
        new_parameters: OrderedDict[str, torch.Tensor],
        verbose: bool,
    ):
        self.client_id = client_id
        self.local_epoch = local_epoch
        self.load_dataset()
        self.set_parameters(new_parameters)
        self.digest()
        res = self.train_and_log(verbose=verbose)
        return trainable_params(self.model, detach=True), res

    @torch.no_grad()
    def get_scores(self, client_id, new_parameters):
        self.client_id = client_id
        self.set_parameters(new_parameters)
        self.model.eval()
        return [self.model(x).clone() for x in self.public_data]

    def digest(self):
        self.model.train()
        for _ in range(self.args.fedmd.digest_epoch):
            for i in range(self.args.fedmd.public_batch_num):
                logit = self.model(self.public_data[i])
                loss = self.mse_criterion(logit, self.consensus[i])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
