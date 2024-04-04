from collections import OrderedDict

import torch

from fedbn import FedBNClient
from src.utils.models import DecoupledModel
from src.utils.tools import Logger, NestedNamespace


class FedAPClient(FedBNClient):
    def __init__(
        self,
        model: DecoupledModel,
        args: NestedNamespace,
        logger: Logger,
        device: torch.device,
    ):
        super(FedAPClient, self).__init__(model, args, logger, device)

        self.model.need_all_features()
        self.pretrain = False

    def load_dataset(self):
        super().load_dataset()
        num_pretrain_samples = int(self.args.fedap.pretrain_ratio * len(self.trainset))
        if self.args.fedap.version != "f":
            if self.pretrain:
                self.trainset.indices = self.trainset.indices[:num_pretrain_samples]
            else:
                self.trainset.indices = self.trainset.indices[num_pretrain_samples:]

    @torch.no_grad()
    def get_all_features(
        self, client_id: int, new_parameters: OrderedDict[str, torch.Tensor]
    ):
        self.client_id = client_id
        self.load_dataset()
        self.set_parameters(new_parameters)
        features_list = []
        batch_size_list = []
        for x, _ in self.trainloader:
            features_list.append(
                [
                    feature.cpu()
                    for feature in self.model.get_all_features(x.to(self.device))
                ]
            )
            batch_size_list.append(len(x))

        self.save_state()

        if self.args.fedap.version == "d":
            for i, features in enumerate(features_list):
                for j in range(len(features)):
                    if len(features[j].shape) == 4 and len(features[j + 1].shape) < 4:
                        features_list[i] = [features[j]]
                        break
        return features_list, batch_size_list
