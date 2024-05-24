from collections import Counter
from typing import Any

import torch
import numpy as np

from src.client.fedavg import FedAvgClient
from src.utils.constants import NUM_CLASSES


class FedPACClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.global_prototypes = {}
        self.label_distribs = {}
        for client_id, indices in enumerate(self.data_indices):
            counter = Counter(np.array(self.dataset.targets)[indices["train"]])
            self.label_distribs[client_id] = torch.tensor(
                [counter.get(i, 0) for i in range(len(self.dataset.classes))],
                dtype=torch.float,
            )

        self.v = None
        self.h_ref = None

    @torch.no_grad
    def extract_stats(self):
        feature_length = self.model.classifier.in_features
        features = self.calculate_prototypes()

        distrib1 = self.label_distribs[self.client_id]
        distrib1 = distrib1 / distrib1.sum()
        distrib2 = distrib1.mul(distrib1)
        self.v = 0
        self.h_ref = torch.zeros(
            (NUM_CLASSES[self.args.common.dataset], feature_length), device=self.device
        )
        for i in range(NUM_CLASSES[self.args.common.dataset]):
            if isinstance(features[i], torch.Tensor):
                size = features[i].shape[0]
                mean = features[i].mean(dim=0)
                self.h_ref[i] = distrib1[i] * mean
                self.v += (
                    distrib1[i]
                    * torch.trace((torch.mm(features[i].t(), features[i]) / size))
                ).item()
                self.v -= (distrib2[i] * (torch.mul(mean, mean))).sum().item()

        self.v /= len(self.trainset.indices)

    def calculate_prototypes(self, mean=False):
        prototypes = [[] for _ in self.dataset.classes]
        for x, y in self.trainloader:
            x = x.to(self.device)
            features = self.model.get_final_features(x, detach=True)
            for i, label in enumerate(y.tolist()):
                prototypes[label].append(features[i])

        for i, features in enumerate(prototypes):
            if len(features) > 0:
                prototypes[i] = torch.stack(features)
                if mean:
                    prototypes[i] = prototypes[i].mean(dim=0)

        return prototypes

    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)
        self.global_prototypes = package["global_prototypes"]
        self.extract_stats()

    def package(self):
        client_package = super().package()
        prototypes = []
        for proto in self.calculate_prototypes(mean=True):
            if isinstance(proto, torch.Tensor):
                prototypes.append(proto.detach().cpu().clone())
            elif isinstance(proto, list) and len(proto) == 0:  # void prototype
                prototypes.append(proto)
        client_package["prototypes"] = prototypes
        client_package["label_distrib"] = self.label_distribs[self.client_id]
        client_package["v"] = self.v
        client_package["h_ref"] = self.h_ref.cpu().clone()
        return client_package

    def fit(self):
        self.model.train()
        self.dataset.train()
        local_prototypes = self.calculate_prototypes(mean=True)
        for E in range(self.local_epoch):
            if E < self.args.fedpac.train_classifier_round:
                self.model.base.requires_grad_(False)
                self.model.classifier.requires_grad_(True)
                for x, y in self.trainloader:
                    x, y = x.to(self.device), y.to(self.device)
                    logits = self.model(x)
                    loss = self.criterion(logits, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            else:
                self.model.base.requires_grad_(True)
                self.model.classifier.requires_grad_(False)
                for x, y in self.trainloader:
                    x, y = x.to(self.device), y.to(self.device)
                    features = self.model.get_final_features(x, detach=False)
                    logits = self.model.classifier(features)
                    loss_ce = self.criterion(logits, y)

                    loss_mse = 0
                    if self.global_prototypes is not None:
                        for i, label in enumerate(y.cpu().tolist()):
                            if label in self.global_prototypes.keys():
                                loss_mse += torch.nn.functional.mse_loss(
                                    self.global_prototypes[label].to(self.device), features[i]
                                )
                            else:
                                loss_mse += torch.nn.functional.mse_loss(
                                    local_prototypes[label], features[i]
                                )
                    self.optimizer.zero_grad()
                    loss = loss_ce + self.args.fedpac.lamda * loss_mse
                    self.optimizer.step()

        self.model.requires_grad_(True)