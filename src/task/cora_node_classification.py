import neptune.new as neptune
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger
from src.config.project_config import settings
from src.constant import CORA_DATA_PATH
from src.data.load_data import load_cora_data
from src.evaluation.evaluation import accuracy
from src.model.gcn import GCN


class CoraNodeClassification:

    def __init__(self, config):
        self.config = config

        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)

        self.neptune = neptune.init(
            project=settings.NEPTUNE_PROJECT_NAME,
            api_token=settings.NEPTUNE_API_TOKEN)

        self.neptune["config"] = self.config.__dict__

    def run(self):
        adj, features, labels, idx_train, idx_val, idx_test = load_cora_data(CORA_DATA_PATH)

        setattr(self.config, "n_features", features.shape[1])
        setattr(self.config, 'n_class', labels.max().item() + 1)

        self.model = GCN(self.config)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)

        for epoch in range(1, self.config.epochs+1):
            loss_train, acc_train = self.train(features, adj, labels, idx_train)
            loss_val, acc_val = self.val(features, adj, labels, idx_val)

            self.neptune["train_loss"].log(loss_train)
            self.neptune["train_accuracy"].log(acc_train)
            self.neptune["val_loss"].log(loss_val)
            self.neptune["val_accuracy"].log(acc_val)

            if epoch % self.config.display_intervals == 0:
                logger.info(f"Epoch: {epoch}")
                logger.info(f"Train Loss: {loss_train} Train Acc: {acc_train}")
                logger.info(f"Val Loss: {loss_val} Val Acc: {acc_val}\n")

        loss_test, acc_test = self.test(features, adj, labels, idx_test)
        logger.info(f"Test Loss: {loss_test} Test Acc: {acc_test}")
        self.neptune["test_loss"] = loss_test
        self.neptune["test_accuracy"] = acc_test

    def train(self, features, adj, labels, idx_train):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(features, adj)

        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

        loss_train.backward()
        self.optimizer.step()

        return loss_train.item(), acc_train.item()

    def val(self, features, adj, labels, idx_val):
        self.model.eval()
        output = self.model(features, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])

        return loss_val.item(), acc_val.item()

    def test(self, features, adj, labels, idx_test):
        self.model.eval()
        output = self.model(features, adj)

        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])

        return loss_test.item(), acc_test.item()
