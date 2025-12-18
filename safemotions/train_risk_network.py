import os
import inspect
import sys
import argparse
import json
import random
import datetime
import shutil
from pathlib import Path
from ast import literal_eval

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (accuracy_score, precision_score, recall_score, roc_auc_score,
                             precision_recall_curve, auc)

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.dirname(current_dir))

from safemotions.model.risk_network import RiskNetwork

# Dataset with rebalancing indices logic
class RiskDataset(Dataset):
    def __init__(self, data_dir):
        self.x, self.y = [], []
        self._state_action_risk = False
        self._state_size = None
        self._action_size = None

        for csv_file in sorted(Path(data_dir).glob("*.csv")):
            df = pd.read_csv(csv_file, converters={'state': literal_eval, 'action': literal_eval})
            df_y = list(df["risk"])
            if "action" in df:
                self._state_action_risk = True
                state = list(df["state"])
                action = list(df["action"])
                df_x = [state[i] + action[i] for i in range(len(state))]
                self._state_size = len(state[0])
                self._action_size = len(action[0])
            else:
                df_x = list(df["state"])
                self._state_size = len(df_x[0])
            self.x.extend(df_x)
            self.y.extend(df_y)

        self.x = np.array(self.x, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)
        assert len(self.x) == len(self.y)

        self.risky_indices = np.where(self.y == 1.0)[0]
        self.not_risky_indices = np.where(self.y == 0.0)[0]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])

    @property
    def state_action_risk(self):
        return self._state_action_risk

    @property
    def state_size(self):
        return self._state_size

    @property
    def action_size(self):
        return self._action_size


class RebalancingSampler(Sampler):
    def __init__(self, dataset, batch_size, risky_fraction):
        self.dataset = dataset
        self.batch_size = batch_size
        self.risky_fraction = risky_fraction
        self.num_risky = int(batch_size * risky_fraction)
        self.num_not_risky = batch_size - self.num_risky

        if len(dataset.risky_indices) < self.num_risky:
            raise ValueError(f"Not enough risky datapoints for batch rebalancing: required {self.num_risky}, "
                             f"but got {len(dataset.risky_indices)}")
        if len(dataset.not_risky_indices) < self.num_not_risky:
            raise ValueError(f"Not enough non-risky datapoints for batch rebalancing: required {self.num_not_risky}, "
                             f"but got {len(dataset.not_risky_indices)}")

        self.num_batches = min(len(dataset.risky_indices) // self.num_risky,
                               len(dataset.not_risky_indices) // self.num_not_risky)

        print("Batches train dataset with rebalancing", self.num_batches)

    def __iter__(self):
        risky_idx = np.random.permutation(self.dataset.risky_indices)
        not_risky_idx = np.random.permutation(self.dataset.not_risky_indices)

        for i in range(self.num_batches):
            batch_indices = np.concatenate((risky_idx[i*self.num_risky:(i+1)*self.num_risky],
                                            not_risky_idx[i*self.num_not_risky:(i+1)*self.num_not_risky]))
            np.random.shuffle(batch_indices)
            yield from batch_indices.tolist()

    def __len__(self):
        return self.num_batches * self.batch_size


def calculate_metrics_at_thresholds(targets, outputs, thresholds):
    metrics = {}
    outputs = np.array(outputs)
    targets = np.array(targets)

    for thr in thresholds:
        preds = (outputs > thr).astype(int)
        precision = precision_score(targets, preds, zero_division=0)
        recall = recall_score(targets, preds, zero_division=0)
        metrics[f'precision_{thr}'] = precision
        metrics[f'recall_{thr}'] = recall

    return metrics

def train(model, train_loader, val_loader, epoch_compensation_factor, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    pos_weight = args.risky_state_class_weight if args.risky_state_class_weight is not None else 1.0
    neg_weight = 1.0  # negative class weight

    loss_fn = nn.BCELoss(reduction='none')
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_regularization, decoupled_weight_decay=True)
    writer = SummaryWriter(log_dir=args.logdir)

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

    for epoch in range(int(args.epochs * epoch_compensation_factor)):
        model.train()
        epoch_compensated = round(epoch / epoch_compensation_factor, 2)
        train_loss, preds, targets, outputs = 0.0, [], [], []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)

            # Create the sample weights based on their class
            weights = torch.where(labels == 1, pos_weight, neg_weight)

            # Calculate loss per sample, multiply by weights, then average
            elementwise_loss = loss_fn(output, labels)
            loss = (elementwise_loss * weights).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            outputs.extend(output.detach().cpu().numpy())
            preds.extend((output.detach().cpu().numpy() > 0.5).astype(int))
            targets.extend(labels.cpu().numpy())

        acc = accuracy_score(targets, preds)
        roc_auc = roc_auc_score(targets, outputs)
        metric_dict = calculate_metrics_at_thresholds(targets, outputs, thresholds)

        precision, recall, _ = precision_recall_curve(targets, outputs)
        pr_auc = auc(recall, precision)

        writer.add_scalar('Loss/train', train_loss / len(train_loader.dataset), epoch_compensated)
        writer.add_scalar('Accuracy/train', acc, epoch_compensated)
        writer.add_scalar('ROC_AUC/train', roc_auc, epoch_compensated)
        writer.add_scalar('PR_AUC/train', pr_auc, epoch_compensated)
        for k, v in metric_dict.items():
            writer.add_scalar(f'{k}/train', v, epoch_compensated)

        print(f"Epoch {epoch_compensated + 1}/{args.epochs} - Train Loss: {train_loss / len(train_loader.dataset):.4f} "
              f"Acc: {acc:.4f} ROC AUC: {roc_auc:.4f} PR AUC: {pr_auc:.4f}")

        if val_loader is not None:
            model.eval()
            val_loss, val_preds, val_targets, val_outputs = 0.0, [], [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    output = model(inputs)

                    # Create the sample weights based on their class
                    weights = torch.where(labels == 1, pos_weight, neg_weight)

                    # Calculate loss per sample, multiply by weights, then average
                    elementwise_loss = loss_fn(output, labels)
                    loss = (elementwise_loss * weights).mean()

                    val_loss += loss.item() * inputs.size(0)
                    val_outputs.extend(output.cpu().numpy())
                    val_preds.extend((output.cpu().numpy() > 0.5).astype(int))
                    val_targets.extend(labels.cpu().numpy())

            val_acc = accuracy_score(val_targets, val_preds)
            val_roc_auc = roc_auc_score(val_targets, val_outputs)
            val_metric_dict = calculate_metrics_at_thresholds(val_targets, val_outputs, thresholds)

            val_precision, val_recall, _ = precision_recall_curve(val_targets, val_outputs)
            val_pr_auc = auc(val_recall, val_precision)

            writer.add_scalar('Loss/val', val_loss / len(val_loader.dataset), epoch_compensated)
            writer.add_scalar('Accuracy/val', val_acc, epoch_compensated)
            writer.add_scalar('ROC_AUC/val', val_roc_auc, epoch_compensated)
            writer.add_scalar('PR_AUC/val', val_pr_auc, epoch_compensated)
            for k, v in val_metric_dict.items():
                writer.add_scalar(f'{k}/val', v, epoch_compensated)

            print(f"Epoch {epoch_compensated + 1}/{args.epochs} - Val Loss: {val_loss / len(val_loader.dataset):.4f} "
                  f"Acc: {val_acc:.4f} ROC AUC: {val_roc_auc:.4f} PR AUC: {val_pr_auc:.4f}")

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--risk_data_dir', type=str, required=True)
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--hidden_layer_activation', default='relu', choices=['relu', 'selu', 'tanh', 'sigmoid', 'elu',
                                                                              'gelu', 'swish', 'leaky_relu'])
    parser.add_argument('--last_layer_activation', default='sigmoid', choices=['linear', 'sigmoid'])
    parser.add_argument('--fcnet_hiddens', type=json.loads, default=[128, 256, 256])
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--risky_state_class_weight', type=float, default=None)
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--l2_regularization', type=float, default=0.0)
    parser.add_argument('--risky_state_rebalancing_fraction', type=float, default=None)
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    training_data_dir = os.path.join(args.risk_data_dir, "train")
    test_data_dir = os.path.join(args.risk_data_dir, "test")

    train_dataset = RiskDataset(training_data_dir)
    num_batches_train_dataset = np.ceil(len(train_dataset) / args.batch_size)

    print("Batches train dataset without rebalancing", num_batches_train_dataset)

    test_dataset = RiskDataset(test_data_dir)
    num_batches_test_dataset = np.ceil(len(test_dataset) / args.batch_size)
    print("Batches test dataset", num_batches_test_dataset)

    if args.risky_state_rebalancing_fraction is not None:
        sampler = RebalancingSampler(train_dataset, batch_size=args.batch_size,
                                     risky_fraction=args.risky_state_rebalancing_fraction)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
        epoch_compensation_factor = num_batches_train_dataset / sampler.num_batches
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
        epoch_compensation_factor = 1.0

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    input_size = train_dataset.x.shape[1]

    risk_network_config = {'input_size': input_size,
                           'fcnet_hiddens': args.fcnet_hiddens,
                           'dropout': args.dropout,
                           'activation': args.hidden_layer_activation,
                           'last_activation': args.last_layer_activation}

    model = RiskNetwork(**risk_network_config)

    log_dir = os.path.join(Path.home(), "risk_results") if args.logdir is None else args.logdir
    log_dir = os.path.join(log_dir, args.experiment_name, datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    args.logdir = log_dir

    # Copy config file if available
    config_file_path = os.path.join(os.path.dirname(args.risk_data_dir), "risk_config.json")
    if os.path.isfile(config_file_path):
        destination_path = os.path.join(log_dir, "risk_config.json")
        shutil.copy(config_file_path, destination_path)

        with open(destination_path, 'r') as f:
            config = json.load(f)

        if "observation_size" in config:
            if train_dataset.state_size != config["observation_size"]:
                raise ValueError("The observation size of the risk data ({}) does not match with the "
                                 "observations size specified in risk_config.json ({}).".format(
                    train_dataset.state_size, config["observation_size"]))
        else:
            config["observation_size"] = train_dataset.state_size

        # action_size to config
        config["action_size"] = train_dataset.action_size
        config["risk_network_config"] = risk_network_config

        # store updated config
        with open(destination_path, 'w') as f:
            f.write(json.dumps(config, sort_keys=True))
            f.flush()

    # Save arguments
    with open(os.path.join(log_dir, "arguments.json"), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    train(model, train_loader, test_loader, epoch_compensation_factor, args)

    torch.save(model.state_dict(), os.path.join(log_dir, "model.pt"))