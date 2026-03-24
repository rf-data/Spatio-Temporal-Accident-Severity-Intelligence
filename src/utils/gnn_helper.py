## gnn_helper.py
# import
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.core.gnn_classes import SimpleBinaryGCN
from torch_geometric.loader import DataLoader


def build_gnn(X, y, edge_index, config):

    in_dim = config
    hidden_dim = config
    drop_out = config
    
    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)

    model = SimpleBinaryGCN(in_dim, hidden_dim, drop_out)

    out = model(X, edge_index)

    print(out.shape)  # (831,)

    return out


def build_simple_binary_gnn(config, in_dim):

    # in_dim = config.get("in_dim", 1)
    hidden_dim = config.get("hidden_dim", 32)
    drop_out = config.get("dropout", 0.2)
    
    model = SimpleBinaryGCN(in_dim, hidden_dim, drop_out)

    return model


def get_all_targets(data_list):
    y_all = []

    for data in data_list:
        y_all.append(data.y)

    return torch.cat(y_all)


def define_criterion(data_list, device, config):
    crit = config.get("criterion", "bce_log_loss")

    y_all = get_all_targets(data_list)

    print("y_all shape:", y_all.shape)
    print("positive ratio:", y_all.mean().item())

    pos = y_all.sum()
    neg = len(y_all) - pos

    pos_weight_value = neg / max(pos, 1)
    pos_weight_ceil = min(pos_weight_value, 10)

    pos_weight = torch.tensor(
                        [pos_weight_ceil],
                        dtype=torch.float32, 
                        device=device)

    print(f"pos: {pos}, neg: {neg}, pos_weight (raw | ceil): {pos_weight_value:.2f} | {pos_weight_ceil:.2f}")

    if crit == "bce_log_loss":
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
        return criterion

    return None


def baseline_persistence(data_list):
    y_true_all = []
    y_pred_all = []

    for data in data_list:
        x = data.x
        y_true = data.y

        # Feature 0 = n_accidents_t
        y_pred = (x[:, 0] > 0).int()

        y_true_all.append(y_true.cpu())
        y_pred_all.append(y_pred.cpu())

    return torch.cat(y_true_all), torch.cat(y_pred_all)


def baseline_zero(data_list):
    y_true_all = []
    y_pred_all = []

    for data in data_list:
        y_true = data.y
        y_pred = torch.zeros_like(y_true)

        y_true_all.append(y_true.cpu())
        y_pred_all.append(y_pred.cpu())

    return torch.cat(y_true_all), torch.cat(y_pred_all)


def baseline_one(data_list):
    y_true_all = []
    y_pred_all = []

    for data in data_list:
        y_true = data.y
        y_pred = torch.ones_like(y_true)

        y_true_all.append(y_true.cpu())
        y_pred_all.append(y_pred.cpu())

    return torch.cat(y_true_all), torch.cat(y_pred_all)


def baseline_prob_persistence(data_list):
    probs_all = []
    y_true_all = []

    for data in data_list:
        x = data.x
        y_true = data.y

        probs = torch.clamp(x[:, 0] / (x[:, 0].max() + 1e-6), 0, 1)

        probs_all.append(probs.cpu())
        y_true_all.append(y_true.cpu())

    return torch.cat(y_true_all), torch.cat(probs_all)


    
def collect_predictions(model, data_list, device):
    model.eval()

    y_true_all = []
    probs_all = []    

    with torch.no_grad():
        for data in data_list:
            data = data.to(device)

            logits = model(data.x, data.edge_index)
            probs = torch.sigmoid(logits)

            y_true_all.append(data.y.cpu())
            probs_all.append(probs.cpu())

    y_true_all = torch.cat(y_true_all)
    probs_all = torch.cat(probs_all)

    return y_true_all, probs_all


def train_n_epoch(train_data, test_data, config):
    
    n_epoch = config.get("n_epochs", 1)
    learn_rate = config.get("learn_rate", 1e-3)
    weight_decay = config.get("weight_decay", 1e-4)
    opti_gnn = config.get("optimizer", "adam")
    batch_size = config.get("batch_size", 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    
    data_sample = data = next(iter(loader))
    print("X shape:", data_sample.x.shape)
    in_dim = data_sample.x.shape[1]

    model = build_simple_binary_gnn(config, in_dim)
    model = model.to(device)

    if opti_gnn == "adam":
        optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=float(learn_rate),
                    weight_decay=float(weight_decay)
                )
    else:
        ValueError("Missing or unknown value for 'optimizer':", opti_gnn)

    model.train()
    total_loss = 0.0
    total_len = len(loader)

    criterion = define_criterion(loader, device, config)
    
    for i, data in enumerate(loader):
        # print(f"Processing {i} of {total_len}")

        y_train = data.y
        X_train = data.x

        data = data.to(device)
        optimizer.zero_grad()

        logits = model(X_train, data.edge_index)

        loss = criterion(logits, y_train)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # print(f"[DEBUG - {i}] loss | total_loss | total_loss (rel):")
        # print(f"{loss:.4f} | {total_loss:.4f} | {total_loss/total_len:.4f}")

    y_true, probs = collect_predictions(model, test_data, device)

    # print("y mean:", np.mean(y_true_all))   # .mean()
    # print("preds_50_gnn mean:", np.round(np.mean(preds_50_gnn), 3))
    # print("preds_90_gnn mean:", np.round(np.mean(preds_90_gnn), 3))
    # print("pred_base mean:", preds_base.float().mean())

    return {
        # "pred_base": baseline, 
        "y_true": y_true,
        "total_loss (abs)": total_loss,
        "total_loss (rel)": total_loss / len(loader),
        "probs": probs,
        # "preds_50": torch.cat(preds_50_gnn),
        # "preds_90": torch.cat(preds_90_gnn)
        }
