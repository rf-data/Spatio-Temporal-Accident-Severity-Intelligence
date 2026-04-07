## gnn_helper.py
# import
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

# from src.core.models_gnn import SimpleBinaryGCN

from src.core.session import session
import src.utils.path_helper as ph
# import src.utils.visualisation_helper as viz


def get_all_targets(data_list):
    y_all = []

    for data in data_list:
        y_all.append(data.y)

    return torch.cat(y_all)


def define_criterion(data_list, device, gnn_config):
    # setup logger
    logger = session.logger

    # 
    weighted = gnn_config["weighted"]
    crit = gnn_config.get("criterion", "bce_log_loss")

    if crit == "bce_log_loss":
        if weighted: 
            logger.info("Setting 'bce_log_loss - weighted' as criterion")
            y_all = get_all_targets(data_list)

            logger.info("y_all shape:\t%s", 
                        y_all.shape)
            logger.info("positive ratio:\t%s", 
                        y_all.mean().item())

            pos = y_all.sum()
            neg = len(y_all) - pos

            pos_weight_value = neg / max(pos, 1)
            pos_weight_ceil = min(pos_weight_value, 10)

            pos_weight = torch.tensor(
                                [pos_weight_ceil],
                                dtype=torch.float32, 
                                device=device)

            logger.info("pos: %s, neg: %s, pos_weight (raw | ceil):\t%.2f | %.2f",
                        pos,
                        neg,
                        pos_weight_value,
                        pos_weight_ceil)

            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
        else: 
            logger.info("Setting 'bce_log_loss - unweighted' as criterion")
            criterion = nn.BCEWithLogitsLoss()
        
        return criterion

    logger.error("No criterion set")

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


def get_gnn_feature_importance(model, data_list, criterion):
    # setup logger
    logger = session.logger 
    logger.info("Start compiling 'feature importance' for test_data")

    # 
    model.eval()
    importance = []

    for i, data in enumerate(data_list):
        # t_stamp = data.time
        
        data.x = data.x.clone().detach().requires_grad_(True)
        model.zero_grad()

        logits = model(data.x, data.edge_index)
        # probs = torch.sigmoid(logits)
        
        loss = criterion(logits.squeeze(), data.y)
        # loss = probs.mean() # probs[:, 0].mean()    

        # if i < 5:
        #     print(f"[DEBUG] shape of probs and value of loss:\t{probs.shape} | {loss:.4f}")
         
        
        # model.zero_grad()
        loss.backward()

        data_importance = data.x.grad.abs().mean(dim=0)

        importance.append(data_importance.detach().cpu().numpy())

    importance = np.stack(importance)

    return {
        "all": importance,
        "mean": importance.mean(axis=0),
        "std": importance.std(axis=0)
        }


def create_gnn_importance_df(model, 
                            data_list, 
                            feats,
                            criterion,
                            save=True):
    # setup logger
    logger = session.logger
    logger.info("Start creating Coef_df from GNN data")

    # 
    result = get_gnn_feature_importance(model, data_list, criterion)
    mean_importance = result["mean"]
    
    assert len(feats) == len(mean_importance), \
        f"Mismatch: feats={len(feats)}, importance={len(mean_importance)}"
    
    df = pd.DataFrame({
        "feature": feats,
        "importance": mean_importance,
        "std": result["std"]
    })

    df["rank"] = df["importance"].rank(ascending=False)

    if save:
        df_path = ph.create_save_path("coefs", "csv")
        df.to_csv(df_path, index=False)
        logger.info("Saved coef_df in ...%s", ph.shorten_path(df_path))

    # if data_viz:
    #     coef_hm_path = ph.create_save_path("plots", "coef_heat", "png")

    #     viz.viz_odds_ratios(df, top_k=5, save_path=True)
    #     viz.viz_odds_heatmap(df, save_path=coef_hm_path)

    return df.sort_values("importance", ascending=False)


def preprocess_graph_data(train_data, test_data, config):
    # setup logger
    logger = session.logger 
    logger.info("Start preprocessing dataset")

    num_cols = config["num_cols"] 

    X_train = train_data.x
    X_test = test_data.x

    # scaling
    scaler = StandardScaler()
    X_train[num_cols] = torch.tensor(
        scaler.fit_transform(X_train[num_cols].numpy()),
        dtype=torch.float32
    )

    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return train_data, test_data

    