## ???.py
# import
import click
import os
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
# import numpy as np
# import pandas as pd
# from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score

# import h3
import torch
from torch_geometric.loader import DataLoader
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv


# print(torch.__version__)
# print(torch.cuda.is_available())

# from torch_geometric.data import Data
# from torch_geometric.nn import GCNConv

from src.core.session import session
from src.core.logger import ModelLogger
from src.utils.file_helper import get_yaml_config
import src.utils.general_helper as gh
import src.utils.file_helper as fh
# import src.utils.gnn_helper as gnn

# import src.utils.path_helper as ph
# import src.utils.df_helper as dfh
import src.utils.split_helper as split
import src.utils.gnn_helper as gnn
        # (build_simple_binary_gnn, 
        #                           baseline_prob_persistence,
        #                           baseline_persistence, 
        #                           baseline_zero, 
        #                           baseline_one)  # as gnn

# Approach:   Static Graph + temporal features
"""
node = h3_cell
sample = t_snapshot
target = n_accidents per (h3_Cell, time)
        = has_accident (-> binary)
features = [weather, datetime, lags, rollings, zero_feats, exposure?]
graph = h3_neighborhood
"""

"""
# installing torch
uv pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

# installing torch_geometry
uv pip install torch-geometric -f https://data.pyg.org/whl/torch-2.10.0+cpu.html

"""




# Part D: time-aware train-test-split


# Part F: comparison against non-GNN approaches



    # gh.load_env_vars()

def train_n_epoch(meta, 
                  train_data, 
                  test_data, 
                  feats,
                  gnn_config):
    # setup logger
    logger = session.logger

    # 
    # n_epochs = int(gnn_config["n_epochs"])       # , 1)
    # learn_rate = float(gnn_config["learn_rate"])
    # weight_decay = float(gnn_config["weight_decay"])
    # opti_gnn = gnn_config["optimizer"]       #, "adam")
    batch_size = int(gnn_config["batch_size"])
    # weighted = gnn_config["weighted"]
    # criterion = gnn_config["criterion"]
    # pred_threshold = gnn_config["pred_threshold"]
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 
    loaded_train = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    data_sample = next(iter(loaded_train))
    logger.info("Start processing '%s'", meta.name)
    logger.info("X shape:\t%s", data_sample.x.shape)

    in_dim = data_sample.x.shape[1]

    model, settings = meta.adapter.build(meta.name, 
                                         gnn_config,
                                         in_dim)
    # gnn.build_simple_binary_gnn(gnn_config, 
    #                                               in_dim)
    
    total_loss = meta.adapter.train(meta.name, 
                                    loaded_train, 
                                    gnn_config)


        # print(f"[DEBUG - {i}] loss | total_loss | total_loss (rel):")
        # print(f"{loss:.4f} | {total_loss:.4f} | {total_loss/total_len:.4f}")

    # X_test, y_test, y_pred, y_proba 
    prediction_dict = meta.adapter.predict()
    
    # now = session.now
    # folder = os.getenv("PATH_EVALUATED")
    # torch.save(results, f"{folder}/{now}_predictions.pt")

    folder = session.save_folder
    # timestamp=session.now

    ml_logger = ModelLogger(base_path=folder)
    ml_logger.log_run(
                    model,
                    prediction_dict["X_test"],
                    prediction_dict["y_true"],
                    y_pred=prediction_dict["y_pred"],
                    y_proba=prediction_dict["y_proba"],
                    extra_params={
                            "model_params": settings,
                            "y_logits": prediction_dict["y_logits"],
                            "total loss (abs)": total_loss,
                            "total loss (rel)": total_loss / len(loaded_train)
                            }
                    )
    
    gnn.create_gnn_importance_df(
                            model, 
                            test_data, 
                            feats, 
                            crit_adapted,
                            save=True
                            )
    # print("y mean:", np.mean(y_true_all))   # .mean()
    # print("preds_50_gnn mean:", np.round(np.mean(preds_50_gnn), 3))
    # print("preds_90_gnn mean:", np.round(np.mean(preds_90_gnn), 3))
    # print("pred_base mean:", preds_base.float().mean())
    
    return # results, settings


def predict_by_simple_bin_gcn(
                        data,
                        general_config,
                        gnn_config, 
                        preprocess: bool=False
                        ):
    # setup logger
    # logger = session.logger

    #
    train_data = data["train"]
    test_data = data["test"]
    feats = data["feats"]

    if preprocess:
        train_data, test_data = gnn.preprocess_graph_data(
                                                train_data,
                                                test_data, 
                                                general_config
                                                )

    # gnn_config = config.get("simple_bin_gcn", {})
    train_n_epoch(train_data, test_data, feats, gnn_config)
    # meta["time"] = now 
    
    # gnn_model, settings = build_logreg_model(config)
    
    return  # results, meta


@click.command()
@click.option("--name", prompt="Name of 'config_file' (no suffix)",
              help='The config_file to use.')
def train_basic_bin_gnn(name):   

    run_train_basic_bin_gnn(name)


def run_train_basic_bin_gnn(name):

    gh.load_env_vars()
    gnn_folder = os.getenv("FOLDER_GNN")
    eval_folder = os.getenv("PATH_EVALUATED")

    graph_folder = Path(f"{gnn_folder}/graph")
    feat_folder = Path(f"{gnn_folder}/features")

    config = get_yaml_config(name)
    general_args = config.get("general_args", {})

    build_time = general_args.get("preparation_time", None)
    # snapshots_name = general_args.get("snapshots_name", None) data_base
    # edge_idx_name = general_args.get("edge_idx_name", None)
    data_name = general_args.get("data_name", None) 

    if (build_time is None) or (data_name is None): # or (edge_idx_name is None):
        raise ValueError(f"""
                         At least one of 'build_time' and 'data_name' is None:\n
                         {build_time}\t{data_name}
                         """) # \t{edge_idx_name}

    gnn_dict = config.get("gnn_settings", {})
    # target_col = gnn_dict.get("target_col", "n_accidents")
    
    # edge_index = torch.load(f"{graph_folder}/{build_time}_{edge_idx_name}.pt")
    data_list = torch.load(f"{feat_folder}/{build_time}_{data_name}.pt",
                           weights_only=False)   
    # split data
    train_data, test_data, val_data = split.simple_time_split(data_list, gnn_dict)
    
    # X_train = train_data.drop(column=target_col)
    # y_train = train_data[target_col]
    # X_test = test_data.drop(column=target_col)
    # y_test = test_data[target_col]
    # X_val = val_data.drop(column=target_col)
    # y_val = val_data[target_col]
    
    # model = build_gnn(X_data, edge_index, config)
    results = train_n_epoch(
                        train_data,
                        test_data,
                        feats, 
                        gnn_dict
                        )

    save_path = f"{eval_folder}/gnn/test_run_v2.json"
    fh.save_dict(results, save_path)

    probs_gnn = results.get("probs")
    y_true = results.get("y_true")
    # preds_50 = results.get("preds_50")
    # preds_90 = results.get("preds_90")

    # gnn
    for t in [0.3, 0.5, 0.7, 0.9]:
        preds = (probs_gnn >= t).int()
        print(f"\nThreshold {t}")
        evaluate(y_true, preds)

    print("y mean:", y_true.float().mean())
    print("pred mean:", preds.float().mean())
    print("prob mean:", probs_gnn.mean())

    # evaluate(y_true_all, preds_50, "GNN @0.5")
    # evaluate(y_true_all, preds_90, "GNN @0.9")
    # Baselines
    y_true_b, y_pred_persist = baseline_persistence(test_data)
    evaluate(y_true_b, y_pred_persist, "Persistence")

    y_true_b, y_pred_zero = baseline_zero(test_data)
    evaluate(y_true_b, y_pred_zero, "Always Zero")

    y_true_b, y_pred_one = baseline_one(test_data)
    evaluate(y_true_b, y_pred_one, "Always One")

    # y_true_b, y_pred_one = baseline_prob_persistence(test_data)
    # evaluate(y_true_b, y_pred_one, "prob_persistence")

    plt.hist(probs_gnn.detach().cpu().numpy(), bins=50)
    plt.title("Prediction Distribution")
    plt.show()

    print("ROC-AUC:", roc_auc_score(y_true, probs_gnn))
    print("PR-AUC :", average_precision_score(y_true, probs_gnn))

    return 
"""
X shape: torch.Size([831, 3])
y_all shape: torch.Size([577545])
positive ratio: 0.20456241071224213
pos: 118144.0, neg: 459401.0, pos_weight (raw | ceil): 3.89 | 3.89

Threshold 0.3
Precision: 0.6071416562993847
Recall   : 0.9991976316299145
F1       : 0.7553254901960784

Threshold 0.5
Precision: 0.6092690597365706
Recall   : 0.9969842016434718
F1       : 0.7563335642171988

Threshold 0.7
Precision: 0.6390740222159993
Recall   : 0.9471266911988491
F1       : 0.7631872296785125

Threshold 0.9
Precision: 0.8283419023136247
Recall   : 0.35661123869075617
F1       : 0.49857841904725064

y mean: tensor(0.2900)
pred mean: tensor(0.1248)
prob mean: tensor(0.3998)

Persistence
Precision: 0.7411800402725292
Recall   : 0.7434357966964558
F1       : 0.742306204762694

Always Zero
Precision: 0.0
Recall   : 0.0
F1       : 0.0

Always One
Precision: 0.2899558764540714
Recall   : 1.0
F1       : 0.4495593713656689
ROC-AUC: 0.9236070835025112
PR-AUC : 0.7902852100749309
"""

if __name__ == "__main__":
    train_basic_bin_gnn()


    # gnn_folder = os.getenv("FOLDER_GNN")
    # graph_folder = Path(f"{gnn_folder}/graph")
    
    # [...]

    # # now = datetime.now().strftime("%Y-%m-%d_%H:%M")   
    # # fh.save_dict(graph_meta, f"{graph_folder}/{now}_graph_meta.json")
    # graph_path = "/home/robfra/0_Portfolio_Projekte/Road_accidents/GNN/graph/2026-03-20_13:34_graph_meta.json"
    # graph_meta = fh.load_dict(graph_path)

    # snapshots = torch.load()
    
    # # edge_path = ""
    # edge_index = torch.load(f"{graph_folder}/edge_index.pt")

    # for time, X, y in snapshots[:5]:
    #     print(f"[DEBUG] snap '{time}' shape X | y:\t{X.shape} | {y.shape}")
    # for i, (time, data) in enumerate(snapshots.items()):
    #     if i < 5:
    #         print("Processing time:\t", time)

    #         X = data["X"]
    #         y = data["y"]
    #         build_gnn(X, y, edge_index)
            
    #     else: 
    #         continue

# def build_gnn_model():

#     # load df_ml_ready
#     df= ""

#     if scope == target_lvl_1:
#         df["has_accident"] = 1 if df["n_accident"] > 0 else 0



# def run_train_static_graph():

    # return 


    # full_index = pd.MultiIndex.from_product(
    #     [all_nodes, all_times],
    #     names=[h3_col, "time_bin"]
    # )

    # df_full = df_all.set_index([h3_col, "time_bin"])\
    #                 .reindex(full_index)\
    #                 .fillna(0)\
    #                 .reset_index()



# # Spatio_temporal_learning
# Approach 1 – Static Graph + temporal features
# Node features enthalten:
# t
# month
# week
# lags

# Einfachster Ansatz.
# Approach 2 – Temporal Graph
# Graph pro Zeitpunkt:
# G_t

# Sequence:
# G1 → G2 → G3 → ...

# Modelle:
# T-GCN
# DCRNN
# ST-GCN

# Approach 3 – Graph + RNN
# Pipeline:
# GNN → embeddings → LSTM
# Sehr häufige Lösung.

# Welchen Output liefert das GNN?
# Das hängt nicht am GNN selbst, sondern an deiner Zieldefinition.
# Ein GNN ist erstmal nur ein Modell, das Information über Nachbarschaften mitverarbeitet.
# Der Output kann sehr unterschiedlich sein.

# In deinem Fall sind drei Varianten realistisch:
# A. Regression
# Vorhersage von:
# n_accidents
# accident_rate
# risk_score
# Dann wäre der Output pro Node etwa:
# ein einzelner Wert pro H3-Zelle und Zeitpunkt
# Also z. B.:
# ŷ(h3_i, t) = erwartete Unfallanzahl
# Das wäre tatsächlich eine Form von Forecasting, wenn deine Features nur Informationen enthalten, die bis zum Vorhersagezeitpunkt verfügbar sind.

# B. Klassifikation
# Vorhersage von:
# Unfall ja/nein
# hoher Risk-Bin / niedriger Risk-Bin
# Hotspot ja/nein
# Dann ist der Output pro Node:
# Wahrscheinlichkeit oder Klasse
# Das ist oft robuster als direkte Count-Regression, gerade bei vielen Nullen.

# C. Node Embeddings als Zwischenschritt
# Das ist unterschätzt und für dich sehr interessant.
# Das GNN muss nicht direkt Endmodell sein.
# Es kann auch pro Node strukturinformierte Repräsentationen lernen, also Embeddings.
# Diese Embeddings kannst du dann:
# an ein klassisches ML-Modell weitergeben
# für Clustering nutzen
# mit tabellarischen Features kombinieren
# Das ist oft methodisch sauberer als sofort „GNN ersetzt alles“.


# Kann es ergänzend zu einem klassischen ML-Modell genutzt werden?
# Ja, absolut. Und genau das ist wahrscheinlich der vernünftigste Weg.
# Ich würde dir sogar davon abraten, das GNN sofort als Hauptmodell zu sehen.

# Der bessere Ansatz:
# Option 1: Klassisches ML als Baseline, GNN als Ergänzung
# Baseline: XGBoost / LightGBM / CatBoost / RF
# GNN: prüft, ob Nachbarschaftsinformation echten Mehrwert bringt
# Dann stellst du sauber fest:
# bringt Graphstruktur überhaupt etwas?
# oder reichen gute tabellarische Features schon aus?
# Das ist wissenschaftlich sauberer als direkt GNN-Hype.

# Option 2: GNN-Output als Zusatzfeature
# Zum Beispiel:
# GNN lernt Node-Embedding
# Embedding wird zusätzlich in klassisches Modell eingespeist
# Dann hast du:
# tabellarische Stärke klassischer Modelle
# plus räumliche Strukturinformation
# Das ist oft stärker als ein „reines“ GNN.

# Option 3: Ensemble
# klassisches Modell sagt A
# GNN sagt B
# finaler Output kombiniert beide
# Das ist später interessant, aber für jetzt noch zu früh.

# Meine klare Empfehlung für dein Vorgehen
# Phase 1

# Baue zuerst ein nicht-temporales, aber zeitabhängiges Snapshot-GNN:

# statischer H3-Graph

# pro Zeitpunkt ein Snapshot

# Ziel: Regression oder Klassifikation pro Node

# Phase 2

# Vergleiche gegen:

# Poisson/NB-artige Ansätze oder einfache Regression

# XGBoost/LightGBM

# evtl. klassisches Modell mit Lag-Features

# Phase 3

# Erst wenn das sinnvoll aussieht:

# Temporal Extension

# mehrere Zeitschritte als Input

# evtl. ST-GNN

# Nicht früher.
############################################

# # edge_construction
# edges = []

# for h in h3_cells:
    
#     neighbors = h3.grid_disk(h, 1)

#     for n in neighbors:
#         if n in h3_cells:
#             edges.append((h, n))

# # edge index
# edge_index = torch.tensor(edges).t().contiguous()

# # Node Features
# X shape

# [num_nodes , num_features]

# # target
# [num_nodes]

# # graph object
# data = Data(
#     x = X,
#     edge_index = edge_index,
#     y = y
# )


# # base_model
# class GNN(torch.nn.Module):

#     def __init__(self, in_dim):
#         super().__init__()

#         self.conv1 = GCNConv(in_dim, 64)
#         self.conv2 = GCNConv(64, 1)

#     def forward(self, x, edge_index):

#         x = self.conv1(x, edge_index).relu()
#         x = self.conv2(x, edge_index)

#         return x

# # evaluation
# Dein Target könnte sein:

# accident count

# oder

# risk score

# Loss:
# Poisson loss
# MSE
# NB regression



###

