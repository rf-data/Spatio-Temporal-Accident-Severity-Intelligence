## ???.py
# import
import click
import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

import h3
import torch
from torch_geometric.data import Data
#region agent log imports
import json
import time
#endregion
# print(torch.__version__)
# print(torch.cuda.is_available())

# from torch_geometric.data import Data
# from torch_geometric.nn import GCNConv

from src.utils.file_helper import get_yaml_config
# from src.core.gnn_classes import SimpleGNN
import src.utils.ml_prep_helper as prep

from src.core.session import session

import src.utils.general_helper as gh
import src.utils.file_helper as fh
import src.utils.path_helper as ph
import src.utils.df_helper as dfh


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

# Part A: node definition
def build_node_index(df, config, gnn_config):
     # , "n_accidents")
    train_model = gnn_config.get("train_model", None)
    h3_col = config["h3_col"]

    print("head of df (build nodes):\n", df.head(3))
    # 1. Filter active cells
    if train_model:
        active_cells = df[h3_col].dropna().astype(str).unique().tolist()
  
    else:
        min_events = config["min_events"]   # , "h3_res4")
        target_col = config["target_col"]  
        
        cell_activity = df.groupby(h3_col)[target_col].sum()
        active_cells = cell_activity[cell_activity >= min_events].index.tolist()
        # print("[DEBUG] total accidents (post_filter):\t", len(active_cells))
    
    # 2. Create mapping
    h3_to_node = {h: i for i, h in enumerate(active_cells)}
    node_to_h3 = {i: h for h, i in h3_to_node.items()}

    return h3_to_node, node_to_h3


# Part B: edge construction
def build_edge_index(h3_to_node, gnn_config):
    k = gnn_config["neighbor_distance"]     # , 1)
    include_self_loops = gnn_config["include_self_loops"]       # , True) 

    edges = set()

    for h3_cell, node_id in h3_to_node.items():
        
        # get neighbors (including h3_cell itself)
        neighbors = set(h3.grid_disk(h3_cell, k)) - {h3_cell}

        for n in neighbors:
            if n not in h3_to_node:
                continue
                
            neighbor_id = h3_to_node[n]

            if include_self_loops:
                edges.add((node_id, node_id))

            if not include_self_loops and node_id == neighbor_id:
                continue
            
            # add BOTH directions (undirected / bidirectional graph)
            edges.add((node_id, neighbor_id))
            edges.add((neighbor_id, node_id))

    # convert to tensor
    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    #   [1, 1, 1]   # 
    
    return edge_index

   
# Part C: snapshot building

def create_full_df(df, graph_data, config):
    h3_to_node = graph_data.get("h3_to_node", {})
    period_col = config.get("period_col", "time_bin")
    h3_col = config.get("h3_col", "h3_res4")

    all_nodes = list(h3_to_node.keys())
    all_times = sorted(df[period_col].unique())

    df_full = dfh.create_complete_grid(
                                df, 
                                h3_col, 
                                all_nodes, 
                                period_col,
                                all_times
                                )

    return df_full
    

def create_snapshots(df_full,  
                    config,
                    graph_data):
    # setup logger 
    logger = session.logger 

    # 
    h3_to_node = graph_data["h3_to_node"]        # , {})
    period_col = config["period_col"]       # , "time_bin")
    h3_col = config["h3_col"]       # , "h3_res4")
    target_col = config["target_col"]
    
    all_times = sorted(df_full[period_col].unique())

    feature_cols = [col for col in df_full.columns 
                    if col not in [target_col, 
                                   period_col, 
                                   h3_col]]

    assert not df_full.duplicated([h3_col, period_col]).any()
    assert df_full[h3_col].notna().all()
    assert df_full[period_col].notna().all()
    assert df_full.groupby([h3_col, period_col]).size().min() >= 1

    snapshots = []

    for i in range(len(all_times) - 1):
        t = all_times[i]
        t_next = all_times[i + 1]

        df_t = df_full[df_full[period_col] == t].copy()
        df_t1 = df_full[df_full[period_col] == t_next].copy()

        df_t["node_id"] = df_t[h3_col].map(h3_to_node)
        df_t1["node_id"] = df_t1[h3_col].map(h3_to_node)

        # df_t = df_t.sort_values("node_id")
        # df_t1 = df_t1.sort_values("node_id")

        num_nodes = len(h3_to_node)
        node_ids = np.arange(num_nodes)

        df_t = df_t.set_index("node_id").reindex(node_ids)
        df_t1 = df_t1.set_index("node_id").reindex(node_ids)

        df_t[feature_cols] = df_t[feature_cols].fillna(0.0)
        df_t1[target_col] = df_t1[target_col].fillna(0.0)

        X = df_t[feature_cols].values.astype("float32")   
        y = df_t1[target_col].values.astype("float32")

        assert df_t.index.max() == num_nodes - 1
        assert df_t.shape[0] == num_nodes

        if np.isnan(X).sum() > 0:
            logger.info("NaNs in X:\t%s", 
                        np.isnan(X).sum())
        if np.isnan(y).sum() > 0:
            logger.info("NaNs in y:\t%s", 
                        np.isnan(y).sum())

        assert not np.isnan(X).any(), "NaNs in features!"
        assert not np.isnan(y).any(), "NaNs in target!"

        # print(f"X shape ({i}):", X.shape)
        # print(f"y shape ({i}):", y.shape)
        # print(f"num_nodes ({i}):", num_nodes)
        # 
        snapshots.append({
            "time": t,
            "time_next": t_next,
            "X": torch.tensor(X, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32)
            })
    
    logger.info("length 'snapshots':\t%s", 
                len(snapshots))
    logger.info("shape of first 5 'snapshots':")
    for info in snapshots[:5]:
        logger.info("snap '%s{}' shape X | y:\t%s | %s",
        info["time"],
        info["X"].shape,
        info["y"].shape)
        
    return snapshots


# build_node_index(df_agg, gnn_dict)
def create_data_from_snapshots(
                            snapshots, 
                            edge_index,
                            feat_folder,
                            timestamp
                            ):
    # setup logger
    logger = session.logger

    # now = session.now

    # if edge_index is None: 
    #     if timestamp is not None:
    #         edge_index = torch.load(f"{graph_folder}/{timestamp}_edge_index.pt")

    #     else:
    #         raise ValueError
    # else:
    #     h3_to_node, _ = build_node_index(df, gnn_settings)
    #     edge_index = build_edge_index(h3_to_node, gnn_settings)
    #     graph_data = {"h3_to_node": h3_to_node}

    # snapshots = create_snapshots(df, target_col, gnn_settings, graph_data)
    # torch.save(snapshots, f"{feat_folder}/{timestamp}_snapshots_base.pt")

    data = []
    for snap in snapshots:
        data.append(
            Data(
                x=snap["X"],
                edge_index=edge_index,
                y=snap["y"]
                )
            )
    data_path = f"{feat_folder}/{timestamp}_data_base.pt"
    torch.save(data, data_path)
    logger.info("Data lists as '%s'",
                ph.shorten_path(data_path))

    return data

@click.command()
@click.option("--name", prompt="Name of 'config_file' (no suffix)",
              help='The config_file to use.')
def prepare_static_graph(name):   

    run_prepare_static_graph(name)

    return 


def run_prepare_static_graph(name):
    # (1) load config + parse arguments
    gh.load_env_vars()
    
    gnn_folder = os.getenv("FOLDER_GNN")
    processed_folder = os.getenv("PATH_PROCESSED") 
    graph_folder = Path(f"{gnn_folder}/graph")
    feat_folder = Path(f"{gnn_folder}/features")

    config = get_yaml_config(name)
    general_config = config.get("general_args", {})
    # data_folder = general_args.get("data_folder", None)

    gnn_config = config.get("gnn_settings", {})
    h3_col = gnn_config.get("h3_col", "h3_res4")
    period_col = gnn_config.get("period_col", "time_bin")
    # target_col = gnn_dict.get("target_col", "n_accidents")

    # df_agg, now = prep.merge_df_ml_ready(general_args, 
    #                                      gnn_dict, 
    #                                      save_df=True)

    df_agg = pd.read_parquet(f"{processed_folder}/ml_ready/2026-03-22_12:32_df_merged.parquet")
    now = datetime.now().strftime("%Y-%m-%d_%H:%M")
    # duplicates = df_agg.duplicated(subset=[h3_col, period_col], keep=False)

    # print(f"{"="*10} [DEBUG] df_agg {"="*10}")
    # print("[DEBUG] shape:", df_agg.shape)
    # print("[DEBUG] head:\n", df_agg.head(10), "\n")
    # print("[DEBUG] n_unique h3_cells:", df_agg[h3_col].nunique())
    # print("[DEBUG] n_unique time_bins:", df_agg[period_col].nunique())
    # print("[DEBUG] total accidents (pre_filter):", df_agg[target_col].sum())

    # print("[DEBUG] duplicates:\n", df_agg[duplicates].sort_values([h3_col, period_col]).head(20))
    # print("[DEBUG] Num duplicates:", duplicates.sum())

    assert not df_agg.duplicated([h3_col, period_col]).any(), \
    "Duplicate (h3, time_bin) found!"

    graph_meta, edge_index = create_edges_indexes(df_agg, 
                                                  general_config,
                                                  gnn_config, 
                                                  graph_folder, 
                                                  now)
    
    df_full = create_full_df(df_agg, graph_meta, config)
    
    snapshots = create_snapshots(df_full, config, graph_meta)
    torch.save(snapshots, f"{feat_folder}/{now}_snapshots_base.pt")

    create_data_from_snapshots(snapshots, 
                        edge_index,
                        feat_folder,
                        now)
    

    return 


def create_edges_indexes(df, config, gnn_config, folder, timestamp):
    # Part A: build node_idx (incl. sanity checks)
    h3_to_node, node_to_h3 = build_node_index(df, 
                                              config,
                                              gnn_config)
    total_nodes = len(h3_to_node)
    print("Total nodes:", total_nodes)

    assert len(h3_to_node) == len(node_to_h3)
    assert len(set(h3_to_node.values())) == len(h3_to_node)

    # Part B: build edge_idx (incl. sanity checks)
    sample_cells = list(h3_to_node.keys())[:5]
    print(sample_cells)
    print([type(x) for x in sample_cells])
    assert all(isinstance(x, str) for x in sample_cells)

    edge_index = build_edge_index(h3_to_node, gnn_config)

    num_edges = edge_index.shape[1]
    edges_per_node = num_edges / total_nodes
    unique_nodes = len(torch.unique(edge_index))
    num_isolated = total_nodes - unique_nodes
    num_isolated_rel = 100*num_isolated / total_nodes

    print(f"Edges: {num_edges}")
    print(f"Edges per node: {edges_per_node:.2f}")

    print("Unique nodes:", unique_nodes)
    print(f"Isolated nodes (abs | rel):\t{num_isolated} | {num_isolated_rel:.2f}%")

    # assert len(unique_nodes) == len(h3_to_node)
    assert edge_index.shape[0] == 2
    assert edge_index.max().item() < len(h3_to_node)
    assert edge_index.min().item() >= 0
        
    torch.save(edge_index, f"{folder}/{timestamp}_edge_index.pt")
        
    graph_meta = {
            "h3_to_node": h3_to_node,
            "node_to_h3": node_to_h3,
            "info": {
                "Total_nodes": total_nodes, 
                "Unique_nodes": unique_nodes,
                "Isolated_nodes (abs)": num_isolated,
                "Isolated_nodes (rel., in %)": num_isolated_rel,
                "Total_edges": num_edges,
                "Edges_per_node": edges_per_node
                },
            "config": gnn_config
            }
    
    fh.save_dict(graph_meta, f"{folder}/{timestamp}_graph_meta.json")

    return graph_meta, edge_index

    # graph_path = "/home/robfra/0_Portfolio_Projekte/Road_accidents/GNN/graph/2026-03-20_13:34_graph_meta.json"
    # graph_meta = fh.load_dict(graph_path)
    # edge_index = torch.load(f"{graph_folder}/edge_index.pt")
    
    # level = 1

    # if level == 1:
    #     target_col = config.get("target_lvl_1", "has_accident")
    #     df_agg["has_accident"] = (df_agg["n_accidents"] > 0).astype("int")
    
    # elif level == 2:
    #     target_col = config.get("target_lvl_2", "n_accidents")

    # graph_meta["now"] = now
    # graph_meta["save_folder"] = feat_folder
    
    # full_index = pd.MultiIndex.from_product(
    #     [all_nodes, all_times],
    #     names=[h3_col, "time_bin"]
    # )

    # df_full = df_all.set_index([h3_col, "time_bin"])\
    #                 .reindex(full_index)\
    #                 .fillna(0)\
    #                 .reset_index()

if __name__ == "__main__":
    prepare_static_graph()



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

