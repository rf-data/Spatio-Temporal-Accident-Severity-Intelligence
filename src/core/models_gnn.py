## models_gnn.py
# imports 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from src.core.session import session

class GNNPipeline:

    def __init__(self, preprocess, model_fn, config):
        self.preprocess = preprocess
        self.model_fn = model_fn
        self.config = config
        self.result = None

    def fit(self, train_df):
        data = self.preprocess(train_df)
        self.result = self.model_fn(data, self.config)
        return self

    def predict(self, test_df):
        data = self.preprocess(test_df)
        return self.model_fn.predict(data)
    

class SimpleGNN(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, 32)
        self.conv2 = GCNConv(32, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x.squeeze()


class SimpleBinaryGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return self.head(x).squeeze(-1)
    
#####################
# BUILDER FUNCTIONS
#####################
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


def build_simple_binary_gnn(gnn_config, in_dim):

    # in_dim = config.get("in_dim", 1)
    n_epochs = int(gnn_config["n_epochs"])       # , 1)
    learn_rate = float(gnn_config["learn_rate"])
    weight_decay = float(gnn_config["weight_decay"])
    opti_gnn = gnn_config["optimizer"]       #, "adam")
    batch_size = int(gnn_config["batch_size"])
    weighted = gnn_config["weighted"]
    criterion = gnn_config["criterion"]
    pred_threshold = gnn_config["pred_threshold"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_dim = gnn_config["hidden_dim"]   # , 32)
    drop_out = gnn_config["dropout"]    # , 0.2)

    model = SimpleBinaryGCN(in_dim, hidden_dim, drop_out)
    model = model.to(device)

    setting = {
            "model_class": model.__class__.__name__,
            "architecture": str(model),
            "hidden_dim": hidden_dim,
            "in_dim": in_dim,
            "drop_out": drop_out,
            "n_epochs": n_epochs,
            "learn_rate": learn_rate,
            "weight_decay": weight_decay,
            "optimizer": opti_gnn,
            "batch_size": batch_size,
            "device": str(device),
            "weighted": weighted,
            "criterion": criterion,
            "pred_threshold": pred_threshold
            }

    return model, setting


#####################
# GNN ADAPTER
#####################
MODEL_DICT= {
        "SimpleBinaryGCN": build_simple_binary_gnn
}

class GNN_Adapter(BaseModelAdapter):
    
    def __init__(self):
        self.pred_threshold = 0.0
        self.device = None
        self.results = {}


    def build(self, model_name, gnn_config): 
        build_fn = MODEL_DICT[model_name]

        self.logger.info("Start building '%s'", model_name)

        return build_fn(*gnn_config)
    

    def train(self, model, train_data: DataLoader, gnn_config): 

        # n_epochs = int(gnn_config["n_epochs"])       # , 1)
        learn_rate = float(gnn_config["learn_rate"])
        weight_decay = float(gnn_config["weight_decay"])
        opti_gnn = gnn_config["optimizer"]       #, "adam")
        self.pred_threshold = gnn_config.get("pred_threshold", 0.5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if opti_gnn == "adam":
            optimizer = torch.optim.Adam(
                        model.parameters(),
                        lr=learn_rate,
                        weight_decay=weight_decay
                    )
        else:
            raise ValueError("Missing or unknown value for 'optimizer':", opti_gnn)

        
        crit_adapted = gnn.define_criterion(train_data, 
                                            self.device, 
                                            gnn_config)
        self.logger.info("Criterion was defined")

        self.logger.info("Start Training 'Simple Binary GCN'")

        model.train()
        total_loss = 0.0

        for i, data in enumerate(train_data):
            if i % 50 == 0:
                self.logger.info("Processing timepoint '%s'", 
                            i)
            # logger.info("max edge index (%s):\t%s", 
            #             i,
            #             data.edge_index.max())

            data = data.to(self.device)
            optimizer.zero_grad()

            logits = model(data.x, data.edge_index)

            loss = crit_adapted(logits, data.y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss         # model.train(X_train, y_train)
    

    def predict(self, model, test_data): 

        self.logger.info("Start collecting predictions")

        if len(self.results) > 0:
            return self.results
        
        model.eval()
    
        x_test_all = [] 
        y_true_all = []
        y_pred_all = []
        y_proba_all = [] 
        y_logits_all = []

        with torch.no_grad():
            for data in test_data:
                data = data.to(self.device)

                logits = model(data.x, data.edge_index)
                logits = logits.view(-1)            # fix shape if necessary

                probs = torch.sigmoid(logits)
                preds = (probs >= self.pred_threshold).float()

                x_test_all.append(data.x.cpu())
                y_true_all.append(data.y.view(-1).cpu())
                y_pred_all.append(preds.cpu())
                y_proba_all.append(probs.cpu())
                y_logits_all.append(logits.cpu())

        x_test_all = torch.cat(x_test_all)
        y_true_all = torch.cat(y_true_all)
        y_pred_all = torch.cat(y_pred_all)
        y_proba_all = torch.cat(y_proba_all)
        y_logits_all = torch.cat(y_logits_all)

        self.results = {
            "X_test": x_test_all.numpy(), 
            "y_true": y_true_all.numpy(), 
            "y_pred": y_pred_all.numpy(), 
            "y_proba": y_proba_all.numpy(),
            "y_logits": y_logits_all.numpy()
            }
        
        return self.results


    def predict_proba(self, model, test_data):
        
        if len(self.results) > 0:
            return self.results
        
        return self.predict(model, test_data)

