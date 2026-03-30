## 
# imports
import json
import os
from pathlib import Path
from typing import Dict, List, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.core.session import session

# if __name__ == "__main__":
#     create_logger()


class EnsembleManager:

    def __init__(self, 
                 pred_list=None, 
                 proba_list=None, 
                 weights=None,
                 threshold=None):
        self.pred_list = pred_list 
        self.proba_list = proba_list
        self.weights = weights          # e.g. based on F2 or PR-AUC
        self.threshold = threshold

    
    def hard_voting(self):
        return np.round(np.mean(self.pred_list, axis=0)).astype(int)


    def soft_voting(self):
        avg_proba = np.mean(self.proba_list, axis=0)
        return (avg_proba >= self.threshold).astype(int), avg_proba


    def weighted_voting(self):
        
        weighted = np.zeros_like(self.proba_list[0])
        
        for p, w in zip(self.proba_list, self.weights):
            weighted += p * w
        
        weighted /= sum(self.weights)
        
        return (weighted >= self.threshold).astype(int), weighted


class EvaluationManager:
    
    def __init__(self, timestamp=None):
        self.base_path = Path(os.getenv("PATH_EVALUATED"))
        self.timestamp = timestamp
        self.folder = Path(f"{self.base_path}/{self.timestamp}") 

    def list_runs(self):
        
        run_list = [p.name for p in self.base_path.iterdir() if p.is_dir()]

        return run_list 
    
    def load_predictions(self, model_name: str):
        # setup logger
        logger = session.logger
        
        predicts = [f for f in self.folder.iterdir() if f.suffix == "parquet"]
        metas = [f for f in self.folder.iterdir() if f.suffix == "json"]
        
        model_predict = [f for f in predicts if str(f.name).split("_")[1] == model_name]
        model_meta = [f for f in metas if str(f.name).split("_")[1] == model_name]
        
        if len(model_predict) != 1:    # ) or ():
            logger.error("Found invalid number (n=%s) of result files for %s @ %s", 
                         len(model_predict), 
                         model_name,
                         self.timestamp) 
            
        if len(model_meta) != 1:
            logger.error("Found invalid number (n=%s) of meta data files for %s @ %s", 
                         len(model_meta),
                         model_name,
                         self.timestamp) 
                
        predict_path = f"{self.folder}/{self.timestamp}_{model_predict[0]}_results.parquet"
        meta_path = f"{self.folder}/{self.timestamp}_{model_meta[0]}_meta.json"
         
        df = pd.read_parquet(predict_path)
        
        with open(meta_path) as f:
            meta = json.load(f)
        
        return df, meta
    

    def compare_roc_pr_curves(self, run_name):
        
        plt.figure()
        
        # for run in run_names:
        df, _ = self.load_predictions(run_name)

        eval.compile_roc_pr_auc(
                            df["y_true"],
                            df["y_proba"],
                            data_viz=True, 
                            suffix=run_name)
            # plot_pr_curve(
            #     df["y_true"],
            #     df["y_proba"],
            #     label=meta["model_class"]
            # )
        
        # plt.show()

    """
    from sklearn.metrics import precision_recall_curve, auc


    def plot_pr_curve(y_true, y_proba, label=None):
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        
        plt.plot(recall, precision, label=f"{label} (AUC={pr_auc:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
    """


    def optimize_threshold(self, 
                           run_name, 
                           metric,
                           beta=2):
        
        df, _ = self.load_predictions(run_name)
        
        res = eval.find_best_threshold(
                                df["y_true"],
                                df["y_proba"],
                                metric, 
                                beta=beta
                                )
        
        return res


    def describe_run(self, 
                     run_name, 
                     metric_fn: List[Callable] | Callable):
        
        results = []
        
        if isinstance(metric_fn, Callable):
            metric_fn = [metric_fn]

        # for run in run_names:
        df, meta = self.load_predictions(run_name)
            
        score_all = []
        for fn in metric_fn:
            score = fn(df["y_true"], df["y_pred"])
            score_all.append({"", score})
            
        results.append({
            "run": run_name,
            "model": meta.get("model_class"),
            "scores": score_all
            })
        
        # pd.DataFrame(results).sort_values("score", ascending=False)
        return results
    

    def get_errors(self, run_name: str):

        # if isinstance(run_names, str):
        #     run_names = [run_names]
        
        # errors = []
        # for run in run_names:
        df, _ = self.load_predictions(self.timestamp, run_name)

        error_df = df[(df["y_true"] != df["y_pred"])]
        # errors.append(error_df)

        return error_df

