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
import src.utils.path_helper as ph
import src.utils.evaluation_helper as eval

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

        """
        [DEBUG] files: ['2026-03-27_14-55-26_LogisticRegression_results.parquet', 
        '2026-03-27_14-55-26_LogisticRegression_coef_1.png', 
        '2026-03-27_14-55-26_LogisticRegression_coef_heat.png', 
        '2026-03-27_14-55-26_LogisticRegression_coefs.csv', 
        '2026-03-27_14-55-26_LGBMClassifier_results.parquet', 
        '2026-03-27_14-55-26_LGBMClassifier_coefs.csv', 
        '2026-03-27_14-55-26_XGBClassifier_results.parquet', 
        '2026-03-27_14-55-26_XGBClassifier_coefs.csv', 
        '2026-03-27_14-55-26_CatBoostClassifier_results.parquet', 
        '2026-03-27_14-55-26_CatBoostClassifier_coefs.csv', 
        '2026-03-27_14-55-26_SimpleBinaryGCN_results.parquet', 
        '2026-03-27_14-55-26_SimpleBinaryGCN_meta.json', 
        '2026-03-27_14-55-26_SimpleBinaryGCN_coefs.csv']
        forecast_stage_1
        """

class EvaluationManager:
    
    def __init__(self, timestamp=None):
        self.base_path = Path(os.getenv("PATH_EVALUATED"))
        self.timestamp = timestamp
        self.folder = Path(f"{self.base_path}/{self.timestamp}") 

        session.now = self.timestamp
        session.save_folder = self.folder

    def list_runs(self):
        
        run_list = [p.name for p in self.base_path.iterdir() if p.is_dir()]

        return run_list 
    
    def load_predictions(self, model_name: str):
        # setup logger
        logger = session.logger
        
        files = list(self.folder.iterdir())
        logger.info("File count in self.folder ('%s'): %s", 
                   ph.shorten_path(self.folder),
                   len([f.name for f in files]))
        
        predicts = [f for f in self.folder.iterdir() if f.suffix == ".parquet"]
        metas = [f for f in self.folder.iterdir() if f.suffix == ".json"]
        
        logger.info("Length of 'predicts' and 'metas':\t%s | %s",
                    len(predicts), 
                    len(metas))

        model_predict = [f for f in predicts if model_name in f.name]
        model_meta = [f for f in metas if model_name in f.name]
      
        if len(model_predict) != 1:
            raise ValueError(f"Expected 1 predict file, found {len(model_predict)}: {model_predict}")

        if len(model_meta) != 1:
            raise ValueError(f"Expected 1 meta file, found {len(model_meta)}: {model_meta}")

        # if len(model_predict) != 1:    # ) or ():
        #     logger.error("Found invalid number (n=%s) of result files for %s @ %s\n%s", 
        #                  len(model_predict), 
        #                  model_name,
        #                  self.timestamp,
        #                  self.folder) 
            
        # if len(model_meta) != 1:
        #     logger.error("Found invalid number (n=%s) of meta data files for %s @ %s\n%s", 
        #                  len(model_meta),
        #                  model_name,
        #                  self.timestamp,
        #                  self.folder) 
                
        # predict_path = f"{self.folder}/{self.timestamp}_{model_predict[0]}_results.parquet"
        # meta_path = f"{self.folder}/{self.timestamp}_{model_meta[0]}_meta.json"
         
        df = pd.read_parquet(model_predict[0])
        
        with open(model_meta[0]) as f:
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
        
        res_dict = eval.find_best_threshold(
                                df["y_true"],
                                df["y_proba"],
                                metric, 
                                beta=beta
                                )
        
        return res_dict


    def describe_run(self, 
                     run_name, 
                     metric_fn: List[Callable] | Callable,
                     class_report_fn: Callable=None):
        # setup logger
        logger = session.logger

        # 
        if isinstance(metric_fn, Callable):
            metric_fn = [metric_fn]

        # for run in run_names:
        df, _ = self.load_predictions(run_name)
            
        score_all = {}
        class_report = {}
        for fn in metric_fn:
            logger.info("Start describing '%s' results by '%s'", 
                        run_name, 
                        fn.__name__)
            
            score = fn(df["y_true"], df["y_pred"])
            # score_df = pd.DataFrame(score)

            score_all.update(score)      # {str(fn.__name__): score}
            
        # results.append({
        #     # "model": run_name,
        #     # "model": meta["model_class"],
        #     # "time": self.timestamp, 
        #     "scores": score_all
        #     })
        
        # pd.DataFrame(results).sort_values("score", ascending=False)
        if class_report_fn:
            class_report = class_report_fn(df["y_true"], df["y_pred"])

        return score_all, class_report
    

    def get_errors(self, run_name: str):
        # setup logger
        logger = session.logger

        # if isinstance(run_names, str):
        #     run_names = [run_names]
        
        # errors = []
        # for run in run_names:
        df, _ = self.load_predictions(run_name)

        logger.info("Start filtering for false predictions in '%s' results", 
                        run_name)
        
        error_df = df[(df["y_true"] != df["y_pred"])]
        # errors.append(error_df)

        return error_df

        