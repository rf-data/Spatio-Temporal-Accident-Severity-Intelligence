# src/core/mlflow_logger.py
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List
import os
import logging
import sys
from datetime import datetime
import numpy as np
import pandas as pd
# import json
# import time
# import mlflow

# from utils.experiment_logger_impl import ExperimentLogger
import src.utils.general_helper as gh
# import src.core.ml_manager as log
from src.core.session import session
import src.utils.file_helper as fh
import src.utils.df_helper as dfh
import src.utils.path_helper as ph



def has_file_handler(logger, log_path):
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            if Path(h.baseFilename) == Path(log_path):
                return True
    return False


def log_header(logger, title: str, level="info"):
    """
    Aufruf: log_header(self.logger, "START ESCALATION CHECK")
    """
    header = (
        "\n"
        + "=" * 50
        + "\n"
        + f"--- {title} --- {datetime.now():%Y-%m-%d %H:%M:%S} ---\n"
        + "=" * 50
    )

    getattr(logger, level)(header)


def log_section(logger, title):
    """
    Header als Ereignis
    """
    logger.info("")
    logger.info("=" * 50)
    logger.info(
        "--- %s --- %s ---", title, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    logger.info("=" * 50 + "\n")

    # else:
    #     print("\n")
    #     print("=" * 50 + "\n")
    #     print(f"--- {title} --- {datetime.now():%Y-%m-%d %H:%M:%S} ---\n")
    #     print("=" * 50 + "\n")


def create_logger(
    name: str, file_name: str, folder: str | Path | None = None, level: str = "info"
) -> logging.Logger:
    """
    Create a configured logger with stdout + optional file logging.

    Parameters
    ----------
    name : str
        Logger name (e.g. "api", "health", "traffic")
    file_name : str | None
        Log file name (without .log). If None, no file logging.
    folder : str | Path | None
        Folder to save log files. If None, uses LOGS env var or "logs".
    level : str
        Log level ("debug", "info", "warning", "error", "critical")
    """

    level_dict = {
        "not_set": logging.NOTSET,
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        # "exception": logging.exception,
        "critical": logging.CRITICAL,
    }

    log_level = level_dict.get(level.lower(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False  # VERY important with Uvicorn / Streamlit

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # --- stdout handler (always) ---
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # --- file handler (optional) ---
    if file_name:
        if not folder:
            gh.load_env_vars()
            folder = os.getenv("LOGS", "logs")

        log_path = ph.ensure_dir(folder)
        log_file = log_path / f"{file_name}.log"

        # check if file handler already exists
        if not has_file_handler(logger, log_file):
            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


# --------------
# MLflow Experiment Logger
# --------------

@dataclass
class ExperimentLogger:
    experiment_name: str
    artifact_location: Path | None = None  # field(default_factory=Path)
    backup_dir: Path = Path(f"{ph.find_project_root()}/mlflow/backups")

    tags: Dict[str, object] = field(default_factory=dict)
    params: Dict[str, object] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    texts: Dict[str, str] = field(default_factory=dict)
    # _model: Dict[str, object] = field(default_factory=dict)
    # dicts: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        root = ph.find_project_root()
        project_name = os.getenv("PROJECT_NAME", "default_project")
        folder = f"{root}/mlflow/logs"

        self.logger = log.create_logger(
            f"mlflow_{project_name}", "mlflow", folder=folder
        )

    def load_latest_backup(self, folder: Path | None = None):
        folder = Path(folder or self.backup_dir / self.experiment_name)

        files = list(folder.glob("*_params.json"))
        if not files:
            raise FileNotFoundError(f"No backups found in {folder}")

        latest = max(files, key=lambda p: p.stem.split("_")[0])
        timestamp = latest.stem.replace("_params", "")

        self.load_logger(folder, timestamp)

    def load_logger(self, folder: Path, timestamp: str):
        """
        Load logger state (tags, params, metrics, texts) from a local backup.

        Parameters
        ----------
        folder : Path
            Backup folder (e.g. mlflow/backups/<experiment_name>)
        timestamp : str
            Timestamp prefix used in backup filenames
        """
        self.logger.info("Start loading logger: {folder} | {timestamp}")

        folder = Path(folder)
        missing = False
        attributes = ["params", "metrics", "texts", "tags"]

        for attr in attributes:
            path = folder / f"{timestamp}_{attr}.json"

            if path.exists():
                setattr(self, attr, fh.load_dict(path))

            else:
                self.logger.warning(f"No {attr} backup found at {path}")
                missing = True

        if missing:
            self.logger.warning(f"Not all backups found in {folder}")
            raise FileNotFoundError()

        self.logger.info(
            f"ExperimentLogger state restored from backup " f"(timestamp={timestamp})"
        )
        # if metrics_path.exists():
        #     self.metrics = fh.load_dict(metrics_path)
        # else:
        #     self.logger.warning(f"No metrics backup found at {metrics_path}")
        #     missing = True

        # if texts_path.exists():
        #     self.texts = fh.load_dict(texts_path)
        # else:
        #     self.logger.warning(f"No texts backup found at {texts_path}")
        #     missing = True

    def local_backup(self, folder=None):
        """Saves all logged data to local files for backup purposes."""
        if not folder:
            folder = Path(f"{self.backup_dir}/{self.experiment_name}")
            # gh.ensure_dir(folder)
        # backup_dir.mkdir(parents=True, exist_ok=True)

        now = session.now

        # Save tags
        tags_path = folder / f"{now}_tags.json"
        fh.save_dict(tags_path, self.tags)

        # Save params
        params_path = folder / f"{now}_params.json"
        fh.save_dict(params_path, self.params)

        # Save metrics
        metrics_path = folder / f"{now}_metrics.json"
        fh.save_dict(metrics_path, self.metrics)

        # Save texts
        texts_path = folder / f"{now}_texts.json"
        fh.save_dict(texts_path, self.texts)

        self.logger.info(f"Local backup saved to {folder}")

    def log_artifact(self, path: str):
        self.artifacts.append(path)

    # def log_dict(self, key: str, value: str):
    #     self.dicts[key] = value

    def log_metric(self, key: str, value: float):
        self.metrics[key] = value

    def log_model(self, model_name, model, exemple_df):
        mlflow.sklearn.log_model(
            sk_model=model,
            name=model_name,
            # artifact_path=path,
            input_example=exemple_df.iloc[:5],
        )

    def log_param(self, key: str, value: object):
        self.params[key] = value

    def log_text(self, name: str, text: str):
        self.texts[name] = text

    def set_tag(self, key: str, value: str):
        self.tags[key] = value

    def setup_experiment(self):
        gh.load_env_vars()
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
        mlflow.set_tracking_uri(mlflow_uri)

        exp = mlflow.get_experiment_by_name(self.experiment_name)

        if exp is None:
            mlflow.create_experiment(
                name=self.experiment_name, artifact_location=str(self.artifact_location)
            )

        self.logger.info("Setting up MLflow experiment: %s", self.experiment_name)

        mlflow.set_experiment(self.experiment_name)

    # def setup_experiment(self):
    #     if self.artifact_location:
    #         mlflow.create_experiment(
    #             name=self.experiment_name,
    #             artifact_location=str(self.artifact_location)
    #         )
    #     mlflow.set_experiment(self.experiment_name)

    # def set_artifact_location(self, folder=None):
    #     if not folder:
    #         self.artifact_location = Path("home/robfra/0_Portfolio_Projekte/LLM/mlflow/artifacts")
    #     else:
    #         self.artifact_location = Path(folder)

    # def set_experiment(self, name=None): # , name_experiment):
    #     self.experiment_name = name

    def flush(self, run_name: str):
        self.logger.info(
            f"Starting MLflow run: {run_name} " f"(experiment={self.experiment_name})"
        )

        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id

            self.logger.info(f"MLflow run_id={run_id}")

            for k, v in self.params.items():
                mlflow.log_param(k, v)
                self.logger.info(f"[PARAM] {k}={v}")

            for k, v in self.metrics.items():
                mlflow.log_metric(k, v)
                self.logger.info(f"[METRIC] {k}={v}")

            for path in self.artifacts:
                mlflow.log_artifact(path)
                self.logger.info(f"[ARTIFACT] {path}")

            for name, text in self.texts.items():
                mlflow.log_text(text, name)
                self.logger.info(f"[TEXT] {name} (len={len(text)})")

            for k, v in self.tags.items():
                mlflow.set_tag(k, v)
                self.logger.info(f"[TAG] {k}={v}")

            self.logger.info(
                "MLflow run finished successfully: %s (%s)", run_name, run_id
            )


# ---------------
# Main MLflow logging function
# ---------------

_experiment_logger: ExperimentLogger | None = None


def get_experiment_logger(
    experiment_name: str | None = None,
    artifact_location: Path | None = None,
) -> ExperimentLogger:
    global _experiment_logger

    if _experiment_logger is None:
        if experiment_name is None:
            raise RuntimeError(
                "ExperimentLogger not initialized yet. "
                "Call get_experiment_logger(experiment_name=...) once."
            )

        _experiment_logger = ExperimentLogger(
            experiment_name=experiment_name,
            artifact_location=artifact_location,
        )

    return _experiment_logger



class ModelLogger:
    
    def __init__(self, base_path):  # , run_name
        self.folder = Path(base_path)
        # self.run_name = run_name # or f"run_{int(time.time())}"
        # self.run_path = self.base_path / self.run_name
        
        self.folder.mkdir(parents=True, exist_ok=True)

    # =========================
    # PUBLIC API
    # =========================

    def log_run(
        self,
        model,
        X_test,
        y_test,
        y_pred=None,
        y_proba=None,
        y_val_pred=None,
        y_val_proba=None,
        metrics=None,
        add_idx: str | List[str]=None,
        extra_params=None
    ):
        framework = self._detect_framework(model)
        
        model_class = model.__class__.__name__
        session.model_class = model_class

        # Calculate predictions
        if y_pred is None:
            y_pred = model.predict(X_test)
            
        if y_proba is None and hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        
        # ===== Save predictions =====
        df_pred = self._build_prediction_df(
                                        X_test, 
                                        y_test, 
                                        y_pred, 
                                        y_proba,
                                        y_val_pred,
                                        y_val_proba, 
                                        framework
                                        )
        
        if add_idx:
            if isinstance(add_idx, str):
                add_idx = [add_idx]
            
            for idx in add_idx:
                df_pred[idx] = X_test[idx]

        timestamp = session.now
        f_name = f"{timestamp}_{model_class}_results"
        dfh.save_df_to_parquet(
                            df_pred,
                            f_name=f_name,
                            folder=self.folder,
                            chunked=True
                            )

        # ===== Metadata =====
        metadata = {
            "model_class": model_class,
            "framework": framework,
            "params": self._extract_params(model),
            "metrics": metrics or {},
            "n_samples": len(y_test),
            "features": list(X_test.columns) if hasattr(X_test, "columns") else None,
            "timestamp": timestamp
            }

        if extra_params:
            metadata["extra"] = extra_params
        
        meta_path = f"{self.folder}/{timestamp}_{model_class}_meta.json"
        fh.save_dict(metadata, meta_path)
        print("saved meta_data")
        
        return  # self.run_path

    # =========================
    # INTERNAL
    # =========================

    def _build_prediction_df(self, 
                             X_test, 
                             y_test, 
                             y_pred, 
                             y_proba,
                             y_val_pred,
                            y_val_proba,
                            framework
                            ):
        
        df = pd.DataFrame({
                    "y_true": y_test,
                    "y_pred": y_pred
                    })
        
        if y_proba is not None:
            df["y_proba"] = y_proba
        
        if y_val_pred is not None:
            df["y_val_pred"] = y_val_pred
        
        if y_val_proba is not None:
            df["y_val_proba"] = y_val_proba

        # save index 
        if hasattr(X_test, "index") and (framework != "pytorch"):
            df["index"] = X_test.index
        
        return df
    

    def _detect_framework(self, model):
        
        module = model.__class__.__module__
        
        if "sklearn" in module:
            return "sklearn"
        elif "xgboost" in module:
            return "xgboost"
        elif "lightgbm" in module:
            return "lightgbm"
        elif "catboost" in module:
            return "catboost"
        elif "torch" in module:
            return "pytorch"
        elif "keras" in module:
            return "keras"
        elif "tensorflow" in module:
            return "tensorflow"
        else:
            return f"unspecified - ({module})"
        
    
    def _extract_params(self, model):
        
        # sklearn / xgboost / lightgbm
        if hasattr(model, "get_params"):
            try:
                return model.get_params()
            except:
                pass
        
        # catboost
        if hasattr(model, "get_all_params"):
            try:
                return model.get_all_params()
            except:
                pass
        
        # pytorch fallback
        return {
            "model_class": model.__class__.__name__,
            "note": "manual parameter logging required"
            }
