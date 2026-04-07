
# imports 
from pydantic import BaseModel
from typing import Callable, Dict, List, Literal, Any, Optional

from enum import Enum

# -----------------
# Model Registry 
# -----------------

class BaseModelAdapter:
    def __init__(self):
        self.logger = session.logger
        

    def build(self, model_name, model_config): 
        some_dict = {}
        some_build_fn = some_dict[model_name]
        return some_build_fn(**model_config)
    
    def train(self, model, X_train, y_train): 
        
        return model.train(X_train, y_train)
    
    def predict(self, model, X_test): 
        
        return model.predict(X_test)
    
    def predict_proba(self, model, X_test):
        if hasattr(model, "predict_proba"):
            # y_proba = model.predict_proba(X_test)[:, 1]
            return model.predict_proba(X_test)
        
        return None

# class XGBoostAdapter(BaseModelAdapter): ...
# class TorchAdapter(BaseModelAdapter): ...

class TaskType(str, Enum):
    BINARY = "binary"
    MULTI_CLASS = "multi_class"
    MULTI_LABEL = "multi_label"
    REGRESSION = "regression"


class ModelMeta(BaseModel):
    name: str
    framework: str

    # core
    learning_type: Literal["supervised", "unsupervised", "semi_supervised"]
    task_type: List[TaskType]

    # data compatibility
    input_type: List[str]   # tabular, image, text, graph, time_series
    output_type: str        # probability, class_label, embedding

    # adapter
    adapter: BaseModelAdapter
    # train_fn: Callable
    # predict_fn: Callable
    # predict_proba_fn: Optional[Callable] = None

    # capabilities
    tags: List[str] = []

    # requirements
    requires_gpu: bool = False
    supports_sparse: bool = False
    handles_missing: bool = True

    # evaluation
    supported_metrics: List[str] = []

    # config
    default_params: Dict[str, Any] = {}
    search_space: Dict[str, Any] = {}

    # explainability
    explainable: bool = False
    explain_methods: List[str] = []  # shap, lime, grad_cam


class ModelRegistry:
    def __init__(self):
        self._models: Dict[str, ModelMeta] = {}

    def register(self, 
                 name: str,
                framework: str,
                learning_type: str,
                task_type: TaskType | List[TaskType],
                input_type: List[str],
                output_type: str, 
                adapter: BaseModelAdapter, 
                # train_fn: Callable,
                # predict_fn: Callable,
                # predict_proba_fn: Optional[Callable]=None,
                tags: List[str]=[], 
                requires_gpu: bool = False,
                supports_sparse: bool = False,
                handles_missing: bool = True,
                supported_metrics: List[str] = [], 
                default_params: Dict[str, Any] = {},
                search_space: Dict[str, Any] = {},
                explainable: bool = False,
                explain_methods: List[str] = []
                 ):
        if name in self._models:
            print("""
                [INFO] Model '{name}' already registered. 
                  Skip and continue with next model.
                  """)

        self._models[name] = ModelMeta(
                                name=name,
                                framework=framework,
                                learning_type=learning_type,
                                task_type=task_type,
                                input_type=input_type,
                                output_type=output_type,
                                adpater=adapter,
                                tags=tags,
                                requires_gpu=requires_gpu,
                                supports_sparse=supports_sparse,
                                handles_missing=handles_missing,
                                supported_metrics=supported_metrics, 
                                default_params=default_params,
                                search_space=search_space,
                                explainable=explainable,
                                explain_methods=explain_methods
                                    )
        return 


    def list(self, meta_only=True):

        if meta_only:
            return list(self._models.values())
        
        return self._models
    

    def get_fn_by_name(self, name: str, fn_type: Literal["train", "predict", "predict_proba"]="train") -> Callable:
        
        meta = self._models[name]
        
        if fn_type == "train":
            func = meta.train_fn
        elif fn_type == "predict":
            func = meta.predict_fn
        elif fn_type == "predict_proba":
            func = meta.predict_proba_fn
        else: 
            raise ValueError(f"Invalid input for 'fn_type': {fn_type}")
        
        return func


    # def 

    # def get_model_by_keywords(
    #                         self, 
    #                         supervised=None,
    #                         classification=None,       # binary, multi_class, multi_label
    #                         regression=None
    #                         ) -> List[str]:

    #     models = []

    #     for meta in self.list():

    #         if meta.supervised != supervised:
    #             continue

    #         if classification and not meta.classification:
    #             continue

    #         if regression and not meta.regression:
    #             continue

    #         models.append(meta.name)

    #     return models
    

    # def export_for_llm(self):
    #     return [
    #         {"name": meta.name,
    #          "description": meta.description,
    #          "category": meta.category,
    #         "eda": meta.eda,
    #         "default": meta.default, 
    #         "cross_file": meta.cross_file
    #          }
    #         for meta in self._checks.values()
    #     ]

# model_registry = ModelRegistry()

