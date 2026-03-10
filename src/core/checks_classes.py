## check_class.py
# imports
from pydantic import BaseModel
from typing import Callable, Dict, List


class CheckMeta(BaseModel):
    name: str
    description: str
    function: Callable
    category: str
    eda: bool
    default: bool
    cross_file: bool


class CheckRegistry:
    def __init__(self):
        self._checks: Dict[str, CheckMeta] = {}

    def register(self, 
                 name: str, 
                 description: str, 
                 func: Callable, 
                 category: str,
                 eda: bool,
                default: bool,
                cross_file: bool
                 ):
        if name in self._checks:
            print("""
                [INFO] Check '{name}' already registered. 
                  Skip and continue with next check.
                  """)
        self._checks[name] = CheckMeta(
                                name=name, 
                                description=description, 
                                function=func, 
                                category=category,
                                eda=eda,
                                default=default, 
                                cross_file=cross_file
                                    )

    def get(self, name: str) -> Callable:
        meta = self._checks[name]
        return meta.function

    def get_by_category(self, 
                        category: str,
                        default_only: bool = False,
                        eda_only: bool = False,
                        cross_file: bool = False
                        ) -> List[str]:

        checks = []

        for meta in self.list():

            if meta.category != category:
                continue

            if default_only and not meta.default:
                continue

            if eda_only and not meta.eda:
                continue

            if cross_file and meta.cross_file != cross_file:
                continue

            checks.append(meta.name)

        return checks

    def list(self, meta_only=True):

        if meta_only:
            return list(self._checks.values())
        
        return self._checks
        

    def export_for_llm(self):
        return [
            {"name": meta.name,
             "description": meta.description,
             "category": meta.category,
            "eda": meta.eda,
            "default": meta.default, 
            "cross_file": meta.cross_file
             }
            for meta in self._checks.values()
        ]

checks_registry = CheckRegistry()


def add_check(registry: CheckRegistry, 
              description: str, 
              category: str,
              eda: bool,
              default: bool,
              cross_file: bool):
    def decorator(func):
        registry.register(
            name=func.__name__, 
            description=description, 
            category=category, 
            func=func,
            eda=eda,
            default=default,
            cross_file=cross_file
        )
        return func

    return decorator
