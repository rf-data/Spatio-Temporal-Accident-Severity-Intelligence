## check_class.py
# imports
from pydantic import BaseModel
from typing import Callable, Dict


class CheckMeta(BaseModel):
    name: str
    description: str
    function: Callable
    category: str


class CheckRegistry:
    def __init__(self):
        self._checks: Dict[str, CheckMeta] = {}

    def register(self, name: str, description: str, func: Callable, category: str):
        if name in self._checks:
            raise ValueError(f"Check '{name}' already registered.")
        self._checks[name] = CheckMeta(
            name=name, description=description, function=func, category=category
        )

    def get(self, name: str) -> Callable:
        return self._checks[name].function

    def list(self):
        return self._checks

    def export_for_llm(self):
        return [
            {"name": meta.name, "description": meta.description}
            for meta in self._checks.values()
        ]


checks_registry = CheckRegistry()


def add_check(registry: CheckRegistry, description: str, category: str):
    def decorator(func):
        registry.register(
            name=func.__name__, description=description, category=category, func=func
        )
        return func

    return decorator
