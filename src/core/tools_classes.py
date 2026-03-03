## tools_class.py
# imports
from pydantic import BaseModel
from typing import List, Callable, Dict, Any, Optional


# --------------------
# DATA AUDITOR
# --------------------
class ToolResult(BaseModel):
    tool_name: str
    success: Optional[bool] = None
    message: Optional[str] = None
    data: Optional[Any] = None
    metadata: Optional[Dict] = None


class ToolMeta(BaseModel):
    name: str
    description: str
    category: str
    function: Callable


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolMeta] = {}

    def register(self, name: str, description: str, category: str, func: Callable):
        if name in self._tools:
            raise ValueError(f"Tool '{name}' already registered.")

        self._tools[name] = ToolMeta(
            name=name, description=description, category=category, function=func
        )

    def get(self, name: str) -> Callable:
        return self._tools[name].function

    def list(self) -> Dict[str, ToolMeta]:
        return self._tools

    def export_for_llm(self):
        return [
            {
                "name": meta.name,
                "description": meta.description,
                "category": meta.category,
            }
            for meta in self._tools.values()
        ]


tools_registry = ToolRegistry()


def add_tool(registry: ToolRegistry, description: str, category: str):
    def decorator(func):
        registry.register(
            name=func.__name__, description=description, category=category, func=func
        )
        return func

    return decorator


###################################################################################################################
# -----------------------------------------------------------------------------------------------------------------
###################################################################################################################
# --------------------
# RAW DATA AGENT
# --------------------
class Observation(BaseModel):
    tool_name: str
    category: str
    column: Optional[str]
    description: str
    metrics: Optional[dict]
    recommendation_hint: Optional[dict] = None
