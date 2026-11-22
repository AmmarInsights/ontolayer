from . import builtin  # noqa: F401  (registers the built-in adapters)
from .base import ADAPTERS, SourceAdapter, get_adapter, register_adapter

__all__ = ["ADAPTERS", "SourceAdapter", "get_adapter", "register_adapter"]
