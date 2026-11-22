"""Adapter interface: how OntoLayer reads rows from a data source."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Iterator

from ..errors import GraphBuildError
from ..schema import Source

ADAPTERS: dict[str, type["SourceAdapter"]] = {}


def register_adapter(cls: type["SourceAdapter"]) -> type["SourceAdapter"]:
    """Class decorator that makes an adapter available under ``cls.name``."""
    ADAPTERS[cls.name] = cls
    return cls


class SourceAdapter(ABC):
    """Reads tabular rows (dicts) from one source.

    To add a new data source, subclass this, set ``name`` and decorate with
    :func:`register_adapter`. Then use ``adapter: <name>`` in the ontology YAML.
    """

    name: ClassVar[str]

    def __init__(self, source: Source, base_dir: Path):
        self.source = source
        path = Path(source.path)
        self.path = path if path.is_absolute() else Path(base_dir) / path
        if not self.path.exists():
            raise GraphBuildError(f"Source file not found: {self.path}")

    @abstractmethod
    def columns(self) -> list[str]:
        """Column names available in the source."""

    @abstractmethod
    def rows(self) -> Iterator[dict[str, Any]]:
        """Stream rows as dictionaries keyed by column name."""


def get_adapter(source: Source, base_dir: Path) -> SourceAdapter:
    try:
        cls = ADAPTERS[source.adapter]
    except KeyError:
        raise GraphBuildError(
            f"Unknown adapter '{source.adapter}'. Available adapters: {sorted(ADAPTERS)}"
        ) from None
    return cls(source, Path(base_dir))
