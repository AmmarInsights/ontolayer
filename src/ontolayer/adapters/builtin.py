"""Built-in adapters: CSV, SQLite and JSON/JSONL."""
from __future__ import annotations

import csv
import json
import re
import sqlite3
from typing import Any, Iterator

from ..errors import GraphBuildError
from .base import SourceAdapter, register_adapter


@register_adapter
class CsvAdapter(SourceAdapter):
    name = "csv"

    def columns(self) -> list[str]:
        with open(self.path, newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f).fieldnames or [])

    def rows(self) -> Iterator[dict[str, Any]]:
        with open(self.path, newline="", encoding="utf-8-sig") as f:
            yield from csv.DictReader(f)


_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@register_adapter
class SqliteAdapter(SourceAdapter):
    name = "sqlite"

    def __init__(self, source, base_dir):
        super().__init__(source, base_dir)
        if not source.table or not _IDENT.match(source.table):
            raise GraphBuildError(f"sqlite source needs a plain table name, got {source.table!r}")

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        return con

    def columns(self) -> list[str]:
        with self._connect() as con:
            cols = [r["name"] for r in con.execute(f'PRAGMA table_info("{self.source.table}")')]
        if not cols:
            raise GraphBuildError(f"Table '{self.source.table}' not found in {self.path}")
        return cols

    def rows(self) -> Iterator[dict[str, Any]]:
        with self._connect() as con:
            for row in con.execute(f'SELECT * FROM "{self.source.table}"'):
                yield dict(row)


@register_adapter
class JsonAdapter(SourceAdapter):
    """Reads a JSON array of objects, an object with a 'records' list, or JSON Lines."""

    name = "json"

    def _load(self) -> list[dict[str, Any]]:
        text = self.path.read_text(encoding="utf-8")
        if self.path.suffix == ".jsonl":
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("records", [])
        if not isinstance(data, list):
            raise GraphBuildError(f"{self.path}: expected a JSON array of objects")
        return data

    def columns(self) -> list[str]:
        cols: dict[str, None] = {}
        for row in self._load():
            cols.update(dict.fromkeys(row))
        return list(cols)

    def rows(self) -> Iterator[dict[str, Any]]:
        yield from self._load()
