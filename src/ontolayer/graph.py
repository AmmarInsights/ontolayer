"""Build a typed knowledge graph (networkx) from an ontology plus its data sources."""
from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any, Optional

import networkx as nx

from .adapters import get_adapter
from .errors import GraphBuildError
from .schema import Ontology

MAX_WARNINGS = 50
_TRUE = {"true", "t", "yes", "y", "1"}
_FALSE = {"false", "f", "no", "n", "0"}


def coerce_value(value: Any, type_: str) -> tuple[Any, bool]:
    """Convert a raw source value to the declared type. Returns (value, ok)."""
    if value is None:
        return None, True
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None, True
    try:
        if type_ == "string":
            return str(value), True
        if type_ == "integer":
            number = float(value)
            return (int(number), True) if number == int(number) else (None, False)
        if type_ == "float":
            return float(value), True
        if type_ == "boolean":
            if isinstance(value, bool):
                return value, True
            low = str(value).lower()
            if low in _TRUE:
                return True, True
            if low in _FALSE:
                return False, True
            return None, False
        if type_ == "date":
            return date.fromisoformat(str(value)[:10]).isoformat(), True
    except (ValueError, TypeError, OverflowError):
        return None, False
    return None, False


class KnowledgeGraph:
    """Nodes are entities (``Concept:key``); edges are relations. Every node keeps provenance."""

    def __init__(self, ontology: Ontology):
        self.ontology = ontology
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.warnings: list[str] = []
        self._extra_warnings = 0
        self.by_concept: dict[str, list[str]] = {c.name: [] for c in ontology.concepts}
        self._raw: dict[str, dict[str, Any]] = {}
        self._columns: dict[str, set[str]] = {}

    @staticmethod
    def node_id(concept: str, key: str) -> str:
        return f"{concept}:{key}"

    def _warn(self, msg: str) -> None:
        if len(self.warnings) < MAX_WARNINGS:
            self.warnings.append(msg)
        else:
            self._extra_warnings += 1

    # ---- build ---------------------------------------------------------
    @classmethod
    def build(cls, ontology: Ontology, base_dir: str | Path = ".") -> "KnowledgeGraph":
        kg = cls(ontology)
        kg._load_nodes(Path(base_dir))
        kg._link_edges()
        return kg

    def _load_nodes(self, base_dir: Path) -> None:
        for concept in self.ontology.concepts:
            src = concept.source
            if src is None:
                continue
            adapter = get_adapter(src, base_dir)
            cols = set(adapter.columns())
            self._columns[concept.name] = cols
            needed = {src.key} | {p.column or p.name for p in concept.properties}
            missing = needed - cols
            if missing:
                raise GraphBuildError(
                    f"Concept '{concept.name}': columns {sorted(missing)} not found in "
                    f"{src.path}. Available columns: {sorted(cols)}"
                )
            for i, row in enumerate(adapter.rows(), start=1):
                raw_key = row.get(src.key)
                if raw_key is None or str(raw_key).strip() == "":
                    self._warn(f"{concept.name} row {i}: empty key, skipped")
                    continue
                key = str(raw_key).strip()
                nid = self.node_id(concept.name, key)
                if nid in self.graph:
                    self._warn(f"{concept.name} row {i}: duplicate key '{key}', skipped")
                    continue
                props: dict[str, Any] = {}
                for p in concept.properties:
                    value, ok = coerce_value(row.get(p.column or p.name), p.type)
                    if not ok:
                        self._warn(f"{concept.name} '{key}': cannot read {p.name}={row.get(p.column or p.name)!r} as {p.type}")
                    props[p.name] = value
                self.graph.add_node(
                    nid, concept=concept.name, key=key, properties=props,
                    provenance={"source": src.path, "table": src.table, "row": i},
                )
                self.by_concept[concept.name].append(nid)
                self._raw[nid] = row

    def _link_edges(self) -> None:
        for rel in self.ontology.relations:
            if not rel.foreign_key:
                continue
            from_name = self.ontology.get_concept(rel.from_).name
            to_name = self.ontology.get_concept(rel.to).name
            cols = self._columns.get(from_name, set())
            if rel.foreign_key not in cols:
                raise GraphBuildError(
                    f"Relation '{rel.name}': foreign key column '{rel.foreign_key}' not found "
                    f"in source of '{from_name}'. Available columns: {sorted(cols)}"
                )
            via = f"{self.ontology.get_concept(from_name).source.path}:{rel.foreign_key}"  # type: ignore[union-attr]
            for nid in self.by_concept[from_name]:
                raw = self._raw[nid].get(rel.foreign_key)
                if raw is None or str(raw).strip() == "":
                    continue
                target = self.node_id(to_name, str(raw).strip())
                if target not in self.graph:
                    self._warn(f"{nid} -[{rel.name}]-> {target}: target not found (dangling reference)")
                    continue
                self.graph.add_edge(nid, target, key=rel.name, relation=rel.name, provenance={"via": via})

    # ---- introspection ------------------------------------------------
    def stats(self) -> dict[str, Any]:
        by_relation: dict[str, int] = {}
        for _, _, rel in self.graph.edges(data="relation"):
            by_relation[rel] = by_relation.get(rel, 0) + 1
        warnings = list(self.warnings)
        if self._extra_warnings:
            warnings.append(f"... and {self._extra_warnings} more warnings")
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "nodes_by_concept": {c: len(v) for c, v in self.by_concept.items()},
            "edges_by_relation": by_relation,
            "warnings": warnings,
        }

    def export_sqlite(self, path: str | Path) -> None:
        """Write the graph to a SQLite file (nodes + edges tables) for inspection or BI tools."""
        path = Path(path)
        if path.exists():
            path.unlink()
        con = sqlite3.connect(path)
        with con:
            con.execute("CREATE TABLE nodes (id TEXT PRIMARY KEY, concept TEXT, key TEXT, properties TEXT, source TEXT, row INTEGER)")
            con.execute("CREATE TABLE edges (source TEXT, target TEXT, relation TEXT, via TEXT)")
            con.executemany(
                "INSERT INTO nodes VALUES (?,?,?,?,?,?)",
                [(n, d["concept"], d["key"], json.dumps(d["properties"]), d["provenance"]["source"], d["provenance"]["row"])
                 for n, d in self.graph.nodes(data=True)],
            )
            con.executemany(
                "INSERT INTO edges VALUES (?,?,?,?)",
                [(s, t, d["relation"], d["provenance"]["via"]) for s, t, d in self.graph.edges(data=True)],
            )
        con.close()

    def node(self, concept: str, key: str) -> Optional[dict[str, Any]]:
        return self.graph.nodes.get(self.node_id(concept, key))
