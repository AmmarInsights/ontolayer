"""The semantic layer: business-level queries over the knowledge graph.

Every method returns plain JSON-serialisable dicts so it can be exposed directly as MCP tools.
Invalid requests raise :class:`SemanticError` with a message that lists valid alternatives,
so an agent can correct itself on the next call.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Optional

import networkx as nx

from .errors import SemanticError
from .graph import KnowledgeGraph, coerce_value
from .schema import Concept, Property

OPS = ("eq", "ne", "gt", "gte", "lt", "lte", "in", "contains")
MAX_LIMIT = 200
MAX_DISTINCT_FOR_VALUE_INDEX = 200

STOPWORDS = set(
    """a an the of in on at for to from by per with without and or not is are was were be been do does did
    what which who whom whose how many much total show list give tell me us find get all each every any
    there their its it this that these those than then as into over under between about during last
    first top bottom number count""".split()
)

# Words ignored when matching labels (grammar glue); STOPWORDS additionally ignores question filler
# when computing how much of a question is covered.
FILLER = set(
    """a an the of in on at for to from by per with without and or is are was were be been do does did
    what which who whom whose how it this that these those than then as into over under between about""".split()
)

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_WORD = re.compile(r"[a-z0-9]+")


def _singular(word: str) -> str:
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _words(text: str) -> list[str]:
    """Lower-case, camelCase/snake_case aware, singularised word list."""
    return [_singular(w) for w in _WORD.findall(_CAMEL.sub(" ", text).lower())]


def _meaningful(text: str) -> frozenset[str]:
    return frozenset(w for w in _words(text) if w not in FILLER)


def _norm_value(value: Any) -> Any:
    return value.lower() if isinstance(value, str) else value


def _round(value: Any) -> Any:
    return round(value, 4) if isinstance(value, float) else value


class SemanticLayer:
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self.ont = kg.ontology
        self.g = kg.graph
        self._forward = nx.DiGraph()
        self._forward.add_nodes_from(c.name for c in self.ont.concepts)
        for r in self.ont.relations:
            self._forward.add_edge(self.ont.get_concept(r.from_).name, self.ont.get_concept(r.to).name)
        self._hops: dict[tuple[str, str], list[str]] = {}
        self._values: Optional[dict[str, list[dict[str, Any]]]] = None
        self._catalog_cache: Optional[list[dict[str, Any]]] = None

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _split_ref(self, ref: str, default: Concept) -> tuple[Concept, Property]:
        cname, _, pname = ref.rpartition(".")
        concept = self.ont.get_concept(cname) if cname else default
        prop = concept.prop(pname)
        if prop is None:
            raise SemanticError(
                f"Concept '{concept.name}' has no property '{pname}'. "
                f"Available properties: {[p.name for p in concept.properties]}"
            )
        return concept, prop

    def _hop_path(self, start: str, target: str) -> list[str]:
        if (start, target) not in self._hops:
            if start == target:
                self._hops[(start, target)] = [start]
            else:
                try:
                    self._hops[(start, target)] = nx.shortest_path(self._forward, start, target)
                except nx.NetworkXNoPath:
                    raise SemanticError(
                        f"'{start}' has no to-one relation path to '{target}', so it cannot be used "
                        f"here. Reachable from '{start}': {sorted(nx.descendants(self._forward, start))}"
                    ) from None
        return self._hops[(start, target)]

    def _related(self, nid: str, start: str, target: str) -> Optional[str]:
        cur: Optional[str] = nid
        for nxt in self._hop_path(start, target)[1:]:
            found = None
            for _, t in self.g.out_edges(cur):
                if self.g.nodes[t]["concept"] == nxt:
                    found = t
                    break
            if found is None:
                return None
            cur = found
        return cur

    def _prop_value(self, nid: str, base: str, concept: Concept, prop: Property) -> Any:
        if concept.name == base:
            return self.g.nodes[nid]["properties"].get(prop.name)
        rel = self._related(nid, base, concept.name)
        return None if rel is None else self.g.nodes[rel]["properties"].get(prop.name)

    @staticmethod
    def _typed(prop: Property, value: Any) -> Any:
        typed, ok = coerce_value(value, prop.type)
        if not ok or typed is None:
            raise SemanticError(f"Cannot use {value!r} as a {prop.type} value for property '{prop.name}'")
        return typed

    def _parse_filters(self, base: Concept, filters: Optional[dict[str, Any]]):
        parsed = []
        for ref, cond in (filters or {}).items():
            concept, prop = self._split_ref(ref, base)
            self._hop_path(base.name, concept.name)
            if isinstance(cond, dict):
                if len(cond) != 1:
                    raise SemanticError(f"Filter '{ref}' must have exactly one operator, e.g. {{'gt': 5}}")
                op, val = next(iter(cond.items()))
            else:
                op, val = "eq", cond
            op = str(op).lower()
            if op not in OPS:
                raise SemanticError(f"Unknown filter operator '{op}'. Use one of {list(OPS)}")
            if op == "in":
                if not isinstance(val, list):
                    raise SemanticError(f"Filter '{ref}': 'in' needs a list of values")
                val = [_norm_value(self._typed(prop, v)) for v in val]
            elif op == "contains":
                val = str(val).lower()
            else:
                val = _norm_value(self._typed(prop, val))
            parsed.append((concept, prop, op, val))
        return parsed

    @staticmethod
    def _compare(value: Any, op: str, target: Any) -> bool:
        if value is None:
            return op == "ne"
        v = _norm_value(value)
        try:
            if op == "eq":
                return v == target
            if op == "ne":
                return v != target
            if op == "in":
                return v in target
            if op == "contains":
                return target in str(v).lower()
            if op == "gt":
                return v > target
            if op == "gte":
                return v >= target
            if op == "lt":
                return v < target
            if op == "lte":
                return v <= target
        except TypeError:
            return False
        return False

    def _passes(self, nid: str, base: str, parsed) -> bool:
        return all(self._compare(self._prop_value(nid, base, c, p), op, val) for c, p, op, val in parsed)

    def _view(self, nid: str, full: bool = False) -> dict[str, Any]:
        d = self.g.nodes[nid]
        out = {"concept": d["concept"], "key": d["key"], "properties": d["properties"]}
        if full:
            out["provenance"] = d["provenance"]
        return out

    def _node_for(self, concept: str, key: str) -> tuple[Concept, str]:
        c = self.ont.get_concept(concept)
        nid = self.kg.node_id(c.name, str(key).strip())
        if nid not in self.g:
            examples = [self.g.nodes[n]["key"] for n in self.kg.by_concept[c.name][:5]]
            raise SemanticError(f"No {c.name} with key '{key}'. Example keys: {examples}")
        return c, nid

    def _inverse_label(self, rel_name: str) -> str:
        for r in self.ont.relations:
            if r.name == rel_name:
                return r.inverse or f"inverse_of_{r.name}"
        return f"inverse_of_{rel_name}"

    # ------------------------------------------------------------------
    # schema discovery
    # ------------------------------------------------------------------
    def overview(self) -> dict[str, Any]:
        return {
            "ontology": self.ont.name,
            "version": self.ont.version,
            "description": self.ont.description,
            "concepts": [
                {"name": c.name, "description": c.description, "entities": len(self.kg.by_concept[c.name]),
                 "properties": [p.name for p in c.properties]}
                for c in self.ont.concepts
            ],
            "relations": [
                {"name": r.name, "from": r.from_, "to": r.to, "inverse": r.inverse, "cardinality": r.cardinality}
                for r in self.ont.relations
            ],
            "metrics": [{"name": m.name, "description": m.description} for m in self.ont.metrics],
            "graph": {"nodes": self.g.number_of_nodes(), "edges": self.g.number_of_edges()},
        }

    def describe_concept(self, name: str) -> dict[str, Any]:
        c = self.ont.get_concept(name)
        outgoing = [
            {"relation": r.name, "direction": "outgoing", "to": r.to, "cardinality": r.cardinality,
             "description": r.description}
            for r in self.ont.relations if r.from_.lower() == c.name.lower()
        ]
        incoming = [
            {"relation": r.inverse or f"inverse_of_{r.name}", "direction": "incoming", "from": r.from_,
             "via_relation": r.name, "description": r.description}
            for r in self.ont.relations if r.to.lower() == c.name.lower()
        ]
        return {
            "name": c.name,
            "description": c.description,
            "synonyms": c.synonyms,
            "entities": len(self.kg.by_concept[c.name]),
            "properties": [
                {"name": p.name, "type": p.type, "description": p.description, "synonyms": p.synonyms}
                for p in c.properties
            ],
            "relations": outgoing + incoming,
            "metrics": [{"name": m.name, "description": m.description}
                        for m in self.ont.metrics if m.concept.lower() == c.name.lower()],
            "samples": [self._view(n) for n in self.kg.by_concept[c.name][:2]],
        }

    def list_metrics(self) -> dict[str, Any]:
        return {"metrics": [self._metric_info(m) for m in self.ont.metrics]}

    def _metric_info(self, m) -> dict[str, Any]:
        target = f"{m.concept}.{m.property}" if m.property else m.concept
        text = f"{m.aggregation} of {target}"
        if m.filters:
            text += " where " + ", ".join(f"{k} = {v}" if not isinstance(v, dict) else f"{k} {v}" for k, v in m.filters.items())
        return {"name": m.name, "description": m.description, "synonyms": m.synonyms, "concept": m.concept,
                "aggregation": m.aggregation, "property": m.property, "filters": m.filters,
                "unit": m.unit, "definition": text}

    def explain_metric(self, name: str) -> dict[str, Any]:
        m = self.ont.get_metric(name)
        info = self._metric_info(m)
        info["can_group_by"] = [
            f"{self.ont.get_concept(t).name}.{p.name}"
            for t in sorted(nx.descendants(self._forward, self.ont.get_concept(m.concept).name) | {self.ont.get_concept(m.concept).name})
            for p in self.ont.get_concept(t).properties
        ]
        return info

    # ------------------------------------------------------------------
    # search and answerability
    # ------------------------------------------------------------------
    def _catalog(self) -> list[dict[str, Any]]:
        if self._catalog_cache is None:
            items: list[dict[str, Any]] = []
            for c in self.ont.concepts:
                labels = [c.name, *c.synonyms]
                items.append({"kind": "concept", "name": c.name, "concept": c.name, "description": c.description, "labels": labels})
                for p in c.properties:
                    items.append({"kind": "property", "name": p.name, "concept": c.name, "description": p.description,
                                  "labels": [p.name, *p.synonyms]})
            for m in self.ont.metrics:
                items.append({"kind": "metric", "name": m.name, "concept": m.concept, "description": m.description,
                              "labels": [m.name, *m.synonyms]})
            for r in self.ont.relations:
                items.append({"kind": "relation", "name": r.name, "concept": r.from_, "description": r.description,
                              "labels": [r.name, *([r.inverse] if r.inverse else [])]})
            for it in items:
                it["label_sets"] = [s for s in (_meaningful(lbl) for lbl in it["labels"]) if s]
                it["desc_tokens"] = _meaningful(it["description"])
            self._catalog_cache = items
        return self._catalog_cache

    def search_concepts(self, query: str, limit: int = 8) -> dict[str, Any]:
        q = _meaningful(query)
        results = []
        for it in self._catalog():
            label_tokens = set().union(*it["label_sets"]) if it["label_sets"] else set()
            score = 5 * len(q & label_tokens) + len(q & it["desc_tokens"])
            if q and any(q == s for s in it["label_sets"]):
                score += 3
            if score > 0:
                results.append({"kind": it["kind"], "name": it["name"], "concept": it["concept"],
                                "description": it["description"], "score": score})
        results.sort(key=lambda r: (-r["score"], r["name"]))
        return {"query": query, "results": results[: max(1, min(limit, 25))]}

    def _value_index(self) -> dict[str, list[dict[str, Any]]]:
        if self._values is None:
            index: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for c in self.ont.concepts:
                for p in c.properties:
                    if p.type != "string":
                        continue
                    counts: dict[str, int] = defaultdict(int)
                    for nid in self.kg.by_concept[c.name]:
                        v = self.g.nodes[nid]["properties"].get(p.name)
                        if v:
                            counts[v] += 1
                    if len(counts) > MAX_DISTINCT_FOR_VALUE_INDEX:
                        continue
                    for v, n in counts.items():
                        norm = " ".join(_words(v))
                        if len(norm) < 2 or norm.isdigit() or norm in STOPWORDS:
                            continue
                        index[norm].append({"concept": c.name, "property": p.name, "value": v, "count": n})
            self._values = dict(index)
        return self._values

    def resolve_value(self, text: str, limit: int = 10) -> dict[str, Any]:
        norm = " ".join(_words(text))
        index = self._value_index()
        hits = list(index.get(norm, []))
        if not hits:
            for key, entries in index.items():
                if norm and (norm in key or key in norm):
                    hits.extend(entries)
        return {"text": text, "matches": hits[: max(1, min(limit, 50))]}

    def check_answerable(self, question: str) -> dict[str, Any]:
        all_words = _words(question)
        q_tokens = [w for w in all_words if w not in STOPWORDS]
        q_set = {w for w in all_words if w not in FILLER}
        covered: set[str] = set()
        matched: dict[str, list[dict[str, Any]]] = {"concepts": [], "properties": [], "metrics": [], "relations": [], "values": []}
        kind_key = {"concept": "concepts", "property": "properties", "metric": "metrics", "relation": "relations"}

        hits: dict[str, list[tuple[dict[str, Any], frozenset[str]]]] = defaultdict(list)
        for it in self._catalog():
            best = None
            for s in it["label_sets"]:
                if s <= q_set and (best is None or len(s) > len(best)):
                    best = s
            if best is not None:
                hits[it["kind"]].append((it, best))
        for kind, found in hits.items():
            for it, toks in found:
                if any(toks < other for _, other in found):  # a longer phrase also matched
                    continue
                covered |= toks
                entry = {"name": it["name"]}
                if kind == "property":
                    entry["concept"] = it["concept"]
                matched[kind_key[kind]].append(entry)

        padded = f" {' '.join(all_words)} "
        for norm, entries in self._value_index().items():
            if f" {norm} " in padded:
                covered |= set(norm.split()) - STOPWORDS
                matched["values"].extend(entries[:3])

        unmatched = [w for w in dict.fromkeys(q_tokens) if w not in covered]
        coverage = round(1 - len(unmatched) / len(set(q_tokens)), 2) if q_tokens else 0.0
        anchored = bool(matched["concepts"] or matched["metrics"] or matched["values"] or matched["properties"])
        answerable = anchored and coverage >= 0.5
        result = {"question": question, "answerable": answerable, "coverage": coverage,
                  "matched": matched, "unmatched_terms": unmatched}
        if not answerable:
            result["suggestion"] = ("The ontology does not cover these terms: "
                                    f"{unmatched}. Use search_concepts to look for related terms, or ask the user to clarify.")
        elif unmatched:
            result["suggestion"] = f"Mostly covered. Unrecognised terms (may be filler or missing data): {unmatched}."
        return result

    # ------------------------------------------------------------------
    # data queries
    # ------------------------------------------------------------------
    def find_entities(self, concept: str, filters: Optional[dict[str, Any]] = None, limit: int = 20,
                      order_by: Optional[str] = None, descending: bool = False) -> dict[str, Any]:
        c = self.ont.get_concept(concept)
        parsed = self._parse_filters(c, filters)
        nodes = [n for n in self.kg.by_concept[c.name] if self._passes(n, c.name, parsed)]
        if order_by:
            _, prop = self._split_ref(order_by, c)
            nodes.sort(key=lambda n: (self.g.nodes[n]["properties"].get(prop.name) is None,
                                      self.g.nodes[n]["properties"].get(prop.name)), reverse=False)
            if descending:
                present = [n for n in nodes if self.g.nodes[n]["properties"].get(prop.name) is not None]
                nodes = present[::-1] + [n for n in nodes if n not in set(present)]
        limit = max(1, min(limit, MAX_LIMIT))
        return {"concept": c.name, "filters": filters or {}, "total_matches": len(nodes),
                "returned": min(limit, len(nodes)), "entities": [self._view(n) for n in nodes[:limit]]}

    def get_entity(self, concept: str, key: str) -> dict[str, Any]:
        c, nid = self._node_for(concept, key)
        out = self._view(nid, full=True)
        out["neighbors"] = self.get_neighbors(c.name, key, limit=10)["neighbors"]
        return out

    def get_neighbors(self, concept: str, key: str, relation: Optional[str] = None,
                      direction: str = "both", limit: int = 50) -> dict[str, Any]:
        c, nid = self._node_for(concept, key)
        if direction not in ("out", "in", "both"):
            raise SemanticError("direction must be 'out', 'in' or 'both'")
        want = relation.lower() if relation else None
        found: list[dict[str, Any]] = []
        if direction in ("out", "both"):
            for _, t, rel in self.g.out_edges(nid, keys=True):
                if want is None or rel.lower() == want:
                    found.append({"relation": rel, "direction": "out", **self._view(t)})
        if direction in ("in", "both"):
            for s, _, rel in self.g.in_edges(nid, keys=True):
                label = self._inverse_label(rel)
                if want is None or label.lower() == want or rel.lower() == want:
                    found.append({"relation": label, "direction": "in", **self._view(s)})
        limit = max(1, min(limit, MAX_LIMIT))
        return {"entity": {"concept": c.name, "key": str(key)}, "total": len(found), "neighbors": found[:limit]}

    def find_path(self, from_concept: str, from_key: str, to_concept: str, to_key: str,
                  max_hops: int = 6) -> dict[str, Any]:
        _, a = self._node_for(from_concept, from_key)
        _, b = self._node_for(to_concept, to_key)
        undirected = self.g.to_undirected(as_view=True)
        try:
            path = nx.shortest_path(undirected, a, b)
        except nx.NetworkXNoPath:
            return {"connected": False, "steps": []}
        if len(path) - 1 > max_hops:
            return {"connected": False, "steps": [], "note": f"shortest path is longer than max_hops={max_hops}"}
        steps = []
        for u, v in zip(path, path[1:]):
            if self.g.has_edge(u, v):
                rel, direction = next(iter(self.g[u][v])), "out"
            else:
                rel, direction = self._inverse_label(next(iter(self.g[v][u]))), "in"
            steps.append({"from": self._view(u), "relation": rel, "direction": direction, "to": self._view(v)})
        return {"connected": True, "hops": len(steps), "steps": steps}

    def compute_metric(self, name: str, group_by: Optional[str] = None,
                       filters: Optional[dict[str, Any]] = None, limit: int = 50) -> dict[str, Any]:
        m = self.ont.get_metric(name)
        base = self.ont.get_concept(m.concept)
        prop = base.prop(m.property) if m.property else None
        parsed = self._parse_filters(base, {**m.filters, **(filters or {})})
        group = self._split_ref(group_by, base) if group_by else None
        if group:
            self._hop_path(base.name, group[0].name)

        buckets: dict[Any, list[Any]] = defaultdict(list)
        rows = 0
        for nid in self.kg.by_concept[base.name]:
            if not self._passes(nid, base.name, parsed):
                continue
            rows += 1
            label = self._prop_value(nid, base.name, group[0], group[1]) if group else None
            value = self.g.nodes[nid]["properties"].get(prop.name) if prop else 1
            buckets[label].append(value)

        def aggregate(values: list[Any]) -> Any:
            if m.aggregation == "count":
                return sum(1 for v in values if v is not None)
            present = [v for v in values if v is not None]
            if not present:
                return None
            if m.aggregation == "sum":
                return _round(sum(present))
            if m.aggregation == "avg":
                return _round(sum(present) / len(present))
            return min(present) if m.aggregation == "min" else max(present)

        out: dict[str, Any] = {"metric": m.name, "definition": self._metric_info(m)["definition"], "unit": m.unit,
                               "group_by": group_by, "extra_filters": filters or {}, "rows_considered": rows}
        if group is None:
            out["value"] = aggregate(buckets.get(None, []))
        else:
            groups = [{"group": "(none)" if k is None else k, "value": aggregate(v), "rows": len(v)} for k, v in buckets.items()]
            groups.sort(key=lambda g: (g["value"] is None, -(g["value"] or 0) if isinstance(g["value"], (int, float)) else 0, str(g["group"])))
            out["groups"] = groups[: max(1, min(limit, MAX_LIMIT))]
        return out
