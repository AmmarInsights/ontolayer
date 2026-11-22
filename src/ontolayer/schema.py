"""Pydantic models that describe a business ontology.

An ontology is made of:

* **concepts**   - business entities (Customer, Order) with typed properties
* **relations**  - typed, directed links between concepts (Order placed_by Customer)
* **metrics**    - certified, named calculations (Revenue = sum of Order.total where completed)
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import SemanticError

PropertyType = Literal["string", "integer", "float", "boolean", "date"]
Aggregation = Literal["count", "sum", "avg", "min", "max"]
Cardinality = Literal["one_to_one", "many_to_one", "one_to_many", "many_to_many"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Property(_Model):
    name: str
    type: PropertyType = "string"
    description: str = ""
    column: Optional[str] = Field(default=None, description="Source column; defaults to the property name.")
    synonyms: list[str] = Field(default_factory=list)


class Source(_Model):
    adapter: str = Field(description="Adapter name: csv, sqlite or json.")
    path: str = Field(description="File path, relative to the ontology file.")
    table: Optional[str] = Field(default=None, description="Table name (sqlite adapter).")
    key: str = Field(description="Source column that uniquely identifies each entity.")


class Concept(_Model):
    name: str
    description: str = ""
    synonyms: list[str] = Field(default_factory=list)
    source: Optional[Source] = None
    properties: list[Property] = Field(default_factory=list)

    def prop(self, name: str) -> Optional[Property]:
        low = name.lower()
        return next((p for p in self.properties if p.name.lower() == low), None)


class Relation(_Model):
    name: str
    from_: str = Field(alias="from")
    to: str
    inverse: Optional[str] = Field(default=None, description="Name when read in the opposite direction.")
    description: str = ""
    cardinality: Cardinality = "many_to_one"
    foreign_key: Optional[str] = Field(
        default=None,
        description="Column in the 'from' concept's source that holds the key of the 'to' concept.",
    )


class Metric(_Model):
    name: str
    description: str = ""
    synonyms: list[str] = Field(default_factory=list)
    concept: str
    aggregation: Aggregation
    property: Optional[str] = None
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Property filters. Keys are 'prop' or 'Concept.prop'; values are scalars or {op: value}.",
    )
    unit: str = ""


def _duplicates(items: list[str]) -> set[str]:
    seen: set[str] = set()
    out: set[str] = set()
    for item in items:
        low = item.lower()
        if low in seen:
            out.add(low)
        seen.add(low)
    return out


class Ontology(_Model):
    name: str
    version: str = "0.1"
    description: str = ""
    concepts: list[Concept]
    relations: list[Relation] = Field(default_factory=list)
    metrics: list[Metric] = Field(default_factory=list)

    # ---- lookups -------------------------------------------------------
    def get_concept(self, name: str) -> Concept:
        low = name.lower()
        for c in self.concepts:
            if c.name.lower() == low:
                return c
        raise SemanticError(f"Unknown concept '{name}'. Available concepts: {[c.name for c in self.concepts]}")

    def get_metric(self, name: str) -> Metric:
        low = name.lower()
        for m in self.metrics:
            if m.name.lower() == low:
                return m
        raise SemanticError(f"Unknown metric '{name}'. Available metrics: {[m.name for m in self.metrics]}")

    # ---- validation ----------------------------------------------------
    @model_validator(mode="after")
    def _check_consistency(self) -> "Ontology":
        errors: list[str] = []
        by_name = {c.name.lower(): c for c in self.concepts}

        for d in _duplicates([c.name for c in self.concepts]):
            errors.append(f"duplicate concept name '{d}'")
        for d in _duplicates([m.name for m in self.metrics]):
            errors.append(f"duplicate metric name '{d}'")
        for c in self.concepts:
            for d in _duplicates([p.name for p in c.properties]):
                errors.append(f"concept '{c.name}': duplicate property '{d}'")

        # Relation names are edge keys, so they are unique globally.
        for d in _duplicates([r.name for r in self.relations]):
            errors.append(f"duplicate relation name '{d}'")
        inverses: dict[str, list[str]] = {}
        for r in self.relations:
            for end in (r.from_, r.to):
                if end.lower() not in by_name:
                    errors.append(f"relation '{r.name}': unknown concept '{end}'")
            if r.foreign_key and r.from_.lower() in by_name and by_name[r.from_.lower()].source is None:
                errors.append(f"relation '{r.name}': foreign_key needs a source on concept '{r.from_}'")
            if r.inverse:
                inverses.setdefault(r.to.lower(), []).append(r.inverse)
        # Inverse labels only need to be unique among the relation labels visible on one concept.
        for target, names in inverses.items():
            outgoing = [r.name for r in self.relations if r.from_.lower() == target]
            for d in _duplicates(names + outgoing):
                errors.append(f"concept '{target}': duplicate relation label '{d}'")

        for m in self.metrics:
            c = by_name.get(m.concept.lower())
            if c is None:
                errors.append(f"metric '{m.name}': unknown concept '{m.concept}'")
                continue
            if m.aggregation != "count" and not m.property:
                errors.append(f"metric '{m.name}': aggregation '{m.aggregation}' needs a property")
            if m.property:
                p = c.prop(m.property)
                if p is None:
                    errors.append(f"metric '{m.name}': concept '{c.name}' has no property '{m.property}'")
                elif m.aggregation in ("sum", "avg") and p.type not in ("integer", "float"):
                    errors.append(f"metric '{m.name}': cannot {m.aggregation} a {p.type} property")
            for key in m.filters:
                cname, _, pname = key.rpartition(".")
                target = by_name.get(cname.lower()) if cname else c
                if target is None:
                    errors.append(f"metric '{m.name}': filter '{key}' names unknown concept '{cname}'")
                elif target.prop(pname) is None:
                    errors.append(f"metric '{m.name}': filter '{key}': no property '{pname}' on '{target.name}'")

        if errors:
            raise ValueError("; ".join(errors))
        return self
