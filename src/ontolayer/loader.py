"""Load, dump and export ontologies."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .errors import OntologyError
from .schema import Ontology


def load_ontology(path: str | Path) -> Ontology:
    """Read and validate an ontology YAML file, with readable error messages."""
    path = Path(path)
    if not path.exists():
        raise OntologyError(f"Ontology file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise OntologyError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise OntologyError(f"{path}: top level must be a mapping with 'name' and 'concepts'")
    try:
        return Ontology.model_validate(data)
    except ValidationError as exc:
        lines = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"]) or "(root)"
            msg = err["msg"].removeprefix("Value error, ")
            lines.append(f"  - {loc}: {msg}")
        raise OntologyError(f"{path} is not a valid ontology:\n" + "\n".join(lines)) from exc


def dump_ontology(ontology: Ontology) -> str:
    """Serialise an ontology back to YAML."""
    data = ontology.model_dump(by_alias=True, exclude_defaults=True)
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def to_jsonld(ontology: Ontology) -> dict[str, Any]:
    """Export the schema (not the data) as JSON-LD using RDFS/OWL vocabulary."""
    xsd = {"string": "xsd:string", "integer": "xsd:integer", "float": "xsd:double",
           "boolean": "xsd:boolean", "date": "xsd:date"}
    base = f"urn:ontolayer:{ontology.name}#"
    graph: list[dict[str, Any]] = []
    for c in ontology.concepts:
        graph.append({"@id": f"ex:{c.name}", "@type": "rdfs:Class", "rdfs:label": c.name,
                      "rdfs:comment": c.description})
        for p in c.properties:
            graph.append({"@id": f"ex:{c.name}.{p.name}", "@type": "owl:DatatypeProperty",
                          "rdfs:label": p.name, "rdfs:comment": p.description,
                          "rdfs:domain": {"@id": f"ex:{c.name}"}, "rdfs:range": {"@id": xsd[p.type]}})
    for r in ontology.relations:
        graph.append({"@id": f"ex:{r.name}", "@type": "owl:ObjectProperty", "rdfs:label": r.name,
                      "rdfs:comment": r.description, "rdfs:domain": {"@id": f"ex:{r.from_}"},
                      "rdfs:range": {"@id": f"ex:{r.to}"}})
    return {
        "@context": {"ex": base, "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                     "owl": "http://www.w3.org/2002/07/owl#", "xsd": "http://www.w3.org/2001/XMLSchema#"},
        "@graph": graph,
    }


def json_schema() -> str:
    """JSON Schema for ontology files (handy for editor validation)."""
    return json.dumps(Ontology.model_json_schema(by_alias=True), indent=2)
