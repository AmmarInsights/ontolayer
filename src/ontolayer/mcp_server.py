"""MCP server that exposes the semantic layer to any MCP-compatible agent.

Run it with::

    python -m ontolayer.mcp_server --ontology examples/retail/ontology.yaml            # stdio
    python -m ontolayer.mcp_server --ontology ... --transport streamable-http --port 8000

All tools are read-only: agents can look things up but never change your data.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional

import functools

try:  # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as _Server
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server
    from mcp.server.fastmcp.exceptions import ToolError

from .errors import OntoLayerError
from .graph import KnowledgeGraph
from .loader import load_ontology
from .semantic import SemanticLayer

INSTRUCTIONS = """\
Semantic layer over a business knowledge graph. Use it to ground answers in defined business concepts,
relationships and certified metrics instead of guessing.

Recommended workflow:
1. check_answerable(question) - does the ontology cover the question? It also reports matched concepts, metrics and data values.
2. describe_concept / explain_metric - learn exact property names and relations before querying.
3. compute_metric (aggregates, optional group_by like 'Customer.country') or find_entities (records).
4. get_neighbors / find_path to follow relationships between entities.
Never invent numbers: quote tool results. If check_answerable says data is missing, tell the user.
"""


def _readable_errors(fn):
    """Re-raise our errors as ToolError so the agent sees the helpful message (the SDK hides other exceptions)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except OntoLayerError as exc:
            raise ToolError(str(exc)) from exc

    return wrapper


def build_semantic_layer(ontology_path: "str | Path") -> SemanticLayer:
    path = Path(ontology_path)
    ontology = load_ontology(path)
    return SemanticLayer(KnowledgeGraph.build(ontology, base_dir=path.parent))


def build_server(ontology_path: "str | Path"):
    layer = build_semantic_layer(ontology_path)
    server = _Server(f"ontolayer-{layer.ont.name}", instructions=INSTRUCTIONS)

    def tool():
        return lambda fn: server.tool()(_readable_errors(fn))

    @tool()
    def get_ontology_overview() -> dict[str, Any]:
        """Overview of the whole ontology: concepts, relations, metrics and graph size. Call this first."""
        return layer.overview()

    @tool()
    def search_concepts(query: str, limit: int = 8) -> dict[str, Any]:
        """Find concepts, properties, metrics or relations related to some words (supports synonyms)."""
        return layer.search_concepts(query, limit)

    @tool()
    def check_answerable(question: str) -> dict[str, Any]:
        """Check whether a natural-language question can be answered from this ontology.
        Returns coverage, matched concepts/properties/metrics/data values and the terms that are not covered."""
        return layer.check_answerable(question)

    @tool()
    def describe_concept(name: str) -> dict[str, Any]:
        """Describe one concept: properties with types, relations (both directions), metrics and sample entities."""
        return layer.describe_concept(name)

    @tool()
    def list_metrics() -> dict[str, Any]:
        """List all certified business metrics and their definitions."""
        return layer.list_metrics()

    @tool()
    def explain_metric(name: str) -> dict[str, Any]:
        """Explain how a metric is defined and which Concept.property values it can be grouped by."""
        return layer.explain_metric(name)

    @tool()
    def compute_metric(name: str, group_by: Optional[str] = None, filters: Optional[dict[str, Any]] = None,
                       limit: int = 50) -> dict[str, Any]:
        """Compute a certified metric. group_by is 'Concept.property' (e.g. 'Customer.country').
        filters maps 'property' or 'Concept.property' to a value or {operator: value}
        (operators: eq, ne, gt, gte, lt, lte, in, contains)."""
        return layer.compute_metric(name, group_by, filters, limit)

    @tool()
    def find_entities(concept: str, filters: Optional[dict[str, Any]] = None, limit: int = 20,
                      order_by: Optional[str] = None, descending: bool = False) -> dict[str, Any]:
        """List entities of a concept, optionally filtered, e.g. concept='Order', filters={'total': {'gt': 1000}}."""
        return layer.find_entities(concept, filters, limit, order_by, descending)

    @tool()
    def get_entity(concept: str, key: str) -> dict[str, Any]:
        """Fetch one entity by key with its properties, provenance (source file and row) and neighbours."""
        return layer.get_entity(concept, key)

    @tool()
    def get_neighbors(concept: str, key: str, relation: Optional[str] = None, direction: str = "both",
                      limit: int = 50) -> dict[str, Any]:
        """Entities directly connected to an entity. direction is 'out', 'in' or 'both'."""
        return layer.get_neighbors(concept, key, relation, direction, limit)

    @tool()
    def find_path(from_concept: str, from_key: str, to_concept: str, to_key: str, max_hops: int = 6) -> dict[str, Any]:
        """Shortest chain of relationships connecting two entities."""
        return layer.find_path(from_concept, from_key, to_concept, to_key, max_hops)

    @tool()
    def resolve_value(text: str) -> dict[str, Any]:
        """Find which concept property contains a data value, e.g. 'Germany' -> Customer.country."""
        return layer.resolve_value(text)

    @server.resource("ontology://schema", mime_type="application/json")
    def schema_resource() -> str:
        """The full ontology overview as JSON."""
        return json.dumps(layer.overview(), indent=2)

    @server.resource("ontology://concept/{name}", mime_type="application/json")
    def concept_resource(name: str) -> str:
        """One concept's full description as JSON."""
        return json.dumps(layer.describe_concept(name), indent=2, default=str)

    return server


def serve(ontology_path: "str | Path", transport: str = "stdio", host: str = "127.0.0.1", port: int = 8000) -> None:
    server = build_server(ontology_path)
    if transport == "stdio":
        server.run("stdio")
        return
    import uvicorn  # installed as a dependency of the mcp package

    app = server.streamable_http_app() if transport == "streamable-http" else server.sse_app()
    uvicorn.run(app, host=host, port=port)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="ontolayer-mcp", description="Serve an ontology over MCP")
    parser.add_argument("--ontology", default=os.environ.get("ONTOLAYER_ONTOLOGY"), required=not os.environ.get("ONTOLAYER_ONTOLOGY"),
                        help="Path to the ontology YAML (or set ONTOLAYER_ONTOLOGY)")
    parser.add_argument("--transport", choices=["stdio", "streamable-http", "sse"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    serve(args.ontology, args.transport, args.host, args.port)


if __name__ == "__main__":
    main()
