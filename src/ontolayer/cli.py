"""Command line interface: ``ontolayer <command>``."""
import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

from .errors import OntoLayerError


def _stats(path: str) -> dict:
    from .graph import KnowledgeGraph
    from .loader import load_ontology

    onto = load_ontology(path)
    return KnowledgeGraph.build(onto, Path(path).parent).stats()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="ontolayer", description="Business ontology layer with an MCP server.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("validate", help="Validate an ontology file and build its graph")
    p.add_argument("ontology")
    p = sub.add_parser("info", help="Print graph statistics")
    p.add_argument("ontology")
    p = sub.add_parser("serve", help="Serve the ontology over MCP")
    p.add_argument("ontology")
    p.add_argument("--transport", choices=["stdio", "streamable-http", "sse"], default="stdio")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p = sub.add_parser("ask", help="Ask a question through an MCP-connected agent")
    p.add_argument("ontology")
    p.add_argument("question")
    p.add_argument("--offline", action="store_true", help="Use the rule-based planner (no LLM needed)")
    p = sub.add_parser("suggest", help="Draft an ontology YAML from a folder of CSV files")
    p.add_argument("csv_dir")
    p.add_argument("--name", default="draft")
    p.add_argument("-o", "--output")
    p = sub.add_parser("export-jsonld", help="Export the ontology schema as JSON-LD")
    p.add_argument("ontology")
    p.add_argument("-o", "--output")
    p = sub.add_parser("export-sqlite", help="Export the built knowledge graph to SQLite")
    p.add_argument("ontology")
    p.add_argument("output")
    sub.add_parser("schema", help="Print the JSON Schema for ontology files")

    args = parser.parse_args(argv)
    try:
        if args.cmd == "validate":
            stats = _stats(args.ontology)
            print(f"OK: {stats['nodes']} nodes, {stats['edges']} edges")
            for w in stats["warnings"]:
                print(f"warning: {w}")
        elif args.cmd == "info":
            print(json.dumps(_stats(args.ontology), indent=2))
        elif args.cmd == "serve":
            from .mcp_server import serve

            serve(args.ontology, args.transport, args.host, args.port)
        elif args.cmd == "ask":
            from .agent import ask

            asyncio.run(ask(args.ontology, args.question, mode="offline" if args.offline else "llm"))
        elif args.cmd == "suggest":
            from .suggest import suggest_yaml

            text = suggest_yaml(args.csv_dir, args.name, Path(args.output).resolve().parent if args.output else None)
            if args.output:
                Path(args.output).write_text(text, encoding="utf-8")
                print(f"Wrote {args.output}")
            else:
                print(text)
        elif args.cmd == "export-jsonld":
            from .loader import load_ontology, to_jsonld

            text = json.dumps(to_jsonld(load_ontology(args.ontology)), indent=2)
            if args.output:
                Path(args.output).write_text(text, encoding="utf-8")
            else:
                print(text)
        elif args.cmd == "export-sqlite":
            from .graph import KnowledgeGraph
            from .loader import load_ontology

            onto = load_ontology(args.ontology)
            KnowledgeGraph.build(onto, Path(args.ontology).parent).export_sqlite(args.output)
            print(f"Wrote {args.output}")
        elif args.cmd == "schema":
            from .loader import json_schema

            print(json_schema())
    except (OntoLayerError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
