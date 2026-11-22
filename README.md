# OntoLayer

**An open-source business ontology layer for AI agents.**
Define business concepts in YAML, build a knowledge graph from your data, and let any agent query it over [MCP](https://modelcontextprotocol.io).

Agents can already query databases, but they guess what "revenue" means, which table holds "customers" and how they relate. OntoLayer gives them a shared, checkable vocabulary:

- **Concepts** with typed properties and synonyms (`Customer`, `Order`, `Product`)
- **Relations** that say how concepts connect (`Order placed_by Customer`)
- **Certified metrics** so "revenue" is computed the same way every time (`sum of Order.total where status = completed`)
- **Answerability checks**: the agent can ask "does this ontology cover my question?" before answering, and say so when the data is missing
- **Provenance**: every entity records which source file and row it came from

```
 CSV / SQLite / JSON ──► adapters ──┐
                                    ├──► knowledge graph ──► semantic layer ──► MCP server ──► any agent
 ontology.yaml ──► validator ───────┘     (networkx)          (metrics, search,     (stdio / HTTP)
                                                               paths, filters)
```

## Quickstart

```bash
pip install -e .                       # Python 3.10+
ontolayer validate examples/retail/ontology.yaml
ontolayer ask examples/retail/ontology.yaml "What was revenue by country for business customers?" --offline
```

```
  -> check_answerable({"question": "What was revenue by country for business customers?"})
  -> compute_metric({"name": "Revenue", "group_by": "Customer.country", "filters": {"Customer.segment": "business"}})
Revenue (sum of Order.total where status = completed) filtered by {'Customer.segment': 'business'}:
  Japan: 2,535.0 USD
  United States: 2,400.0 USD
  Germany: 596.0 USD
```

`--offline` uses a small rule-based planner so you can try everything with no API key. Questions the ontology cannot answer are refused:

```
ontolayer ask examples/retail/ontology.yaml "What was the return on ad spend?" --offline
The ontology does not cover these terms: ['return', 'ad', 'spend']. Use search_concepts to look for related terms, or ask the user to clarify.
```

## Define an ontology

```yaml
name: retail
concepts:
  - name: Customer
    synonyms: [client, buyer]
    source: {adapter: csv, path: customers.csv, key: customer_id}
    properties:
      - {name: country, type: string, description: Where the customer lives}
  - name: Order
    source: {adapter: csv, path: orders.csv, key: order_id}
    properties:
      - {name: status, type: string}
      - {name: total, type: float}
relations:
  - name: placed_by
    from: Order
    to: Customer
    inverse: places
    foreign_key: customer_id        # column in orders.csv holding the customer key
metrics:
  - name: Revenue
    synonyms: [sales, income]
    concept: Order
    aggregation: sum                # count | sum | avg | min | max
    property: total
    filters: {status: completed}
```

Paths are relative to the YAML file. Property types: `string`, `integer`, `float`, `boolean`, `date`. Validation catches unknown concepts, bad metric definitions, duplicate names and missing columns, with messages that say what is wrong.

**Don't want to write YAML by hand?** `ontolayer suggest data_folder/ -o ontology.yaml` drafts one from a folder of CSVs (keys, types and foreign-key relations are inferred from names and sample values). It is a heuristic starting point: review it and add descriptions, synonyms and metrics.

Other commands: `ontolayer info`, `export-jsonld` (schema as JSON-LD), `export-sqlite` (graph as SQLite tables), `schema` (JSON Schema for editor support).

## Adapters

| Adapter | `source` fields | Notes |
|---|---|---|
| `csv` | `path`, `key` | UTF-8, header row |
| `sqlite` | `path`, `table`, `key` | opened read-only, table name validated |
| `json` | `path`, `key` | array of objects, `{"records": [...]}` or `.jsonl` |

Add your own by subclassing `SourceAdapter` (`columns()` and `rows()`) and decorating it with `@register_adapter`.

## MCP server

```bash
python -m ontolayer.mcp_server --ontology examples/retail/ontology.yaml                       # stdio
python -m ontolayer.mcp_server --ontology examples/retail/ontology.yaml \
       --transport streamable-http --port 8000                                                # http://127.0.0.1:8000/mcp
```

All tools are **read-only**.

| Tool | What it does |
|---|---|
| `get_ontology_overview` | Concepts, relations, metrics, graph size. Call first. |
| `check_answerable` | Is the question covered? Matched concepts, metrics, data values, unmatched terms. |
| `search_concepts` | Find concepts, properties, metrics or relations by words (synonym aware). |
| `describe_concept` | Properties, relations in both directions, metrics, sample entities. |
| `list_metrics`, `explain_metric` | Metric definitions and what they can be grouped by. |
| `compute_metric` | Run a certified metric with optional `group_by` (`Customer.country`) and `filters`. |
| `find_entities` | Filter records (`eq ne gt gte lt lte in contains`), including on related concepts. |
| `get_entity`, `get_neighbors`, `find_path` | Walk the graph; entities carry provenance. |
| `resolve_value` | Which property holds a value, e.g. `Germany` -> `Customer.country`. |

Resources: `ontology://schema` and `ontology://concept/{name}`.

Errors are written for agents: an unknown metric returns the list of valid metrics, so the agent can correct itself.

Grouping or filtering across relations follows many-to-one links only (an order has one customer, so `Revenue` by `Customer.country` is well defined; a customer has many orders, so grouping customers by order status is rejected rather than guessed).

### Connect an agent

| Client | How |
|---|---|
| **MCP Inspector** (no LLM, best first check) | `npx @modelcontextprotocol/inspector python -m ontolayer.mcp_server --ontology examples/retail/ontology.yaml` |
| **Claude Code** | `claude mcp add ontolayer -- python -m ontolayer.mcp_server --ontology /abs/path/examples/retail/ontology.yaml` |
| **Claude Desktop / Cursor** | add to the MCP config: `{"mcpServers": {"ontolayer": {"command": "python", "args": ["-m", "ontolayer.mcp_server", "--ontology", "/abs/path/ontology.yaml"]}}}` |
| **Your own agent** | any MCP client over stdio or streamable HTTP. See `src/ontolayer/agent.py`. |

### Built-in demo agent (open-source models)

`ontolayer.agent` is a small MCP client plus tool-calling loop for any OpenAI-compatible endpoint. Pick whatever is cheapest for you:

```bash
# local and free: Ollama
export LLM_BASE_URL=http://localhost:11434/v1 LLM_API_KEY=ollama LLM_MODEL=qwen2.5:7b
# or a hosted OpenAI-compatible provider: set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL from its docs
ontolayer ask examples/retail/ontology.yaml "Which product category sells the most units?"
```

Model names and free-tier limits change, so check your provider's docs.

## Google Colab

Open `notebooks/colab_demo.ipynb`. It installs the package, validates the ontology, starts the MCP server, and runs the agent either offline (no key) or with an API key stored in Colab Secrets.

## Tests

```bash
pip install -e ".[dev]" && pytest -q
```

Covers schema validation, all adapters, graph building, every semantic query, the MCP server over a real stdio connection, and the LLM loop against a fake endpoint. Developed against `mcp` 2.3; the server falls back to the 1.x import path (`FastMCP`) but that path is not covered by tests here.

## Design notes and credits

The design was informed by two projects (ideas only, no code copied):

- [NVIDIA auto-ontology](https://github.com/NVIDIA/auto-ontology): ontology of terms, relationships and certified definitions, answerability checks and MCP tools for agents.
- [Omnigraph](https://github.com/ModernRelay/omnigraph): typed ontology as code that agents share, with a built-in MCP server.

Known limits: the graph is built in memory at start-up (fine for tens of thousands of rows, not millions); the store is a networkx graph behind a small interface so a graph database can replace it later; `suggest` is heuristic, not LLM-based; the offline planner is intentionally simple.

## License

MIT
