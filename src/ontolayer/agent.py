"""A small MCP client agent for the demo.

Two modes:

* ``llm``     - any OpenAI-compatible chat endpoint with tool calling (Gemini, Groq, OpenRouter, Ollama, ...).
* ``offline`` - a rule-based planner that needs no model or API key. Good for smoke tests and the demo
                when you have no key; it is intentionally simple.

The agent talks to the ontology only through MCP tools, exactly like any external agent would.
"""
import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SYSTEM_PROMPT = """You are a careful business analyst. Answer questions using ONLY the ontology tools.
Workflow: check_answerable -> describe_concept / explain_metric -> compute_metric or find_entities.
Use exact concept, property and metric names from the tools. Never invent numbers; quote tool results.
If the ontology does not cover the question, say what is missing instead of guessing. Keep answers short."""

MAX_TOOL_CHARS = 5000


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Read LLM_BASE_URL, LLM_API_KEY and LLM_MODEL (Ollama defaults need no key)."""
        return cls(
            base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/"),
            api_key=os.environ.get("LLM_API_KEY", "ollama"),
            model=os.environ.get("LLM_MODEL", "qwen2.5:7b"),
        )


@dataclass
class AgentResult:
    answer: str
    trace: list[dict[str, Any]] = field(default_factory=list)


@asynccontextmanager
async def connect(ontology_path: "str | Path", env: Optional[dict[str, str]] = None):
    """Start the OntoLayer MCP server as a subprocess and yield a ready ClientSession."""
    src_dir = str(Path(__file__).resolve().parents[1])
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "ontolayer.mcp_server", "--ontology", str(ontology_path)],
        env={"PYTHONPATH": src_dir, **(env or {})},
    )
    # Notebooks replace sys.stderr with an object that has no file descriptor, so send server logs to devnull.
    with open(os.devnull, "w") as errlog:
        async with stdio_client(params, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


def _text(result: Any) -> str:
    parts = [getattr(c, "text", "") for c in getattr(result, "content", [])]
    text = "\n".join(p for p in parts if p)
    return text[:MAX_TOOL_CHARS] + ("\n...[truncated]" if len(text) > MAX_TOOL_CHARS else "")


async def call(session: ClientSession, name: str, args: dict[str, Any], trace: list[dict[str, Any]]) -> Any:
    result = await session.call_tool(name, args)
    text = _text(result)
    trace.append({"tool": name, "args": args, "error": bool(getattr(result, "is_error", getattr(result, "isError", False))),
                  "result": text})
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


# ----------------------------------------------------------------------------------------------
# LLM mode
# ----------------------------------------------------------------------------------------------
def _chat(cfg: LLMConfig, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
    resp = requests.post(
        f"{cfg.base_url}/chat/completions",
        headers={"Authorization": f"Bearer {cfg.api_key}"},
        json={"model": cfg.model, "messages": messages, "tools": tools, "temperature": 0},
        timeout=120,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"LLM endpoint returned {resp.status_code}: {resp.text[:500]}")
    return resp.json()["choices"][0]["message"]


async def run_llm_agent(session: ClientSession, question: str, cfg: Optional[LLMConfig] = None,
                        max_steps: int = 8, verbose: bool = True) -> AgentResult:
    cfg = cfg or LLMConfig.from_env()
    listed = await session.list_tools()
    tools = [{"type": "function", "function": {"name": t.name, "description": t.description or "",
                                                 "parameters": getattr(t, "input_schema", None) or t.inputSchema}} for t in listed.tools]
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT},
                                      {"role": "user", "content": question}]
    trace: list[dict[str, Any]] = []
    for _ in range(max_steps):
        msg = await asyncio.to_thread(_chat, cfg, messages, tools)
        calls = msg.get("tool_calls") or []
        if not calls:
            return AgentResult(answer=(msg.get("content") or "").strip(), trace=trace)
        messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": calls})
        for tc in calls:
            fn = tc["function"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if verbose:
                print(f"  -> {fn['name']}({json.dumps(args)})")
            result = await session.call_tool(fn["name"], args)
            text = _text(result)
            trace.append({"tool": fn["name"], "args": args, "result": text})
            messages.append({"role": "tool", "tool_call_id": tc.get("id", fn["name"]), "content": text})
    return AgentResult(answer="(stopped: too many tool calls)", trace=trace)


# ----------------------------------------------------------------------------------------------
# Offline (rule-based) mode
# ----------------------------------------------------------------------------------------------
def _fmt_groups(res: dict[str, Any]) -> str:
    unit = f" {res['unit']}" if res.get("unit") else ""
    if "groups" in res:
        lines = [f"  {g['group']}: {g['value']:,}{unit}" if isinstance(g["value"], (int, float)) else f"  {g['group']}: {g['value']}"
                 for g in res["groups"]]
        return "\n".join(lines)
    value = res["value"]
    return f"{value:,}{unit}" if isinstance(value, (int, float)) else str(value)


async def run_offline_agent(session: ClientSession, question: str, verbose: bool = True) -> AgentResult:
    trace: list[dict[str, Any]] = []
    check = await call(session, "check_answerable", {"question": question}, trace)
    matched = check["matched"]
    if not check["answerable"]:
        return AgentResult(check.get("suggestion", "The ontology cannot answer this question."), trace)

    filters: dict[str, Any] = {}
    for v in matched["values"]:
        filters[f"{v['concept']}.{v['property']}"] = v["value"]
    q = question.lower()
    wants_groups = any(w in f" {q} " for w in (" by ", " per ", " each ", " across "))
    group_prop = next((f"{p['concept']}.{p['name']}" for p in matched["properties"]
                       if f"{p['concept']}.{p['name']}" not in filters), None)

    if matched["metrics"]:
        metric = matched["metrics"][0]["name"]
        args: dict[str, Any] = {"name": metric}
        if wants_groups and group_prop:
            args["group_by"] = group_prop
        if filters:
            args["filters"] = filters
        res = await call(session, "compute_metric", args, trace)
        if isinstance(res, str):  # tool error (for example an unsupported group_by); retry without it
            args.pop("group_by", None)
            res = await call(session, "compute_metric", args, trace)
        if isinstance(res, str):
            return AgentResult(f"Could not compute {metric}: {res}", trace)
        header = f"{metric} ({res['definition']})"
        if filters:
            header += f" filtered by {filters}"
        return AgentResult(f"{header}:\n{_fmt_groups(res)}", trace)

    if matched["concepts"]:
        concept = matched["concepts"][0]["name"]
        res = await call(session, "find_entities", {"concept": concept, "filters": {k.split('.', 1)[1]: v for k, v in filters.items()
                                                                                    if k.startswith(concept + ".")}, "limit": 5}, trace)
        if isinstance(res, str):
            return AgentResult(res, trace)
        rows = [f"  {e['key']}: {e['properties']}" for e in res["entities"]]
        return AgentResult(f"{res['total_matches']} {concept} record(s) match. First {res['returned']}:\n" + "\n".join(rows), trace)

    return AgentResult("I could not map the question to a metric or concept.", trace)


async def ask(ontology_path: "str | Path", question: str, mode: str = "llm", cfg: Optional[LLMConfig] = None,
              verbose: bool = True) -> AgentResult:
    """Connect to the MCP server and answer one question. mode is 'llm' or 'offline'."""
    async with connect(ontology_path) as session:
        if mode == "offline":
            result = await run_offline_agent(session, question, verbose)
        else:
            result = await run_llm_agent(session, question, cfg, verbose=verbose)
    if verbose:
        for step in result.trace:
            if mode == "offline":
                print(f"  -> {step['tool']}({json.dumps(step['args'])})")
        print(result.answer)
    return result
