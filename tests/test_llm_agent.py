"""Verify the LLM tool-calling loop against a fake OpenAI-compatible endpoint (no API key needed)."""
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from ontolayer.agent import LLMConfig, ask


def _serve(replies):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            payload = json.dumps({"choices": [{"message": replies[min(len(seen) - 1, len(replies) - 1)]}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, seen


def test_llm_loop_calls_mcp_tools_and_returns_answer(retail_path):
    tool_call = {"id": "c1", "type": "function",
                 "function": {"name": "compute_metric", "arguments": json.dumps({"name": "Order Count"})}}
    server, seen = _serve([{"role": "assistant", "content": None, "tool_calls": [tool_call]},
                           {"role": "assistant", "content": "There are 60 orders."}])
    try:
        cfg = LLMConfig(f"http://127.0.0.1:{server.server_port}/v1", "k", "fake")
        result = asyncio.run(ask(retail_path, "How many orders?", mode="llm", cfg=cfg, verbose=False))
    finally:
        server.shutdown()
    assert result.answer == "There are 60 orders."
    assert [t["tool"] for t in result.trace] == ["compute_metric"]
    assert '"value": 60' in result.trace[0]["result"]
    # the second request must contain the tool result and the full MCP tool list was advertised
    assert any(m["role"] == "tool" for m in seen[1]["messages"])
    assert len(seen[0]["tools"]) == 12
