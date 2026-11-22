"""Integration test: talk to the real MCP server over stdio, like any external agent."""
import asyncio
import json

from ontolayer.agent import ask, call, connect

EXPECTED_TOOLS = {"get_ontology_overview", "search_concepts", "check_answerable", "describe_concept", "list_metrics",
                  "explain_metric", "compute_metric", "find_entities", "get_entity", "get_neighbors", "find_path",
                  "resolve_value"}


def test_tools_resources_and_calls(retail_path):
    async def go():
        async with connect(retail_path) as session:
            tools = {t.name for t in (await session.list_tools()).tools}
            assert EXPECTED_TOOLS <= tools
            res = await session.list_resources()
            assert "ontology://schema" in {str(r.uri) for r in res.resources}
            trace = []
            revenue = await call(session, "compute_metric", {"name": "Revenue", "group_by": "Customer.country"}, trace)
            assert revenue["groups"][0]["group"] == "Japan"
            err = await session.call_tool("compute_metric", {"name": "Nope"})
            assert getattr(err, "is_error", getattr(err, "isError", False))
            assert "Available metrics" in err.content[0].text
            schema = await session.read_resource("ontology://schema")
            assert json.loads(schema.contents[0].text)["ontology"] == "retail"
    asyncio.run(go())


def test_offline_agent_answers_and_refuses(retail_path):
    async def go():
        good = await ask(retail_path, "How many orders were cancelled?", mode="offline", verbose=False)
        assert "8" in good.answer
        bad = await ask(retail_path, "What was the return on ad spend?", mode="offline", verbose=False)
        assert "does not cover" in bad.answer
    asyncio.run(go())
