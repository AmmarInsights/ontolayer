import csv

import pytest

from ontolayer import SemanticError


def _orders(retail_path):
    with open(retail_path.parent / "orders.csv") as f:
        return list(csv.DictReader(f))


def test_revenue_matches_raw_data(layer, retail_path):
    expected = sum(float(r["total"]) for r in _orders(retail_path) if r["status"] == "completed")
    assert layer.compute_metric("Revenue")["value"] == pytest.approx(expected, abs=0.01)


def test_group_by_across_relation_sums_to_total(layer):
    total = layer.compute_metric("Revenue")["value"]
    groups = layer.compute_metric("Revenue", group_by="Customer.country")["groups"]
    assert sum(g["value"] for g in groups) == pytest.approx(total, abs=0.01)
    assert [g["value"] for g in groups] == sorted((g["value"] for g in groups), reverse=True)


def test_filters_on_related_concept(layer):
    germany = layer.compute_metric("Revenue", filters={"Customer.country": "germany"})["value"]
    by_country = {g["group"]: g["value"] for g in layer.compute_metric("Revenue", group_by="Customer.country")["groups"]}
    assert germany == pytest.approx(by_country["Germany"], abs=0.01)


def test_count_and_avg(layer):
    assert layer.compute_metric("Order Count")["value"] == 60
    avg = layer.compute_metric("Average Order Value")["value"]
    rev = layer.compute_metric("Revenue")
    assert avg == pytest.approx(rev["value"] / rev["rows_considered"], abs=0.01)


def test_find_entities_filters_and_order(layer):
    res = layer.find_entities("Order", {"total": {"gt": 1500}, "status": "completed"}, order_by="total", descending=True, limit=5)
    totals = [e["properties"]["total"] for e in res["entities"]]
    assert all(t > 1500 for t in totals) and totals == sorted(totals, reverse=True)
    assert layer.find_entities("Customer", {"name": {"contains": "KHAN"}})["total_matches"] > 0
    assert layer.find_entities("Customer", {"country": {"in": ["Japan", "Germany"]}})["total_matches"] == 8


def test_errors_are_actionable(layer):
    with pytest.raises(SemanticError, match="Available metrics"):
        layer.compute_metric("Profit")
    with pytest.raises(SemanticError, match="Available properties"):
        layer.find_entities("Customer", {"colour": "red"})
    with pytest.raises(SemanticError, match="Unknown filter operator"):
        layer.find_entities("Order", {"total": {"between": 3}})
    with pytest.raises(SemanticError, match="to-one relation path"):
        layer.compute_metric("Customer Count", group_by="Order.status")
    with pytest.raises(SemanticError, match="Cannot use"):
        layer.find_entities("Order", {"total": {"gt": "lots"}})
    with pytest.raises(SemanticError, match="Example keys"):
        layer.get_entity("Customer", "nope")


def test_neighbors_and_path(layer):
    n = layer.get_neighbors("Order", "O0001")
    rels = {x["relation"] for x in n["neighbors"]}
    assert {"placed_by", "contains"} <= rels
    items = layer.get_neighbors("Order", "O0001", relation="contains")["neighbors"]
    product = layer.get_neighbors("OrderItem", items[0]["key"], relation="for_product")["neighbors"][0]["key"]
    path = layer.find_path("Order", "O0001", "Product", product)
    assert path["connected"] and path["hops"] == 2
    assert [s["relation"] for s in path["steps"]] == ["contains", "for_product"]


def test_describe_and_overview(layer):
    d = layer.describe_concept("customer")
    assert {r["relation"] for r in d["relations"]} == {"places"}
    assert d["entities"] == 20 and d["samples"]
    assert layer.overview()["graph"]["nodes"] == 236
    assert "Customer.country" in layer.explain_metric("Revenue")["can_group_by"]


def test_search_resolve_and_answerable(layer):
    assert layer.search_concepts("buyer")["results"][0]["name"] == "Customer"
    assert layer.search_concepts("sales")["results"][0]["name"] == "Revenue"
    assert layer.resolve_value("germany")["matches"][0]["property"] == "country"
    ok = layer.check_answerable("What was revenue by country for business customers?")
    assert ok["answerable"] and ok["matched"]["metrics"][0]["name"] == "Revenue"
    bad = layer.check_answerable("What was the return on ad spend?")
    assert not bad["answerable"] and "ad" in bad["unmatched_terms"]
