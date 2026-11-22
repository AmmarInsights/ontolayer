from ontolayer.cli import main
from ontolayer.schema import Ontology
from ontolayer.suggest import suggest_ontology


def test_suggest_infers_keys_types_relations(retail_path):
    draft = suggest_ontology(retail_path.parent, "draft")
    Ontology.model_validate(draft)
    names = {c["name"]: c for c in draft["concepts"]}
    assert set(names) == {"Customer", "OrderItem", "Order", "Product"}
    assert names["Order"]["source"]["key"] == "order_id"
    types = {p["name"]: p["type"] for p in names["Order"]["properties"]}
    assert types == {"order_date": "date", "status": "string", "total": "float"}
    rels = {(r["from"], r["to"], r["foreign_key"]) for r in draft["relations"]}
    assert ("Order", "Customer", "customer_id") in rels and ("OrderItem", "Product", "product_id") in rels


def test_cli(retail_path, tmp_path, capsys):
    assert main(["validate", str(retail_path)]) == 0
    assert "236 nodes" in capsys.readouterr().out
    out = tmp_path / "draft.yaml"
    assert main(["suggest", str(retail_path.parent), "-o", str(out)]) == 0
    assert main(["validate", str(out)]) == 0
    assert main(["validate", str(tmp_path / "missing.yaml")]) == 1
