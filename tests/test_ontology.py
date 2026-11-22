import pytest
import yaml

from ontolayer import KnowledgeGraph, OntologyError, GraphBuildError, load_ontology
from ontolayer.graph import coerce_value
from ontolayer.loader import dump_ontology, to_jsonld
from ontolayer.schema import Ontology


def _base():
    return {"name": "t", "concepts": [{"name": "A", "properties": [{"name": "x", "type": "integer"}]}]}


def test_loads_retail(retail_path):
    onto = load_ontology(retail_path)
    assert [c.name for c in onto.concepts] == ["Customer", "Product", "Order", "OrderItem"]
    assert onto.get_metric("revenue").aggregation == "sum"


def test_roundtrip_yaml(retail_path):
    onto = load_ontology(retail_path)
    again = Ontology.model_validate(yaml.safe_load(dump_ontology(onto)))
    assert again == onto


@pytest.mark.parametrize("mutate,message", [
    (lambda d: d["concepts"].append({"name": "a"}), "duplicate concept"),
    (lambda d: d.update(relations=[{"name": "r", "from": "A", "to": "Nope"}]), "unknown concept 'Nope'"),
    (lambda d: d.update(metrics=[{"name": "m", "concept": "A", "aggregation": "sum"}]), "needs a property"),
    (lambda d: d.update(metrics=[{"name": "m", "concept": "A", "aggregation": "sum", "property": "y"}]), "no property 'y'"),
    (lambda d: d.update(metrics=[{"name": "m", "concept": "A", "aggregation": "count", "filters": {"zzz": 1}}]), "zzz"),
])
def test_validation_errors(mutate, message):
    data = _base()
    mutate(data)
    with pytest.raises(ValueError, match=message):
        Ontology.model_validate(data)


def test_cannot_sum_strings():
    data = _base()
    data["concepts"][0]["properties"].append({"name": "s", "type": "string"})
    data["metrics"] = [{"name": "m", "concept": "A", "aggregation": "sum", "property": "s"}]
    with pytest.raises(ValueError, match="cannot sum"):
        Ontology.model_validate(data)


def test_loader_errors(tmp_path):
    with pytest.raises(OntologyError, match="not found"):
        load_ontology(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nconcepts: [{name: A, bogus: 1}]\n")
    with pytest.raises(OntologyError, match="bogus"):
        load_ontology(bad)


def test_graph_counts(layer):
    s = layer.kg.stats()
    assert s["nodes_by_concept"] == {"Customer": 20, "Product": 12, "Order": 60, "OrderItem": 144}
    assert s["edges_by_relation"] == {"placed_by": 60, "part_of": 144, "for_product": 144}
    assert s["warnings"] == []


def test_missing_column_is_clear(tmp_path):
    (tmp_path / "a.csv").write_text("id,x\n1,2\n")
    onto = Ontology.model_validate({"name": "t", "concepts": [{
        "name": "A", "source": {"adapter": "csv", "path": "a.csv", "key": "id"},
        "properties": [{"name": "y", "type": "integer"}]}]})
    with pytest.raises(GraphBuildError, match=r"\['y'\]"):
        KnowledgeGraph.build(onto, tmp_path)


def test_dangling_and_duplicate_warnings(tmp_path):
    (tmp_path / "a.csv").write_text("id,b_id,n\n1,9,x\n1,1,y\n2,,z\n")
    (tmp_path / "b.csv").write_text("id\n1\n")
    onto = Ontology.model_validate({"name": "t", "concepts": [
        {"name": "A", "source": {"adapter": "csv", "path": "a.csv", "key": "id"}, "properties": [{"name": "n"}]},
        {"name": "B", "source": {"adapter": "csv", "path": "b.csv", "key": "id"}}],
        "relations": [{"name": "to_b", "from": "A", "to": "B", "foreign_key": "b_id"}]})
    stats = KnowledgeGraph.build(onto, tmp_path).stats()
    assert stats["nodes_by_concept"] == {"A": 2, "B": 1}
    assert any("duplicate" in w for w in stats["warnings"])
    assert any("dangling" in w for w in stats["warnings"])


def test_sqlite_and_json_adapters(tmp_path):
    import sqlite3
    con = sqlite3.connect(tmp_path / "s.db")
    con.execute("CREATE TABLE people (pid TEXT, age INTEGER)")
    con.execute("INSERT INTO people VALUES ('p1', 30), ('p2', 41)")
    con.commit(); con.close()
    (tmp_path / "pets.json").write_text('[{"id": "d1", "owner": "p1"}, {"id": "d2", "owner": "p2"}]')
    onto = Ontology.model_validate({"name": "t", "concepts": [
        {"name": "Person", "source": {"adapter": "sqlite", "path": "s.db", "table": "people", "key": "pid"},
         "properties": [{"name": "age", "type": "integer"}]},
        {"name": "Pet", "source": {"adapter": "json", "path": "pets.json", "key": "id"}}],
        "relations": [{"name": "owned_by", "from": "Pet", "to": "Person", "foreign_key": "owner"}]})
    kg = KnowledgeGraph.build(onto, tmp_path)
    assert kg.stats()["edges"] == 2
    assert kg.node("Person", "p2")["properties"]["age"] == 41


def test_sqlite_rejects_bad_table_name(tmp_path):
    from ontolayer.adapters import get_adapter
    from ontolayer.schema import Source
    (tmp_path / "s.db").write_bytes(b"")
    with pytest.raises(GraphBuildError):
        get_adapter(Source(adapter="sqlite", path="s.db", table='x"; DROP TABLE y;--', key="id"), tmp_path)


@pytest.mark.parametrize("value,typ,expected", [
    ("12", "integer", (12, True)), ("1.5", "integer", (None, False)), ("", "float", (None, True)),
    ("yes", "boolean", (True, True)), ("2025-03-01", "date", ("2025-03-01", True)),
    ("nope", "date", (None, False)), ("abc", "float", (None, False)),
])
def test_coerce(value, typ, expected):
    assert coerce_value(value, typ) == expected


def test_exports(retail_path, tmp_path):
    onto = load_ontology(retail_path)
    ld = to_jsonld(onto)
    assert any(n["@id"] == "ex:Customer" for n in ld["@graph"])
    out = tmp_path / "g.db"
    KnowledgeGraph.build(onto, retail_path.parent).export_sqlite(out)
    import sqlite3
    assert sqlite3.connect(out).execute("select count(*) from nodes").fetchone()[0] == 236
