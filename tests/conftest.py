from pathlib import Path

import pytest

from ontolayer import KnowledgeGraph, SemanticLayer, load_ontology

RETAIL = Path(__file__).resolve().parents[1] / "examples" / "retail"


@pytest.fixture(scope="session")
def retail_path():
    return RETAIL / "ontology.yaml"


@pytest.fixture(scope="session")
def layer(retail_path):
    onto = load_ontology(retail_path)
    return SemanticLayer(KnowledgeGraph.build(onto, RETAIL))
