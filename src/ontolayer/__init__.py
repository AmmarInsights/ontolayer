"""OntoLayer: a business ontology layer that agents can query over MCP."""
from .errors import GraphBuildError, OntoLayerError, OntologyError, SemanticError
from .graph import KnowledgeGraph
from .loader import load_ontology
from .schema import Ontology
from .semantic import SemanticLayer

__version__ = "0.1.0"

__all__ = [
    "GraphBuildError", "KnowledgeGraph", "OntoLayerError", "Ontology", "OntologyError",
    "SemanticError", "SemanticLayer", "load_ontology", "__version__",
]
