"""Exception types used across OntoLayer."""


class OntoLayerError(Exception):
    """Base class for all OntoLayer errors."""


class OntologyError(OntoLayerError, ValueError):
    """The ontology file is missing, malformed or internally inconsistent."""


class GraphBuildError(OntoLayerError):
    """The knowledge graph could not be built from the configured sources."""


class SemanticError(OntoLayerError, ValueError):
    """A semantic-layer query referenced something that does not exist or is unsupported."""
