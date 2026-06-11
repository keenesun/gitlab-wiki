"""Minimal types replacing adalflow.core.types for model clients."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ModelType(Enum):
    UNDEFINED = "undefined"
    LLM = "llm"
    EMBEDDER = "embedder"


@dataclass
class Embedding:
    """A single embedding vector."""
    embedding: List[float]
    index: int = 0


@dataclass
class EmbedderOutput:
    """Output from an embedding model call."""
    data: List[Embedding] = field(default_factory=list)
    error: Optional[str] = None
    raw_response: Any = None
    usage: Optional[Any] = None


@dataclass
class GeneratorOutput:
    """Output from an LLM generator call."""
    data: Optional[str] = None
    error: Optional[str] = None
    raw_response: Any = None
    usage: Optional[Any] = None


@dataclass
class CompletionUsage:
    """Token usage stats."""
    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0


@dataclass
class TokenLogProb:
    """Token log probability."""
    token: str = ""
    logprob: float = 0.0


# Type aliases used by DashScope
EmbedderOutputType = EmbedderOutput
EmbedderInputType = Union[str, List[str], List[Dict[str, Any]]]
BatchEmbedderOutputType = List[EmbedderOutput]
BatchEmbedderInputType = List[EmbedderInputType]


class DataComponent:
    """Minimal base class for data processing components.

    Replaces adalflow.core.component.DataComponent.
    """

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError
