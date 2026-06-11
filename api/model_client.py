"""Minimal ModelClient base class and utilities replacing adalflow."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from api.model_types import EmbedderOutput, Embedding, ModelType

logger = logging.getLogger(__name__)


class ModelClient(ABC):
    """Minimal base class for model clients.

    Replaces adalflow.core.model_client.ModelClient.
    Subclasses must implement init_sync_client() and init_async_client().
    """

    def __init__(self):
        self.sync_client = None
        self.async_client = None
        self._api_kwargs: Dict[str, Any] = {}

    @abstractmethod
    def init_sync_client(self):
        """Initialize and return a synchronous client."""

    @abstractmethod
    def init_async_client(self):
        """Initialize and return an asynchronous client."""

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        model_type: Optional[ModelType] = None,
    ) -> Dict[str, Any]:
        """Convert inputs to the format expected by the underlying API client.

        Override in subclasses for provider-specific conversion.
        """
        kwargs = dict(model_kwargs or {})
        if input is not None:
            kwargs["input"] = input
        return kwargs

    async def acall(self, api_kwargs: Dict[str, Any], model_type: Optional[ModelType] = None):
        """Async call to the model. Override in subclasses."""
        raise NotImplementedError

    def call(self, api_kwargs: Dict[str, Any], model_type: Optional[ModelType] = None):
        """Sync call to the model. Override in subclasses."""
        raise NotImplementedError


class _MissingModule:
    """Lazy proxy that defers ImportError until the module is actually used."""

    def __init__(self, package_names: List[str], error_message: str):
        self._names = package_names
        self._msg = error_message or f"Install with: pip install {package_names[0]}"

    def __getattr__(self, name):
        raise ImportError(
            f"None of these packages could be imported: {', '.join(self._names)}. {self._msg}"
        )


def safe_import(package_names: List[str], error_message: str = ""):
    """Lazy-import packages with a friendly error message.

    Returns a module if a single name is given and found.
    Returns a list of modules if multiple names are given.
    Returns a _MissingModule proxy that defers the error if nothing is found.

    Replaces adalflow.utils.lazy_import.safe_import.
    """
    import importlib

    results = []
    for name in package_names:
        # PyPI package names use hyphens, Python imports use dots
        module_name = name.replace("-", ".")
        try:
            results.append(importlib.import_module(module_name))
        except ImportError:
            continue

    if results:
        if len(package_names) == 1:
            return results[0]
        return results

    # Return lazy proxy so module-level imports don't crash
    return _MissingModule(package_names, error_message)


class OptionalPackages:
    """Package name constants for lazy imports."""

    class OPENAI:
        value = (["openai"], "openai is not installed. Run: pip install openai")

    class AZURE:
        value = (
            ["azure-identity", "azure-core"],
            "Azure packages not installed. Run: pip install azure-identity azure-core",
        )


def parse_embedding_response(response) -> EmbedderOutput:
    """Parse an OpenAI-compatible embedding response into EmbedderOutput.

    Replaces adalflow.components.model_client.utils.parse_embedding_response.
    """
    try:
        data = response.data if hasattr(response, "data") else response.get("data", [])
        embeddings = []
        for i, item in enumerate(data):
            emb = item.embedding if hasattr(item, "embedding") else item.get("embedding", [])
            embeddings.append(Embedding(embedding=list(emb), index=i))
        return EmbedderOutput(data=embeddings, raw_response=response)
    except Exception as e:
        logger.error(f"Error parsing embedding response: {e}")
        return EmbedderOutput(data=[], error=str(e), raw_response=response)
