"""Ollama model client using the ollama Python package.

Replaces adalflow's built-in OllamaClient.
"""

import logging
from typing import Any, Dict, Optional

from api.model_client import ModelClient

logger = logging.getLogger(__name__)


class OllamaChunk:
    """Wrapper so streaming chunks have a .response attribute for backward compat."""

    def __init__(self, content: str):
        self.response = content
        self.text = content


class OllamaClient(ModelClient):
    """Minimal Ollama client using the ollama Python package.

    Replaces adalflow.components.model_client.ollama_client.OllamaClient.
    """

    def __init__(self, host: Optional[str] = None):
        super().__init__()
        import os
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def init_sync_client(self):
        try:
            import ollama
            ollama_client = ollama.Client(host=self.host)
            return ollama_client
        except ImportError:
            raise ImportError("ollama package not installed. Run: pip install ollama")

    def init_async_client(self):
        try:
            import ollama
            return ollama.AsyncClient(host=self.host)
        except ImportError:
            raise ImportError("ollama package not installed. Run: pip install ollama")

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        model_type: Any = None,
    ) -> Dict[str, Any]:
        """Convert prompt + model_kwargs to Ollama API kwargs."""
        kwargs = dict(model_kwargs or {})

        if input is not None:
            kwargs["messages"] = [{"role": "user", "content": input}]

        # Extract ollama-specific options
        options = kwargs.pop("options", {})
        if options:
            kwargs["options"] = options

        return kwargs

    def call(self, api_kwargs: Dict[str, Any], model_type: Any = None):
        """Synchronous call — used for embeddings."""
        client = self.init_sync_client()
        messages = api_kwargs.get("messages", [])
        prompt = messages[0]["content"] if messages else ""

        if api_kwargs.get("embedding", False) or model_type is not None and str(model_type) == "embedder":
            result = client.embeddings(model=api_kwargs.get("model"), prompt=prompt)
            # Return in EmbedderOutput-compatible format
            return type("OllamaEmbedResult", (), {
                "data": [type("OllamaEmbedItem", (), {"embedding": result["embedding"]})()]
            })()

        result = client.chat(
            model=api_kwargs.get("model", "llama3"),
            messages=messages,
            stream=False,
            options=api_kwargs.get("options"),
        )
        content = result.get("message", {}).get("content", "")
        return type("OllamaChatResult", (), {"response": content, "text": content})()

    async def acall(self, api_kwargs: Dict[str, Any], model_type: Any = None):
        """Async streaming call — used for chat."""
        client = self.init_async_client()
        messages = api_kwargs.get("messages", [])
        stream = api_kwargs.get("stream", True)

        async for chunk in await client.chat(
            model=api_kwargs.get("model", "llama3"),
            messages=messages,
            stream=stream,
            options=api_kwargs.get("options"),
        ):
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield OllamaChunk(content=content)
