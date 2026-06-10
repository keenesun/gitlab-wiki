"""Standalone types to replace adalflow dependencies."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    """A document with text, metadata, and an optional embedding vector.

    Replaces adalflow.core.types.Document so the project no longer depends
    on adalflow for its core data type.
    """

    text: str
    meta_data: Dict[str, Any] = field(default_factory=dict)
    vector: Optional[List[float]] = None


def deepwiki_root_path() -> str:
    """Return the DeepWiki data directory.

    Uses DEEPWIKI_DATA_DIR env var if set, otherwise ~/.deepwiki.
    Replaces adalflow.utils.get_adalflow_default_root_path.
    """
    env = os.environ.get("DEEPWIKI_DATA_DIR")
    if env:
        return os.path.expanduser(env)
    return os.path.expanduser(os.path.join("~", ".deepwiki"))
