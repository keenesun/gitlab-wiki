import hashlib
import logging
import os
from typing import Iterable, List, Optional

from api.types import Document

logger = logging.getLogger(__name__)


def repo_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def collection_name_for_repo(repo_url_or_path: str) -> str:
    return f"repo_{repo_hash(repo_url_or_path)}"


class ChromaStore:
    """Thin ChromaDB adapter with lazy imports so the app can fail with a clear setup message."""

    def __init__(self, persist_dir: str, repo_url_or_path: str):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("ChromaDB is not installed. Run `poetry install` or install `chromadb`.") from exc

        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = collection_name_for_repo(repo_url_or_path)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    @staticmethod
    def _metadata(doc: Document) -> dict:
        metadata = dict(getattr(doc, "meta_data", {}) or {})
        return {
            key: value
            for key, value in metadata.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }

    @staticmethod
    def _vector(doc: Document) -> Optional[List[float]]:
        vector = getattr(doc, "vector", None)
        if vector is None:
            return None
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        return list(vector)

    def upsert_documents(self, documents: Iterable[Document]) -> None:
        ids = []
        texts = []
        embeddings = []
        metadatas = []

        for doc in documents:
            metadata = self._metadata(doc)
            chunk_id = metadata.get("chunk_id")
            vector = self._vector(doc)
            if not chunk_id or vector is None:
                logger.warning("Skipping Chroma upsert for document without chunk_id or vector")
                continue
            ids.append(chunk_id)
            texts.append(doc.text)
            embeddings.append(vector)
            metadatas.append(metadata)

        if not ids:
            return

        self.collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    def delete_by_file(self, repo_id: str, file_path: str) -> None:
        self.collection.delete(where={"$and": [{"repo_id": repo_id}, {"file_path": file_path}]})

    def query(self, embedding: List[float], top_k: int = 20):
        return self.collection.query(query_embeddings=[embedding], n_results=top_k)

