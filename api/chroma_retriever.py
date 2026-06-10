from dataclasses import dataclass
from typing import List

from api.types import Document

from api.db.chroma_store import ChromaStore


@dataclass
class ChromaRetrieverResult:
    documents: List[Document]
    doc_indices: List[int]


def extract_embedding_vector(embedder_output) -> List[float]:
    data = getattr(embedder_output, "data", None)
    if data:
        first = data[0]
        embedding = getattr(first, "embedding", None)
        if embedding is not None:
            return list(embedding)
    if isinstance(embedder_output, list):
        return list(embedder_output)
    raise ValueError("Unable to extract query embedding from embedder output")


class ChromaRetriever:
    def __init__(self, store: ChromaStore, embedder, top_k: int = 20):
        self.store = store
        self.embedder = embedder
        self.top_k = top_k

    def __call__(self, query: str):
        embedding = extract_embedding_vector(self.embedder(input=query))
        result = self.store.query(embedding, top_k=self.top_k)
        documents = []
        ids = result.get("ids", [[]])[0]
        texts = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0] if result.get("distances") else []

        for idx, text in enumerate(texts):
            metadata = metadatas[idx] or {}
            if idx < len(ids):
                metadata["chunk_id"] = ids[idx]
            if idx < len(distances):
                metadata["retrieval_distance"] = distances[idx]
            documents.append(Document(text=text, meta_data=metadata))

        return [ChromaRetrieverResult(documents=documents, doc_indices=list(range(len(documents))))]

