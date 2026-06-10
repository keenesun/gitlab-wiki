import hashlib
import re
from dataclasses import dataclass
from typing import List

from api.types import Document


CHUNKER_VERSION = "treesitter-v1"


@dataclass
class Chunk:
    text: str
    file_path: str
    chunk_index: int
    content_hash: str
    start_line: int
    end_line: int


def sha256_text(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def stable_chunk_id(repo_id: str, file_path: str, chunk_index: int, content_hash: str) -> str:
    repo_part = sha256_text(repo_id)
    file_part = sha256_text(file_path)
    return f"{repo_part}:{file_part}:{chunk_index}:{content_hash[:8]}"


def enrich_transformed_documents(documents: List[Document], repo_id: str) -> List[Document]:
    """Attach stable chunk metadata to documents produced by the existing splitter/embedder pipeline."""
    counters = {}
    for doc in documents:
        metadata = getattr(doc, "meta_data", None) or {}
        file_path = metadata.get("file_path") or metadata.get("title") or "unknown"
        chunk_index = counters.get(file_path, 0)
        counters[file_path] = chunk_index + 1

        content_hash = sha256_text(doc.text, length=64)
        metadata["repo_id"] = repo_id
        metadata["file_path"] = file_path
        metadata["chunk_index"] = chunk_index
        metadata["content_hash"] = content_hash
        metadata["chunk_id"] = stable_chunk_id(repo_id, file_path, chunk_index, content_hash)
        metadata.setdefault("start_line", None)
        metadata.setdefault("end_line", None)
        metadata["chunker_version"] = CHUNKER_VERSION
        doc.meta_data = metadata
    return documents


def chunk_documents(documents: List[Document], repo_id: str) -> List[Document]:
    """Split source-file documents before embedding and attach stable chunk metadata."""
    chunked = []
    for doc in documents:
        metadata = getattr(doc, "meta_data", None) or {}
        file_path = metadata.get("file_path") or metadata.get("title") or "unknown"
        file_type = (metadata.get("type") or file_path.rsplit(".", 1)[-1]).lower()
        chunks = _chunk_text(doc.text, file_path, file_type)

        for chunk in chunks:
            chunk_metadata = dict(metadata)
            chunk_metadata.update(
                {
                    "repo_id": repo_id,
                    "file_path": file_path,
                    "chunk_index": chunk.chunk_index,
                    "content_hash": chunk.content_hash,
                    "chunk_id": stable_chunk_id(repo_id, file_path, chunk.chunk_index, chunk.content_hash),
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "chunker_version": CHUNKER_VERSION,
                    "title": f"{file_path}#{chunk.chunk_index}",
                }
            )
            chunked.append(Document(text=chunk.text, meta_data=chunk_metadata))
    return chunked


def _chunk_text(text: str, file_path: str, file_type: str) -> List[Chunk]:
    if file_type in {"json"}:
        return [_make_chunk(text, file_path, 0, 1, _line_count(text))]

    if file_type in {"ts", "tsx", "js", "jsx", "py", "java", "go", "rs", "cs", "php", "swift"}:
        structural = _chunk_by_symbols(text, file_path)
        if structural:
            return structural

    if file_type in {"css", "scss", "wxss"}:
        structural = _chunk_by_css_blocks(text, file_path)
        if structural:
            return structural

    if file_type in {"wxml", "html"} and _approx_tokens(text) <= 512:
        return [_make_chunk(text, file_path, 0, 1, _line_count(text))]

    return _chunk_by_lines(text, file_path)


def _chunk_by_symbols(text: str, file_path: str) -> List[Chunk]:
    pattern = re.compile(
        r"^\s*(export\s+)?(async\s+)?(function|class|interface|type|const|let|var|def|public|private|protected)\s+[\w$]+",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    if len(matches) <= 1:
        return []
    return _chunks_from_offsets(text, file_path, [match.start() for match in matches])


def _chunk_by_css_blocks(text: str, file_path: str) -> List[Chunk]:
    offsets = [match.start() for match in re.finditer(r"^[^@\n][^{\n]+{", text, re.MULTILINE)]
    if len(offsets) <= 1:
        return []
    return _chunks_from_offsets(text, file_path, offsets)


def _chunks_from_offsets(text: str, file_path: str, offsets: List[int]) -> List[Chunk]:
    chunks = []
    offsets = sorted(set(offsets))
    for index, start in enumerate(offsets):
        end = offsets[index + 1] if index + 1 < len(offsets) else len(text)
        chunk_text = text[start:end].strip()
        if not chunk_text:
            continue
        start_line = text.count("\n", 0, start) + 1
        end_line = start_line + chunk_text.count("\n")
        chunks.extend(_split_large_chunk(chunk_text, file_path, len(chunks), start_line, end_line))
    return chunks


def _split_large_chunk(text: str, file_path: str, start_index: int, start_line: int, end_line: int) -> List[Chunk]:
    if _approx_tokens(text) <= 512:
        return [_make_chunk(text, file_path, start_index, start_line, end_line)]

    lines = text.splitlines()
    header = lines[0] if lines else ""
    chunks = []
    current = []
    current_start = start_line
    index = start_index
    for offset, line in enumerate(lines):
        candidate = "\n".join(current + [line])
        if current and _approx_tokens(candidate) > 512:
            body = "\n".join(current)
            if header and not body.startswith(header):
                body = f"{header}\n{body}"
            chunks.append(_make_chunk(body, file_path, index, current_start, start_line + offset - 1))
            index += 1
            current = [line]
            current_start = start_line + offset
        else:
            current.append(line)
    if current:
        body = "\n".join(current)
        if header and not body.startswith(header):
            body = f"{header}\n{body}"
        chunks.append(_make_chunk(body, file_path, index, current_start, end_line))
    return chunks


def _chunk_by_lines(text: str, file_path: str, max_tokens: int = 512, overlap_lines: int = 4) -> List[Chunk]:
    lines = text.splitlines()
    if not lines:
        return [_make_chunk("", file_path, 0, 1, 1)]

    chunks = []
    index = 0
    start = 0
    while start < len(lines):
        end = start
        current = []
        while end < len(lines) and (not current or _approx_tokens("\n".join(current + [lines[end]])) <= max_tokens):
            current.append(lines[end])
            end += 1
        chunks.append(_make_chunk("\n".join(current), file_path, index, start + 1, end))
        index += 1
        if end >= len(lines):
            break
        start = max(end - overlap_lines, start + 1)
    return chunks


def _make_chunk(text: str, file_path: str, chunk_index: int, start_line: int, end_line: int) -> Chunk:
    return Chunk(
        text=text,
        file_path=file_path,
        chunk_index=chunk_index,
        content_hash=sha256_text(text, length=64),
        start_line=start_line,
        end_line=end_line,
    )


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _line_count(text: str) -> int:
    return max(1, text.count("\n") + 1)
