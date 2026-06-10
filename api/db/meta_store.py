import contextlib
import os
import sqlite3
from datetime import datetime
from typing import Dict, Iterable, Optional


SCHEMA_VERSION = 1


class MetaStore:
    """SQLite metadata store for repo, job, file, chunk, and wiki dependency state."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_schema()

    @contextlib.contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS repos (
                    id TEXT PRIMARY KEY,
                    clone_path TEXT NOT NULL,
                    remote_url TEXT,
                    last_indexed_sha TEXT,
                    embedding_base_url TEXT,
                    embedding_model TEXT,
                    embedding_dim INTEGER,
                    embedding_normalize INTEGER,
                    chunker_version TEXT,
                    index_schema_version INTEGER,
                    indexed_at DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS wiki_pages (
                    id TEXT PRIMARY KEY,
                    repo_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    page_order INTEGER,
                    commit_sha TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS page_file_deps (
                    page_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    PRIMARY KEY (page_id, file_path)
                );

                CREATE TABLE IF NOT EXISTS page_chunk_deps (
                    page_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    retrieval_score REAL,
                    PRIMARY KEY (page_id, chunk_id)
                );

                CREATE TABLE IF NOT EXISTS index_jobs (
                    id TEXT PRIMARY KEY,
                    repo_id TEXT NOT NULL,
                    from_sha TEXT,
                    to_sha TEXT,
                    status TEXT NOT NULL,
                    error TEXT,
                    started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    finished_at DATETIME
                );

                CREATE TABLE IF NOT EXISTS file_index (
                    repo_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    indexed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (repo_id, file_path)
                );

                CREATE TABLE IF NOT EXISTS chunk_index (
                    repo_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    chunk_id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    start_line INTEGER,
                    end_line INTEGER
                );
                """
            )

    def get_repo(self, repo_id: str) -> Optional[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM repos WHERE id = ?", (repo_id,)).fetchone()

    def upsert_repo(self, repo_id: str, clone_path: str, remote_url: Optional[str], fingerprint: Dict) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO repos (
                    id, clone_path, remote_url, embedding_base_url, embedding_model,
                    embedding_dim, embedding_normalize, chunker_version, index_schema_version
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    clone_path=excluded.clone_path,
                    remote_url=excluded.remote_url,
                    embedding_base_url=excluded.embedding_base_url,
                    embedding_model=excluded.embedding_model,
                    embedding_dim=excluded.embedding_dim,
                    embedding_normalize=excluded.embedding_normalize,
                    chunker_version=excluded.chunker_version,
                    index_schema_version=excluded.index_schema_version
                """,
                (
                    repo_id,
                    clone_path,
                    remote_url,
                    fingerprint.get("embedding_base_url"),
                    fingerprint.get("embedding_model"),
                    fingerprint.get("embedding_dim"),
                    int(bool(fingerprint.get("embedding_normalize"))),
                    fingerprint.get("chunker_version"),
                    fingerprint.get("index_schema_version", SCHEMA_VERSION),
                ),
            )

    def assert_compatible(self, repo_id: str, fingerprint: Dict) -> None:
        repo = self.get_repo(repo_id)
        if not repo:
            return

        checks = {
            "embedding_base_url": fingerprint.get("embedding_base_url"),
            "embedding_model": fingerprint.get("embedding_model"),
            "embedding_dim": fingerprint.get("embedding_dim"),
            "embedding_normalize": int(bool(fingerprint.get("embedding_normalize"))),
            "chunker_version": fingerprint.get("chunker_version"),
            "index_schema_version": fingerprint.get("index_schema_version", SCHEMA_VERSION),
        }
        mismatches = [
            key for key, expected in checks.items()
            if repo[key] is not None and repo[key] != expected
        ]
        if mismatches:
            raise ValueError(
                "Embedding/index fingerprint changed; rebuild the repository index before continuing. "
                f"Mismatched fields: {', '.join(mismatches)}"
            )

    def create_job(self, job_id: str, repo_id: str, from_sha: Optional[str], to_sha: Optional[str]) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO index_jobs (id, repo_id, from_sha, to_sha, status) VALUES (?, ?, ?, ?, 'running')",
                (job_id, repo_id, from_sha, to_sha),
            )

    def finish_job(self, job_id: str, status: str, error: Optional[str] = None) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE index_jobs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
                (status, error, datetime.utcnow().isoformat(), job_id),
            )

    def has_running_job(self, repo_id: str) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT id FROM index_jobs WHERE repo_id = ? AND status = 'running' LIMIT 1",
                (repo_id,),
            ).fetchone()
            return row is not None

    def update_repo_sha(self, repo_id: str, sha: Optional[str]) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE repos SET last_indexed_sha = ?, indexed_at = ? WHERE id = ?",
                (sha, datetime.utcnow().isoformat(), repo_id),
            )

    def replace_file_chunks(self, repo_id: str, file_path: str, file_hash: str, chunks: Iterable[Dict]) -> None:
        chunks = list(chunks)
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM chunk_index WHERE repo_id = ? AND file_path = ?",
                (repo_id, file_path),
            )
            conn.execute(
                """
                INSERT INTO file_index (repo_id, file_path, file_hash, chunk_count, indexed_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(repo_id, file_path) DO UPDATE SET
                    file_hash=excluded.file_hash,
                    chunk_count=excluded.chunk_count,
                    indexed_at=excluded.indexed_at
                """,
                (repo_id, file_path, file_hash, len(chunks), datetime.utcnow().isoformat()),
            )
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunk_index
                    (repo_id, file_path, chunk_id, content_hash, start_line, end_line)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        repo_id,
                        file_path,
                        chunk["chunk_id"],
                        chunk["content_hash"],
                        chunk.get("start_line"),
                        chunk.get("end_line"),
                    )
                    for chunk in chunks
                ],
            )

    def delete_file(self, repo_id: str, file_path: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM chunk_index WHERE repo_id = ? AND file_path = ?", (repo_id, file_path))
            conn.execute("DELETE FROM file_index WHERE repo_id = ? AND file_path = ?", (repo_id, file_path))

    def replace_wiki_pages(self, repo_id: str, pages: Iterable[Dict], commit_sha: str) -> None:
        pages = list(pages)
        with self.connect() as conn:
            page_ids = [page["id"] for page in pages]
            if page_ids:
                placeholders = ",".join("?" for _ in page_ids)
                conn.execute(f"DELETE FROM page_file_deps WHERE page_id IN ({placeholders})", page_ids)
                conn.execute(f"DELETE FROM page_chunk_deps WHERE page_id IN ({placeholders})", page_ids)

            conn.executemany(
                """
                INSERT INTO wiki_pages
                    (id, repo_id, title, content, page_order, commit_sha, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    repo_id=excluded.repo_id,
                    title=excluded.title,
                    content=excluded.content,
                    page_order=excluded.page_order,
                    commit_sha=excluded.commit_sha,
                    updated_at=excluded.updated_at
                """,
                [
                    (
                        page["id"],
                        repo_id,
                        page.get("title", ""),
                        page.get("content", ""),
                        page.get("page_order", index),
                        commit_sha,
                        datetime.utcnow().isoformat(),
                    )
                    for index, page in enumerate(pages)
                ],
            )

            deps = []
            for page in pages:
                for file_path in page.get("file_paths", []) or []:
                    deps.append((page["id"], file_path))
            if deps:
                conn.executemany(
                    "INSERT OR IGNORE INTO page_file_deps (page_id, file_path) VALUES (?, ?)",
                    deps,
                )
