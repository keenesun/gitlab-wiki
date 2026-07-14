import hashlib
import os
import subprocess
from typing import Iterable, List, Optional, Tuple
from uuid import uuid4

from api.types import Document

from api.db.chroma_store import ChromaStore
from api.db.meta_store import MetaStore


def _git_env(repo_path: str) -> dict:
    """Return environment dict for git operations, with SSH command if needed."""
    from api.ssh_auth import git_env_for_repo
    return git_env_for_repo(repo_path)


def git_head(repo_path: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_git_env(repo_path),
        )
        return result.stdout.strip()
    except Exception:
        return None


def git_pull(repo_path: str) -> None:
    subprocess.run(
        ["git", "-C", repo_path, "pull", "--ff-only"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_git_env(repo_path),
    )


def git_commit_exists(repo_path: str, sha: str) -> bool:
    try:
        subprocess.run(
            ["git", "-C", repo_path, "cat-file", "-e", f"{sha}^{{commit}}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_git_env(repo_path),
        )
        return True
    except Exception:
        return False


def git_diff_name_status(repo_path: str, from_sha: str, to_sha: str) -> List[Tuple[str, str, Optional[str]]]:
    result = subprocess.run(
        ["git", "-C", repo_path, "diff", "--name-status", f"{from_sha}..{to_sha}"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_git_env(repo_path),
    )
    changes = []
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if not parts:
            continue
        status = parts[0]
        if status.startswith("R") and len(parts) >= 3:
            changes.append(("R", parts[1], parts[2]))
        elif len(parts) >= 2:
            changes.append((status[0], parts[1], None))
    return changes


def file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sync_index_state(
    meta_store: MetaStore,
    chroma_store: ChromaStore,
    repo_id: str,
    repo_path: str,
    transformed_docs: Iterable[Document],
    pull_remote: bool = False,
) -> None:
    """Record index state after transformed docs are produced by the current pipeline."""
    if meta_store.has_running_job(repo_id):
        raise RuntimeError(f"Repository {repo_id} already has a running index job")

    repo = meta_store.get_repo(repo_id)
    from_sha = repo["last_indexed_sha"] if repo else None

    if pull_remote and os.path.isdir(os.path.join(repo_path, ".git")):
        git_pull(repo_path)

    to_sha = git_head(repo_path)
    job_id = str(uuid4())
    meta_store.create_job(job_id, repo_id, from_sha, to_sha)

    try:
        docs_by_file = {}
        for doc in transformed_docs:
            metadata = getattr(doc, "meta_data", {}) or {}
            file_path = metadata.get("file_path", "unknown")
            docs_by_file.setdefault(file_path, []).append(doc)

        if from_sha and to_sha and from_sha != to_sha and git_commit_exists(repo_path, from_sha) and git_commit_exists(repo_path, to_sha):
            for status, old_path, new_path in git_diff_name_status(repo_path, from_sha, to_sha):
                if status == "D":
                    chroma_store.delete_by_file(repo_id, old_path)
                    meta_store.delete_file(repo_id, old_path)
                elif status == "R":
                    chroma_store.delete_by_file(repo_id, old_path)
                    meta_store.delete_file(repo_id, old_path)
                    chroma_store.delete_by_file(repo_id, new_path)
                    meta_store.delete_file(repo_id, new_path)
                elif status in {"A", "M"}:
                    chroma_store.delete_by_file(repo_id, old_path)
                    meta_store.delete_file(repo_id, old_path)

        chroma_store.upsert_documents(doc for docs in docs_by_file.values() for doc in docs)

        for file_path, docs in docs_by_file.items():
            absolute_path = os.path.join(repo_path, file_path)
            digest = file_hash(absolute_path) if os.path.exists(absolute_path) else ""
            meta_store.replace_file_chunks(
                repo_id,
                file_path,
                digest,
                [getattr(doc, "meta_data", {}) for doc in docs],
            )

        if to_sha:
            meta_store.update_repo_sha(repo_id, to_sha)
        meta_store.finish_job(job_id, "completed")
    except Exception as exc:
        meta_store.finish_job(job_id, "failed", str(exc))
        raise
