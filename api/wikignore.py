""".wikignore file parser — gitignore-compatible syntax for excluding files from indexing.

Read from repo root, with built-in filters for binary files and common artifacts.
"""

import fnmatch
import logging
import mimetypes
import os
from typing import Callable, List, Optional, Set

logger = logging.getLogger(__name__)

# Built-in excluded patterns — always applied regardless of .wikignore
BUILTIN_IGNORE_PATTERNS: List[str] = [
    "node_modules/",
    "dist/",
    "vendor/",
    ".git/",
    ".svn/",
    ".hg/",
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    "build/",
    "target/",
    "out/",
    ".idea/",
    ".vscode/",
    ".DS_Store",
    "Thumbs.db",
    "*.pyc",
    "*.pyo",
    "*.class",
    "*.o",
    "*.obj",
    "*.so",
    "*.dylib",
    "*.dll",
    "*.exe",
    "*.jar",
    "*.war",
    "*.zip",
    "*.tar",
    "*.gz",
    "*.tgz",
    "*.7z",
    "*.rar",
    "*.iso",
    "*.dmg",
    "*.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Cargo.lock",
    "Gemfile.lock",
    "*.map",
    "*.min.js",
    "*.min.css",
    "*.bundle.js",
    "coverage/",
    "htmlcov/",
    ".nyc_output/",
    ".tox/",
    ".venv/",
    "venv/",
    "env/",
    ".env",
]

# Files that are always safe to include (override built-in ignores)
BUILTIN_INCLUDE_PATTERNS: List[str] = [
    "pyproject.toml",
    "Cargo.toml",
    "go.mod",
    "go.sum",
    "package.json",
    "Makefile",
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    ".gitignore",
    ".dockerignore",
    "README.md",
    "LICENSE",
]

# Binary MIME types to exclude
BINARY_MIME_PREFIXES: Set[str] = {
    "image/",
    "audio/",
    "video/",
    "application/zip",
    "application/gzip",
    "application/x-tar",
    "application/x-gzip",
    "application/x-bzip2",
    "application/x-7z-compressed",
    "application/x-rar-compressed",
    "application/x-iso9660-image",
    "application/x-shockwave-flash",
    "application/vnd.microsoft.portable-executable",
    "application/x-mach-binary",
    "application/x-msdownload",
    "application/java-archive",
    "application/octet-stream",
    "application/x-executable",
    "application/x-sharedlib",
    "application/x-font-",
    "application/vnd.android.package-archive",
    "application/x-apple-diskimage",
    "font/",
    "model/",
}


def _has_null_bytes(path: str) -> bool:
    """Check if a file contains null bytes (strong indicator of binary content)."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
            return b"\x00" in chunk
    except (OSError, PermissionError):
        # If we can't read it, treat it as binary / skip
        return True


def is_binary_file(path: str) -> bool:
    """Detect binary files using MIME type and null-byte heuristics.

    Returns True if the file should be excluded as binary.
    """
    # Quick null-byte check first
    if _has_null_bytes(path):
        return True

    # MIME type check
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type:
        for prefix in BINARY_MIME_PREFIXES:
            if mime_type.startswith(prefix) or mime_type == prefix:
                return True

    return False


def parse_wikignore(repo_root: str) -> Callable[[str], bool]:
    """Parse .wikignore from repo root and return a predicate.

    The returned function takes a file path (relative to repo root)
    and returns True if the file should be EXCLUDED.

    Supports gitignore-compatible syntax:
    - # comments
    - * and ? wildcards (fnmatch)
    - directory patterns (trailing /)
    - ! negation
    """
    wikignore_path = os.path.join(repo_root, ".wikignore")
    patterns: List[tuple] = []  # List of (negate, pattern)

    # Parse .wikignore if it exists
    if os.path.isfile(wikignore_path):
        try:
            with open(wikignore_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith("#"):
                        continue

                    negate = False
                    if line.startswith("!"):
                        negate = True
                        line = line[1:]

                    # Remove leading / (root-relative is default)
                    if line.startswith("/"):
                        line = line[1:]

                    patterns.append((negate, line))
        except (OSError, PermissionError) as e:
            logger.warning(f"Could not read .wikignore at {wikignore_path}: {e}")

    # Build built-in patterns (negative = exclude)
    for pat in BUILTIN_IGNORE_PATTERNS:
        patterns.append((False, pat))

    for pat in BUILTIN_INCLUDE_PATTERNS:
        patterns.append((True, pat))

    def should_exclude(rel_path: str) -> bool:
        """Check if a relative file path should be excluded.

        Last matching pattern wins (gitignore semantics).
        """
        excluded = False
        for negate, pattern in patterns:
            # Check if the path matches this pattern
            if _match_pattern(rel_path, pattern):
                excluded = not negate  # True negates (include), False excludes
        return excluded

    return should_exclude


def _match_pattern(rel_path: str, pattern: str) -> bool:
    """Check if a relative file path matches a gitignore-style pattern."""
    # Directory pattern — match if the directory name appears at any level
    if pattern.endswith("/"):
        dirname = pattern[:-1]
        parts = rel_path.replace(os.sep, "/").split("/")
        dir_parts = dirname.split("/")
        # Check if dir_parts is a contiguous subsequence of parts
        for i in range(len(parts) - len(dir_parts) + 1):
            if all(fnmatch.fnmatch(parts[i + j], dp) for j, dp in enumerate(dir_parts)):
                return True
        return False

    # ** pattern — match any number of directory levels
    if "**" in pattern:
        # Simple ** support: match **/suffix
        if pattern.startswith("**/"):
            suffix = pattern[3:]
            parts = rel_path.replace(os.sep, "/").split("/")
            # Check if any suffix matches
            for i in range(len(parts)):
                candidate = "/".join(parts[i:])
                if fnmatch.fnmatch(candidate, suffix):
                    return True
            return False
        return fnmatch.fnmatch(rel_path.replace(os.sep, "/"), pattern)

    # Simple fnmatch — matches in any directory
    filename = os.path.basename(rel_path)
    if fnmatch.fnmatch(filename, pattern):
        return True

    # Also try matching full relative path (for patterns with /)
    if "/" in pattern:
        return fnmatch.fnmatch(rel_path.replace(os.sep, "/"), pattern)

    return False


def filter_files(
    root: str,
    candidates: List[str],
    wikignore_predicate: Optional[Callable[[str], bool]] = None,
) -> List[str]:
    """Filter a list of file paths, removing binary files and wikignore matches.

    Args:
        root: Repository root directory (for binary detection).
        candidates: List of absolute file paths.
        wikignore_predicate: Optional predicate from parse_wikignore.

    Returns:
        Filtered list of absolute file paths.
    """
    result = []
    for file_path in candidates:
        rel_path = os.path.relpath(file_path, root)

        # Check built-in and .wikignore patterns
        if wikignore_predicate and wikignore_predicate(rel_path):
            logger.debug(f"Ignoring {rel_path}: matched wikignore pattern")
            continue

        # Check binary
        if is_binary_file(file_path):
            logger.debug(f"Ignoring {rel_path}: detected as binary")
            continue

        result.append(file_path)

    return result
