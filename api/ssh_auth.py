"""SSH authentication for private Git repositories.

Supports:
- SSH URL format: git@host:owner/repo or ssh://git@host/owner/repo
- Deploy key via GIT_SSH_KEY env var or ~/.ssh/id_rsa
- Multiple Git hosts via ~/.ssh/config
- Known hosts via GIT_SSH_KNOWN_HOSTS or ssh-keyscan
- Connectivity verification before cloning
"""

import logging
import os
import re
import subprocess
import stat
from dataclasses import dataclass
from typing import Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Environment variable configuration
ENV_SSH_KEY = "GIT_SSH_KEY"              # Path to private key file
ENV_SSH_KNOWN_HOSTS = "GIT_SSH_KNOWN_HOSTS"  # Path to known_hosts file
ENV_SSH_STRICT = "GIT_SSH_STRICT_HOST_KEY_CHECKING"  # accept-new | yes | no
ENV_SSH_COMMAND = "GIT_SSH_COMMAND"      # Override entire ssh command


@dataclass
class SSHHost:
    """Parsed SSH host information from a Git URL."""
    hostname: str
    user: str = "git"


def is_ssh_url(url: str) -> bool:
    """Detect whether a Git URL uses SSH protocol.

    Returns True for:
        git@github.com:owner/repo.git
        ssh://git@github.com/owner/repo.git
        git@gitlab.internal:group/subgroup/repo.git
    """
    if not url:
        return False
    # SCP-like: git@host:path
    if re.match(r"^[\w.-]+@[\w.-]+:.+", url):
        return True
    # SSH protocol: ssh://[user@]host[:port]/path
    if url.startswith("ssh://"):
        return True
    return False


def parse_ssh_url(url: str) -> SSHHost:
    """Extract host information from an SSH Git URL.

    Args:
        url: Git SSH URL (git@host:path or ssh://git@host/path)

    Returns:
        SSHHost with hostname and user.

    Raises:
        ValueError: If the URL cannot be parsed.
    """
    # SCP-like format: git@github.com:owner/repo.git
    match = re.match(r"^([\w.-]+)@([\w.-]+):.+", url)
    if match:
        return SSHHost(user=match.group(1), hostname=match.group(2))

    # SSH protocol: ssh://git@github.com/owner/repo.git
    if url.startswith("ssh://"):
        parsed = urlparse(url)
        user = parsed.username or "git"
        hostname = parsed.hostname or ""
        if not hostname:
            raise ValueError(f"Cannot parse host from SSH URL: {url}")
        return SSHHost(user=user, hostname=hostname)

    raise ValueError(f"Not a valid SSH URL: {url}")


def find_ssh_key() -> Optional[str]:
    """Find the SSH private key to use.

    Checks in order:
    1. GIT_SSH_KEY environment variable
    2. ~/.ssh/id_ed25519
    3. ~/.ssh/id_rsa
    4. ~/.ssh/id_ecdsa

    Returns path to private key, or None if not found.
    """
    env_key = os.environ.get(ENV_SSH_KEY)
    if env_key:
        expanded = os.path.expanduser(env_key)
        if os.path.isfile(expanded):
            return expanded
        logger.warning(f"GIT_SSH_KEY is set but file not found: {expanded}")

    ssh_dir = os.path.expanduser("~/.ssh")
    if not os.path.isdir(ssh_dir):
        return None

    for name in ("id_ed25519", "id_rsa", "id_ecdsa"):
        candidate = os.path.join(ssh_dir, name)
        if os.path.isfile(candidate):
            return candidate

    return None


def validate_key_permissions(key_path: str) -> Tuple[bool, str]:
    """Check that the private key has secure permissions (600).

    Returns:
        Tuple of (is_valid, message).
    """
    try:
        st = os.stat(key_path)
        mode = st.st_mode & 0o777
        # Key file should be readable only by owner
        if mode & 0o077:  # Any group or other permissions set
            return False, (
                f"SSH private key {key_path} has insecure permissions ({oct(mode)}). "
                f"Run: chmod 600 {key_path}"
            )
        return True, f"Key permissions OK ({oct(mode)})"
    except OSError as e:
        return False, f"Cannot stat key file {key_path}: {e}"


def ensure_known_hosts(hostname: str, known_hosts_path: Optional[str] = None) -> str:
    """Ensure the host key is in known_hosts, scanning if needed.

    Args:
        hostname: The SSH server hostname.
        known_hosts_path: Optional path to known_hosts file.

    Returns:
        Path to the known_hosts file to use.
    """
    if known_hosts_path is None:
        known_hosts_path = os.environ.get(ENV_SSH_KNOWN_HOSTS)
        if known_hosts_path:
            known_hosts_path = os.path.expanduser(known_hosts_path)
        else:
            known_hosts_path = os.path.expanduser("~/.ssh/known_hosts")

    # Ensure parent directory exists
    parent = os.path.dirname(known_hosts_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # Check if host is already in known_hosts
    host_known = False
    if os.path.isfile(known_hosts_path):
        try:
            with open(known_hosts_path, "r") as f:
                for line in f:
                    if line.strip().startswith(hostname) or line.strip().split()[0] == hostname:
                        host_known = True
                        break
        except (OSError, PermissionError):
            pass

    if not host_known:
        logger.info(f"Scanning host key for {hostname}")
        try:
            result = subprocess.run(
                ["ssh-keyscan", "-H", hostname],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0 and result.stdout.strip():
                with open(known_hosts_path, "a") as f:
                    f.write(result.stdout)
                logger.info(f"Added host key for {hostname} to {known_hosts_path}")
                # Secure the known_hosts file
                os.chmod(known_hosts_path, 0o644)
            else:
                logger.warning(
                    f"ssh-keyscan for {hostname} returned nothing. "
                    f"The host key will need to be trusted via StrictHostKeyChecking."
                )
        except FileNotFoundError:
            logger.warning("ssh-keyscan not available; cannot auto-populate known_hosts")
        except Exception as e:
            logger.warning(f"Failed to scan host key for {hostname}: {e}")

    return known_hosts_path


def build_ssh_command(
    key_path: Optional[str] = None,
    known_hosts_path: Optional[str] = None,
    strict_host_key_checking: Optional[str] = None,
) -> str:
    """Build the GIT_SSH_COMMAND string.

    Args:
        key_path: Path to SSH private key.
        known_hosts_path: Path to known_hosts file.
        strict_host_key_checking: 'yes', 'accept-new', or 'no'.

    Returns:
        SSH command string suitable for GIT_SSH_COMMAND env var.
    """
    parts = ["ssh"]

    if key_path:
        parts.append(f"-i {key_path}")

    # StrictHostKeyChecking: prefer explicit, then env var, then default to accept-new
    if strict_host_key_checking is None:
        strict_host_key_checking = os.environ.get(ENV_SSH_STRICT, "accept-new")
    parts.append(f"-o StrictHostKeyChecking={strict_host_key_checking}")

    if known_hosts_path:
        parts.append(f"-o UserKnownHostsFile={known_hosts_path}")

    # Batch mode: never prompt for passwords
    parts.append("-o BatchMode=yes")

    # Connection timeout
    parts.append("-o ConnectTimeout=10")

    return " ".join(parts)


def verify_ssh_connection(hostname: str, key_path: Optional[str] = None) -> Tuple[bool, str]:
    """Verify SSH connectivity to a Git host before cloning.

    Runs: ssh -T git@hostname

    Args:
        hostname: The Git server hostname.
        key_path: Optional path to SSH private key.

    Returns:
        Tuple of (success, message).
    """
    cmd = ["ssh"]
    if key_path:
        cmd.extend(["-i", key_path])
    cmd.extend([
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        "-T",
        f"git@{hostname}",
    ])

    logger.info(f"Verifying SSH connection to git@{hostname}...")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15,
        )
        # SSH to git hosts typically exits with code 1 even on success
        # (they don't provide shell access, just say "Hi {user}!")
        # Exit code 255 means connection failure
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()

        if result.returncode == 255:
            return False, f"SSH connection to {hostname} failed: {stderr}"

        # Any other exit code (including 1) with a message means success
        message = stderr or stdout or "Connection OK"
        logger.info(f"SSH connection to {hostname} successful: {message[:100]}")
        return True, message

    except subprocess.TimeoutExpired:
        return False, f"SSH connection to {hostname} timed out after 15s"
    except FileNotFoundError:
        return False, "ssh command not found. Is OpenSSH installed?"
    except Exception as e:
        return False, f"SSH verification error for {hostname}: {e}"


def prepare_ssh_for_repo(url: str) -> Optional[str]:
    """Full SSH preparation for a repository URL.

    - Parses the host from the URL
    - Finds and validates the SSH key
    - Ensures known_hosts
    - Verifies SSH connectivity
    - Returns the GIT_SSH_COMMAND string, or None if not an SSH URL

    Raises:
        ValueError: If SSH preparation fails.
    """
    if not is_ssh_url(url):
        return None

    # Check for explicit GIT_SSH_COMMAND override
    explicit_command = os.environ.get(ENV_SSH_COMMAND)
    if explicit_command:
        logger.info("Using explicit GIT_SSH_COMMAND from environment")
        host = parse_ssh_url(url)
        ok, msg = verify_ssh_connection(host.hostname)
        if not ok:
            raise ValueError(f"SSH verification failed: {msg}")
        return explicit_command

    host = parse_ssh_url(url)
    logger.info(f"Preparing SSH for {host.user}@{host.hostname}")

    # Find SSH key
    key_path = find_ssh_key()
    if not key_path:
        raise ValueError(
            f"No SSH key found. Set GIT_SSH_KEY env var or place a key in ~/.ssh/. "
            f"Cannot clone SSH URL: {url}"
        )

    # Validate key permissions
    ok, msg = validate_key_permissions(key_path)
    if not ok:
        # Try to fix permissions automatically
        logger.warning(msg)
        try:
            os.chmod(key_path, 0o600)
            logger.info(f"Fixed permissions on {key_path} to 600")
        except OSError:
            raise ValueError(f"{msg} Please run: chmod 600 {key_path}")

    # Ensure known_hosts
    known_hosts = ensure_known_hosts(host.hostname)

    # Verify SSH connection
    ok, msg = verify_ssh_connection(host.hostname, key_path)
    if not ok:
        raise ValueError(
            f"SSH connection to {host.hostname} failed: {msg}. "
            f"Verify your SSH key is authorized and the host is reachable."
        )

    # Build GIT_SSH_COMMAND
    ssh_command = build_ssh_command(
        key_path=key_path,
        known_hosts_path=known_hosts,
    )
    logger.info(f"SSH preparation complete for {host.hostname}")
    return ssh_command


def git_env_for_repo(repo_path: str) -> dict:
    """Return environment dict with GIT_SSH_COMMAND set if the repo uses SSH remotes.

    Call this before any git operation (pull, diff, fetch, etc.) on a repo
    that may have been cloned via SSH.

    Args:
        repo_path: Path to the local repository.

    Returns:
        Environment dict suitable for subprocess env= parameter.
        If the repo doesn't use SSH, returns os.environ.copy() unchanged.
    """
    env = os.environ.copy()

    if not os.path.isdir(os.path.join(repo_path, ".git")):
        return env

    # Read the remote URL from git config
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return env
        remote_url = result.stdout.strip()
    except Exception:
        return env

    if not is_ssh_url(remote_url):
        return env

    # Check for explicit GIT_SSH_COMMAND (set during clone or by user)
    if ENV_SSH_COMMAND in os.environ:
        return env

    # Re-derive SSH command from key
    key_path = find_ssh_key()
    if not key_path:
        logger.warning(f"SSH key not found; git operations on {repo_path} may fail")
        return env

    ok, _ = validate_key_permissions(key_path)
    if not ok:
        try:
            os.chmod(key_path, 0o600)
        except OSError:
            pass

    host = parse_ssh_url(remote_url)
    known_hosts = ensure_known_hosts(host.hostname)
    ssh_command = build_ssh_command(key_path=key_path, known_hosts_path=known_hosts)
    env[ENV_SSH_COMMAND] = ssh_command
    logger.info(f"Prepared SSH env for repo at {repo_path} (remote: {host.hostname})")
    return env
