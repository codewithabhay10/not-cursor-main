"""Central configuration, loaded once from the environment (.env).

Keeping all tunables here means no values are hardcoded inside the
application logic, which makes the app portable across machines.
"""
import os

from dotenv import load_dotenv

load_dotenv()


# --- LLM (Ollama, runs locally) -------------------------------------------
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))  # seconds

# --- Target repository the agent operates on ------------------------------
# Defaults to the current working directory so the project works out of the
# box without anyone editing source code.
TARGET_REPO_PATH = os.getenv("TARGET_REPO_PATH", os.getcwd())

# "owner/name" used only to build convenient GitHub links in the UI.
GITHUB_REPO = os.getenv("GITHUB_REPO", "")

# Safety: never push to a remote unless explicitly opted in. A local commit
# on a throwaway branch is always reversible; an automatic push is not.
AUTO_PUSH = os.getenv("AUTO_PUSH", "false").lower() in ("1", "true", "yes")

# --- Agent behaviour ------------------------------------------------------
# Only these file types are considered editable by the agent.
EDITABLE_EXTENSIONS = (".py", ".js", ".ts", ".jsx", ".tsx")

# How many recent commits to feed the model as context.
N_COMMITS_CONTEXT = int(os.getenv("N_COMMITS_CONTEXT", "5"))

# Each rewritten file is one LangGraph superstep; raise the recursion limit
# above the default (25) so plans with many files don't get cut off.
GRAPH_RECURSION_LIMIT = int(os.getenv("GRAPH_RECURSION_LIMIT", "100"))


def is_safe_path(repo_path: str, target_rel_path: str) -> bool:
    """Return True only if ``target_rel_path`` stays inside ``repo_path``.

    The model picks file paths, so we must defend against path traversal
    (e.g. ``../../etc/passwd``) before ever writing to disk.
    """
    repo_root = os.path.abspath(repo_path)
    resolved = os.path.abspath(os.path.join(repo_root, target_rel_path))
    return os.path.commonpath([repo_root, resolved]) == repo_root
