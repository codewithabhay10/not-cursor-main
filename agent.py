"""The coding agent, implemented as a LangGraph state machine.

Graph shape:

    START -> load_context -> plan -> rewrite --(more files?)--> rewrite
                                       |
                                       (done) -> commit -> END

Each node receives the shared ``State`` and returns a *partial* update that
LangGraph merges back in. The ``rewrite`` node loops over itself via a
conditional edge until every planned file has been processed.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from typing import Callable, Optional

from git import Repo
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

import config


class State(TypedDict, total=False):
    """Shared state threaded through every node of the graph."""

    prompt: str
    repo_path: str
    files: dict[str, str]
    commits: list[str]
    file_list: list[str]
    plan: list[dict]
    current_idx: int
    modified_files: list[str]
    committed_branch: str


def build_llm() -> ChatOllama:
    """Factory for the local Ollama chat model, configured from .env."""
    return ChatOllama(
        model=config.OLLAMA_MODEL,
        temperature=config.OLLAMA_TEMPERATURE,
        num_ctx=config.OLLAMA_NUM_CTX,
        timeout=config.OLLAMA_TIMEOUT,
    )


def _extract_json_array(text: str) -> list[dict]:
    """Pull the first ``[ {...} ]`` JSON array out of a model response."""
    match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON array found in model response")
    return json.loads(match.group(0))


class CodingAgent:
    """Compiles and runs the LangGraph workflow for a single request.

    ``emit`` is a callback used to stream human-readable progress to the UI;
    it replaces the previous (thread-unsafe) ``builtins.print`` monkeypatch.
    """

    def __init__(
        self,
        llm: ChatOllama,
        repo_path: str,
        emit: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.llm = llm
        self.repo_path = repo_path
        self.emit = emit or (lambda _msg: None)
        self.graph = self._build_graph()

    # --- graph wiring ------------------------------------------------------
    def _build_graph(self):
        g = StateGraph(State)
        g.add_node("load_context", self.load_context)
        g.add_node("plan", self.plan)
        g.add_node("rewrite", self.rewrite)
        g.add_node("commit", self.commit)

        g.add_edge(START, "load_context")
        g.add_edge("load_context", "plan")
        g.add_edge("plan", "rewrite")
        g.add_conditional_edges(
            "rewrite",
            self._should_continue,
            {"rewrite": "rewrite", "commit": "commit"},
        )
        g.add_edge("commit", END)
        return g.compile()

    def run(self, prompt: str) -> State:
        initial: State = {
            "prompt": prompt,
            "repo_path": self.repo_path,
            "files": {},
            "commits": [],
            "file_list": [],
            "plan": [],
            "current_idx": 0,
            "modified_files": [],
        }
        return self.graph.invoke(
            initial,
            config={"recursion_limit": config.GRAPH_RECURSION_LIMIT},
        )

    # --- nodes -------------------------------------------------------------
    def load_context(self, state: State) -> State:
        path = state["repo_path"]
        repo = Repo(path)
        files: dict[str, str] = {}
        for rel in repo.git.ls_files().split("\n"):
            if not rel:
                continue
            abs_path = os.path.join(repo.working_tree_dir, rel)
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    files[rel] = f.read()
            except (UnicodeDecodeError, FileNotFoundError):
                continue  # skip binaries / deleted files

        commits = [
            f"{c.hexsha[:7]}: {c.message.strip()}"
            for c in repo.iter_commits("HEAD", max_count=config.N_COMMITS_CONTEXT)
        ]
        self.emit(f"📁 Loaded {len(files)} files, {len(commits)} recent commits")
        return {
            "files": files,
            "commits": commits,
            "file_list": list(files.keys()),
            "current_idx": 0,
        }

    def plan(self, state: State) -> State:
        files_summary = "\n".join(f"{k}: {len(v)} chars" for k, v in state["files"].items())
        commits_summary = "\n".join(state["commits"])

        system_prompt = f"""
You are an AI software engineer. A user gave you this task: "{state['prompt']}"

Based on the current repo files and recent commits, list which files you'll
need to modify or create. Only include editable source files
({", ".join(config.EDITABLE_EXTENSIONS)}). New files are allowed.

Recent commits:
{commits_summary}

Files in repo:
{files_summary}

Return ONLY a JSON list of objects with "file" and "action" keys, e.g.:
[
  {{"file": "src/components/NavBar.tsx", "action": "Add login/logout button"}},
  {{"file": "src/components/ui/dialog.tsx", "action": "Create login modal"}}
]
"""
        self.emit("🧠 Generating plan with the model (this may take a moment)...")
        response = self.llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=state["prompt"])]
        )
        plan = _extract_json_array(response.content)
        self.emit(f"📋 Plan: {len(plan)} file action(s)")
        for item in plan:
            self.emit(f"   • {item.get('file')} — {item.get('action')}")
        return {"plan": plan, "current_idx": 0}

    def rewrite(self, state: State) -> State:
        idx = state["current_idx"]
        plan = state["plan"]
        entry = plan[idx]
        path = entry["file"]

        updates: State = {"current_idx": idx + 1}

        # Guard 1: never write outside the repo root.
        if not config.is_safe_path(self.repo_path, path):
            self.emit(f"🚫 Refused unsafe path: {path}")
            return updates

        # Guard 2: only touch editable source files.
        if not path.endswith(config.EDITABLE_EXTENSIONS):
            self.emit(f"⏭️ Skipped non-editable file: {path}")
            return updates

        existing = state["files"].get(path, "")
        if existing.strip() == "":
            system_prompt = (
                f"You are creating a NEW file for this project.\n\n"
                f"File: {path}\nAction: {entry['action']}\n\n"
                f"Generate the full file content."
            )
        else:
            system_prompt = (
                f"You are modifying an existing file. Do not change anything "
                f"unrelated and do not reformat the whole file.\n\n"
                f"File: {path}\nAction: {entry['action']}\n\n"
                f"--- FILE CONTENT BEFORE ---\n{existing}"
            )

        response = self.llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content="Return the full file content. "
                    "If no changes are needed, reply exactly NO_CHANGE."
                ),
            ]
        )
        content = response.content.strip()
        if content.upper() == "NO_CHANGE":
            self.emit(f"⏭️ No change: {path}")
            return updates

        files = dict(state["files"])
        files[path] = content
        modified = list(state["modified_files"])
        if path not in modified:
            modified.append(path)
        self.emit(f"✅ Modified/created: {path}")
        updates.update(files=files, modified_files=modified)
        return updates

    def _should_continue(self, state: State) -> str:
        return "rewrite" if state["current_idx"] < len(state["plan"]) else "commit"

    def commit(self, state: State) -> State:
        modified = state.get("modified_files", [])
        if not modified:
            self.emit("ℹ️ No files were modified — nothing to commit.")
            return {}

        repo_path = state["repo_path"]
        branch = f"ai-feature-{uuid.uuid4().hex[:8]}"
        self._git(["checkout", "-b", branch], repo_path)

        for rel in modified:
            abs_path = os.path.join(repo_path, rel)
            parent = os.path.dirname(abs_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(state["files"][rel])

        self._git(["add", *modified], repo_path)
        self._git(
            ["commit", "-m", f"AI changes for: {state['prompt'][:60]}"], repo_path
        )

        if config.AUTO_PUSH:
            self._git(["push", "-u", "origin", branch], repo_path)
            self.emit(f"🚀 Pushed branch: {branch}")
        else:
            self.emit(f"💾 Committed locally to branch: {branch} (push disabled)")

        return {"committed_branch": branch}

    @staticmethod
    def _git(args: list[str], cwd: str) -> None:
        subprocess.run(["git", *args], cwd=cwd, check=True)
