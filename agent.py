"""The coding agent, implemented as a LangGraph state machine.

Graph shape (with human-in-the-loop review before anything is committed):

    START -> load_context -> plan -> rewrite --(more files?)--> rewrite
                                       |
                                       (done, has changes) -> review -> commit -> END
                                       |
                                       (no changes) -> END

The ``review`` node calls LangGraph's ``interrupt()``: the run pauses, the
proposed diffs are surfaced to the UI, and execution only continues once a
human resumes the graph with their approval decision. A ``MemorySaver``
checkpointer makes that pause/resume possible.

Each node receives the shared ``State`` and returns a *partial* update that
LangGraph merges back in.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import subprocess
import uuid
from typing import Callable, Optional

from git import Repo
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

import config


class State(TypedDict, total=False):
    """Shared state threaded through every node of the graph."""

    prompt: str
    repo_path: str
    files: dict[str, str]          # original (baseline) repo content
    commits: list[str]
    file_list: list[str]
    plan: list[dict]
    current_idx: int
    proposed: dict[str, str]       # path -> new content the model produced
    modified_files: list[str]
    approved_files: list[str]      # subset the human approved at review time
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


def _unified_diff(path: str, before: str, after: str) -> dict:
    """Build a UI-friendly diff record for one file."""
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    diff_lines = list(
        difflib.unified_diff(
            before_lines, after_lines, fromfile=f"a/{path}", tofile=f"b/{path}"
        )
    )
    additions = sum(1 for ln in diff_lines if ln.startswith("+") and not ln.startswith("+++"))
    deletions = sum(1 for ln in diff_lines if ln.startswith("-") and not ln.startswith("---"))
    return {
        "file": path,
        "status": "new" if before.strip() == "" else "modified",
        "additions": additions,
        "deletions": deletions,
        "diff": "".join(diff_lines),
    }


class CodingAgent:
    """Compiles and runs the LangGraph workflow for a single request.

    Usage is two-phase because of the human-in-the-loop interrupt:

        result = agent.start(prompt)        # runs until the review interrupt
        ... show result["__interrupt__"] ...
        result = agent.resume({"approved": [...]})  # commits approved files

    ``emit`` streams human-readable progress to the UI; it replaces the old
    (thread-unsafe) ``builtins.print`` monkeypatch.
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
        self.thread_id = uuid.uuid4().hex
        self.graph = self._build_graph()

    # --- graph wiring ------------------------------------------------------
    def _build_graph(self):
        g = StateGraph(State)
        g.add_node("load_context", self.load_context)
        g.add_node("plan", self.plan)
        g.add_node("rewrite", self.rewrite)
        g.add_node("review", self.review)
        g.add_node("commit", self.commit)

        g.add_edge(START, "load_context")
        g.add_edge("load_context", "plan")
        g.add_edge("plan", "rewrite")
        g.add_conditional_edges(
            "rewrite",
            self._route_after_rewrite,
            {"rewrite": "rewrite", "review": "review", "end": END},
        )
        g.add_edge("review", "commit")
        g.add_edge("commit", END)
        # A checkpointer is required for interrupt()/resume to work.
        return g.compile(checkpointer=MemorySaver())

    def _invoke_config(self) -> dict:
        return {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": config.GRAPH_RECURSION_LIMIT,
        }

    def start(self, prompt: str) -> State:
        """Run the graph until it either finishes or hits the review interrupt."""
        initial: State = {
            "prompt": prompt,
            "repo_path": self.repo_path,
            "files": {},
            "commits": [],
            "file_list": [],
            "plan": [],
            "current_idx": 0,
            "proposed": {},
            "modified_files": [],
            "approved_files": [],
        }
        return self.graph.invoke(initial, config=self._invoke_config())

    def resume(self, decision: dict) -> State:
        """Resume a paused run, passing the human's approval ``decision``."""
        return self.graph.invoke(Command(resume=decision), config=self._invoke_config())

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

        # Stage the proposal; the baseline in state["files"] is left untouched
        # so we can diff old vs new at review time.
        proposed = dict(state.get("proposed", {}))
        proposed[path] = content
        modified = list(state.get("modified_files", []))
        if path not in modified:
            modified.append(path)
        self.emit(f"✍️ Drafted change: {path}")
        updates.update(proposed=proposed, modified_files=modified)
        return updates

    def _route_after_rewrite(self, state: State) -> str:
        if state["current_idx"] < len(state["plan"]):
            return "rewrite"
        return "review" if state.get("modified_files") else "end"

    def review(self, state: State) -> State:
        """Pause for human approval, surfacing the proposed diffs."""
        diffs = [
            _unified_diff(path, state["files"].get(path, ""), state["proposed"][path])
            for path in state["modified_files"]
        ]
        self.emit(f"🔍 Awaiting review of {len(diffs)} proposed change(s)...")

        # Execution stops here. The value is handed to the UI; the return value
        # of interrupt() is whatever the human passes to resume().
        decision = interrupt({"type": "review", "diffs": diffs})

        approved = decision.get("approved", []) if isinstance(decision, dict) else []
        return {"approved_files": approved}

    def commit(self, state: State) -> State:
        modified = state.get("modified_files", [])
        approved = state.get("approved_files", modified)
        to_commit = [p for p in modified if p in approved]

        if not to_commit:
            self.emit("ℹ️ No changes approved — nothing was committed.")
            return {}

        repo_path = state["repo_path"]
        branch = f"ai-feature-{uuid.uuid4().hex[:8]}"
        self._git(["checkout", "-b", branch], repo_path)

        for rel in to_commit:
            abs_path = os.path.join(repo_path, rel)
            parent = os.path.dirname(abs_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(state["proposed"][rel])

        self._git(["add", *to_commit], repo_path)
        self._git(["commit", "-m", f"AI changes for: {state['prompt'][:60]}"], repo_path)

        self.emit(f"✅ Committed {len(to_commit)} file(s) to branch: {branch}")
        if config.AUTO_PUSH:
            self._git(["push", "-u", "origin", branch], repo_path)
            self.emit(f"🚀 Pushed branch: {branch}")
        else:
            self.emit("💾 Push disabled (AUTO_PUSH=false) — branch is local only.")

        return {"committed_branch": branch}

    @staticmethod
    def _git(args: list[str], cwd: str) -> None:
        subprocess.run(["git", *args], cwd=cwd, check=True)
