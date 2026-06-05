# Not Cursor — an autonomous coding agent

Not Cursor turns a natural-language request ("add a dark-mode toggle") into a
set of multi-file Git changes. It loads your repository as context, asks a
locally-running LLM to produce a structured plan, rewrites each file, and
commits the result to a fresh feature branch — streaming every step to the
browser in real time.

It is built as a **LangGraph state machine** running entirely on local
infrastructure (Ollama), so there are no API keys and no per-request cost.

> ⚠️ This is an experimental project. By design it only **commits locally** —
> pushing to a remote is opt-in (`AUTO_PUSH=true`).

## Architecture

The workflow is a compiled `langgraph.StateGraph`. Each node receives a shared
`State` (a `TypedDict`) and returns a partial update that LangGraph merges back
in. The `rewrite` node loops over itself through a **conditional edge** until
every planned file has been processed, then the run **pauses at `review`** for
human approval before anything is committed:

```
START → load_context → plan → rewrite ──(more files?)──┐
                                 ▲                       │
                                 └───────────────────────┘
                                          │ (done, has changes)
                                          ▼
                                   review  ⏸  ← interrupt(): waits for human
                                          │
                                          ▼
                                       commit → END
```

| Node           | Responsibility |
|----------------|----------------|
| `load_context` | Read every tracked text file + the last *N* commit messages via GitPython |
| `plan`         | Ask the LLM for a JSON list of `{file, action}` edits and parse it |
| `rewrite`      | For each planned file, draft new content (path-traversal–guarded) into `state["proposed"]`, leaving the baseline intact for diffing |
| `review`       | Compute unified diffs and call LangGraph's `interrupt()` — execution pauses until a human approves |
| `commit`       | Write **only the approved** files, create a branch, commit (and optionally push) |

### Tech stack

- **Orchestration:** LangGraph (`StateGraph`, conditional edges)
- **LLM:** Ollama (`llama3` by default) via `langchain-ollama` — runs locally
- **Backend:** Flask, with per-job Server-Sent Events (SSE) for streaming
- **VCS:** GitPython + `git` CLI
- **Frontend:** a single dependency-free HTML/JS page

### Project layout

```
config.py          # All tunables, loaded from .env (no hardcoded values)
agent.py           # State, CodingAgent, the LangGraph graph + nodes
ui_server.py       # Flask app: /execute, /stream/<id>, /decision/<id>
templates/
  index.html       # Terminal-style UI: live SSE output + diff review panel
.env.example       # Copy to .env to configure
```

## How streaming works

`/execute` creates a `job_id`, spins up a background thread, and returns
immediately. Each job owns its **own queue**, so concurrent requests never
interfere. The agent is given an `emit` callback that pushes progress messages
onto that queue, and the browser consumes them from `/stream/<job_id>` over
SSE. (An earlier version monkeypatched `builtins.print` and shared one global
queue — that was replaced because it broke under concurrent requests.)

## Human-in-the-loop review

The agent never commits blindly. After drafting all edits it hits the `review`
node, which calls LangGraph's [`interrupt()`](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/).
That **pauses the graph mid-run** (state is persisted by a `MemorySaver`
checkpointer) and pushes the proposed unified diffs to the browser, which
renders them with per-file checkboxes:

```
POST /execute            → {job_id}                 (start the run)
GET  /stream/<job_id>     → progress + a `review` SSE event with the diffs
POST /decision/<job_id>   → {"approved": ["a.py"]}   (resume with the choice)
```

The decision is handed back into the graph via `Command(resume=decision)`; only
the approved files are written and committed. This mirrors how real assistive
coding tools (Cursor, Copilot) gate changes behind human review.

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running, with a model pulled:
  ```bash
  ollama pull llama3
  ```
- A Git repository for the agent to operate on

### Install

```bash
pip install -r requirements.txt
cp .env.example .env        # then edit as needed
```

Key settings in `.env`:

| Variable           | Default        | Purpose |
|--------------------|----------------|---------|
| `OLLAMA_MODEL`     | `llama3`       | Which local model to use |
| `TARGET_REPO_PATH` | cwd            | Absolute path to the repo to edit |
| `AUTO_PUSH`        | `false`        | Push the branch to `origin` after committing |
| `GITHUB_REPO`      | —              | `owner/name`, used only to build UI links |

### Run

```bash
python ui_server.py
# open http://localhost:5000
```

Type a prompt (e.g. *"add input validation to the login form"*) and watch the
agent plan, edit, and commit.

## Safety notes

- **Path traversal is blocked** — file paths the model returns are validated to
  stay inside the target repo before anything is written (`config.is_safe_path`).
- **Only source files** (`.py .js .ts .jsx .tsx`) are edited.
- **No automatic push** unless you set `AUTO_PUSH=true`. Everything lands on a
  throwaway `ai-feature-*` branch, so the change is always easy to discard.

## Known limitations / roadmap

This is a learning project; the current focus was correctness, a faithful
LangGraph implementation, and human-in-the-loop review. Natural next steps:

- **Structured output** via Pydantic / tool-calling instead of regex JSON parsing
- **Test-and-retry loop** — run the repo's tests after editing and feed failures
  back to the model (ReAct-style self-correction)
- **RAG over the repo** instead of stuffing every file into the prompt
- Unit tests + Docker Compose (app + Ollama)

---

Built with LangGraph, Flask, and Ollama.
