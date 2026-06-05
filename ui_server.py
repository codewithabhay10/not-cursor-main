"""Flask web server: thin HTTP layer over the LangGraph CodingAgent.

Flow per request:

    POST /execute            -> spawns a background job, returns {job_id}
    GET  /stream/<job_id>    -> SSE stream of progress + a "review" event
    POST /decision/<job_id>  -> human's approve/reject decision, resumes the run

Each job is fully isolated (its own queues), so concurrent requests never
share state.
"""
import json
import queue
import threading
import uuid
from dataclasses import dataclass, field

from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS

import config
from agent import CodingAgent, build_llm

app = Flask(__name__)
CORS(app)

_DONE = "__DONE__"
_STREAM_TIMEOUT_S = 600          # stop streaming after 10 idle minutes
_DECISION_TIMEOUT_S = 600        # how long to wait for a human decision


@dataclass
class Job:
    """In-memory handle for one agent run."""

    out: "queue.Queue" = field(default_factory=queue.Queue)       # progress -> browser
    decision: "queue.Queue" = field(default_factory=queue.Queue)  # browser -> agent


JOBS: dict[str, Job] = {}


def _run_job(job_id: str, prompt: str) -> None:
    job = JOBS[job_id]

    def emit(message: str) -> None:
        job.out.put(message)

    try:
        agent = CodingAgent(build_llm(), config.TARGET_REPO_PATH, emit=emit)
        result = agent.start(prompt)

        # If the graph paused at the review interrupt, surface the diffs and
        # wait for the human's decision before resuming.
        if "__interrupt__" in result:
            payload = result["__interrupt__"][0].value  # {"type": "review", "diffs": [...]}
            job.out.put({"type": "review", "diffs": payload.get("diffs", [])})
            try:
                decision = job.decision.get(timeout=_DECISION_TIMEOUT_S)
            except queue.Empty:
                emit("⏱️ Review timed out — discarding changes.")
                decision = {"approved": []}
            result = agent.resume(decision)

        branch = result.get("committed_branch")
        if branch and config.GITHUB_REPO:
            base = f"https://github.com/{config.GITHUB_REPO}"
            emit(f"GitHub: {base}/tree/{branch}")
            emit(f"Pull Request: {base}/pull/new/{branch}")
        emit("🎉 Workflow finished.")
    except Exception as exc:  # surface any failure to the UI
        emit(f"❌ Error: {exc}")
    finally:
        job.out.put(_DONE)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/execute", methods=["POST"])
def execute():
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    job_id = uuid.uuid4().hex
    JOBS[job_id] = Job()
    threading.Thread(target=_run_job, args=(job_id, prompt), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/decision/<job_id>", methods=["POST"])
def decision(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job_id"}), 404
    data = request.get_json(silent=True) or {}
    approved = data.get("approved", [])
    job.decision.put({"approved": approved})
    return jsonify({"ok": True, "approved": approved})


@app.route("/stream/<job_id>")
def stream(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job_id"}), 404

    def generate():
        idle = 0
        try:
            while idle < _STREAM_TIMEOUT_S:
                try:
                    item = job.out.get(timeout=1)
                    idle = 0
                    if item == _DONE:
                        yield f"data: {json.dumps({'type': 'done'})}\n\n"
                        return
                    if isinstance(item, dict):
                        # Structured event (e.g. the review payload).
                        yield f"data: {json.dumps(item)}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'output', 'message': item})}\n\n"
                except queue.Empty:
                    idle += 1
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
            yield f"data: {json.dumps({'type': 'output', 'message': '❌ Timed out.'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        finally:
            JOBS.pop(job_id, None)  # clean up regardless of how the stream ends

    response = Response(generate(), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Connection"] = "keep-alive"
    return response


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
