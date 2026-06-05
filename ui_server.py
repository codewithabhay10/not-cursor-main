"""Flask web server: thin HTTP layer over the LangGraph CodingAgent.

Each /execute call spawns a background job with its own message queue, keyed
by a job_id. The browser then opens an SSE stream at /stream/<job_id>. This
keeps concurrent requests fully isolated (no shared global state).
"""
import json
import queue
import threading
import uuid

from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS

import config
from agent import CodingAgent, build_llm

app = Flask(__name__)
CORS(app)

# job_id -> Queue of human-readable progress strings. A sentinel "__DONE__"
# marks the end of a job's output.
JOBS: dict[str, "queue.Queue[str]"] = {}
_DONE = "__DONE__"
_STREAM_TIMEOUT_S = 600  # stop streaming after 10 idle minutes


def _run_job(job_id: str, prompt: str) -> None:
    q = JOBS[job_id]

    def emit(message: str) -> None:
        q.put(message)

    try:
        agent = CodingAgent(build_llm(), config.TARGET_REPO_PATH, emit=emit)
        result = agent.run(prompt)

        branch = result.get("committed_branch")
        if branch and config.GITHUB_REPO:
            base = f"https://github.com/{config.GITHUB_REPO}"
            emit("🎉 Workflow completed successfully!")
            emit(f"GitHub: {base}/tree/{branch}")
            emit(f"Pull Request: {base}/pull/new/{branch}")
        else:
            emit("🎉 Workflow completed successfully!")
    except Exception as exc:  # surface any failure to the UI
        emit(f"❌ Error: {exc}")
    finally:
        emit(_DONE)


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
    JOBS[job_id] = queue.Queue()
    thread = threading.Thread(target=_run_job, args=(job_id, prompt), daemon=True)
    thread.start()
    return jsonify({"job_id": job_id})


@app.route("/stream/<job_id>")
def stream(job_id: str):
    q = JOBS.get(job_id)
    if q is None:
        return jsonify({"error": "Unknown job_id"}), 404

    def generate():
        idle = 0
        try:
            while idle < _STREAM_TIMEOUT_S:
                try:
                    message = q.get(timeout=1)
                    idle = 0
                    if message == _DONE:
                        yield f"data: {json.dumps({'type': 'done'})}\n\n"
                        return
                    yield f"data: {json.dumps({'type': 'output', 'message': message})}\n\n"
                except queue.Empty:
                    idle += 1
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
            yield f"data: {json.dumps({'type': 'output', 'message': '❌ Timeout waiting for the model'})}\n\n"
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
