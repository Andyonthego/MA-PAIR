"""
Flask server — bridges the MA-PAIR engine to the React dashboard.
Streams logs via Server-Sent Events (SSE).
"""

import json
import queue
import threading
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import ma_pair as engine

app = Flask(__name__)
CORS(app)

_job_queues: dict[str, queue.Queue] = {}
_job_results: dict[str, dict] = {}


def _stream_run(job_id: str, goal: str):
    q = _job_queues[job_id]

    def log(msg: str):
        q.put({"type": "log", "message": msg})

    try:
        result = engine.run_ma_pair(goal, log=log)
        _job_results[job_id] = result
        q.put({"type": "done", "result": result})
    except Exception as e:
        q.put({"type": "error", "message": str(e)})
    finally:
        q.put(None)  # sentinel


@app.route("/run", methods=["POST"])
def start_run():
    data = request.json
    goal = data.get("goal", "").strip()
    if not goal:
        return jsonify({"error": "goal is required"}), 400

    import uuid
    job_id = str(uuid.uuid4())
    _job_queues[job_id] = queue.Queue()

    thread = threading.Thread(target=_stream_run, args=(job_id, goal), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/stream/<job_id>")
def stream(job_id: str):
    if job_id not in _job_queues:
        return jsonify({"error": "unknown job"}), 404

    def generate():
        q = _job_queues[job_id]
        while True:
            item = q.get()
            if item is None:
                yield "data: [DONE]\n\n"
                break
            yield f"data: {json.dumps(item)}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/result/<job_id>")
def get_result(job_id: str):
    if job_id not in _job_results:
        return jsonify({"error": "not ready or unknown"}), 404
    return jsonify(_job_results[job_id])


if __name__ == "__main__":
    app.run(port=5001, debug=False)
