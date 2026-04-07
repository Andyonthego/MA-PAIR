"""
Flask server — bridges the MA-PAIR engine to the React dashboard.
Streams logs via Server-Sent Events (SSE).
"""

import json
import queue
import threading
from dataclasses import asdict
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import ma_pair as engine
import harmless_backend as harmless_engine

app = Flask(__name__)
CORS(app)

_job_queues: dict[str, queue.Queue] = {}
_job_results: dict[str, dict] = {}


def _stream_run(job_id: str, goal: str, strategy_name: str):
    q = _job_queues[job_id]

    def log(msg: str):
        q.put({"type": "log", "message": msg})

    try:
        if strategy_name == "harmless_approach":
            # Use harmless_backend
            strategy = harmless_engine.STRATEGY
            result = harmless_engine.run_single_test("custom_goal", goal, strategy, 0, log=log)
        else:
            # Use ma_pair
            strategy = next((s for s in engine.STRATEGY_DB if s["name"] == strategy_name), None)
            if not strategy:
                raise ValueError(f"Unknown strategy: {strategy_name}")
            result = engine.run_single(goal, strategy, 0, log=log)
        
        # Wrap in dict to match expected format
        wrapped_result = {
            "goal": goal,
            "results": [asdict(result)],
            "summary": {
                "total_runs": 1,
                "successes": 1 if result.success else 0,
                "success_rate": 1.0 if result.success else 0.0,
            }
        }
        
        _job_results[job_id] = wrapped_result
        q.put({"type": "done", "result": wrapped_result})
    except Exception as e:
        q.put({"type": "error", "message": str(e)})
    finally:
        q.put(None)  # sentinel


@app.route("/run", methods=["POST"])
def start_run():
    data = request.json
    goal = data.get("goal", "").strip()
    strategy_name = data.get("strategy", "harmless_approach")
    if not goal:
        return jsonify({"error": "goal is required"}), 400

    import uuid
    job_id = str(uuid.uuid4())
    _job_queues[job_id] = queue.Queue()

    thread = threading.Thread(target=_stream_run, args=(job_id, goal, strategy_name), daemon=True)
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
