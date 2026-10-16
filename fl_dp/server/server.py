"""
Federated Learning Aggregation Server
--------------------------------------
Implements FedAvg (McMahan et al., 2017) aggregation.
Clients submit local model updates; the server averages
them weighted by each client's sample count and broadcasts
the updated global model back.

Endpoint summary:
  GET  /model          → returns current global weights + round number
  POST /update         → receives a client's local update
  GET  /metrics        → returns per-round accuracy history
"""

import threading
import logging
from flask import Flask, request, jsonify
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [SERVER] %(message)s")
app = Flask(__name__)

# ── global state ──────────────────────────────────────────
N_FEATURES   = 30   # breast-cancer dataset
NUM_CLIENTS  = 3
ROUNDS       = 10

state = {
    "weights":      np.zeros(N_FEATURES),
    "bias":         0.0,
    "round":        0,
    "pending":      [],      # client updates waiting to be aggregated
    "round_metrics": [],     # [(round, avg_train_acc)] logged by clients
}
lock = threading.Lock()
round_complete = threading.Event()


# ── helpers ───────────────────────────────────────────────
def fedavg(updates: list[dict]) -> tuple[np.ndarray, float]:
    """Weighted average of client weight vectors."""
    total = sum(u["n_samples"] for u in updates)
    w = sum(u["weights"] * u["n_samples"] / total for u in updates)
    b = sum(u["bias"]    * u["n_samples"] / total for u in updates)
    return w, b


# ── endpoints ─────────────────────────────────────────────
@app.get("/model")
def get_model():
    with lock:
        return jsonify({
            "weights": state["weights"].tolist(),
            "bias":    float(state["bias"]),
            "round":   state["round"],
        })


@app.post("/update")
def receive_update():
    data = request.get_json()
    with lock:
        state["pending"].append({
            "weights":  np.array(data["weights"]),
            "bias":     float(data["bias"]),
            "n_samples": int(data["n_samples"]),
            "client_id": data.get("client_id"),
            "local_acc": data.get("local_acc"),
        })
        logging.info(
            "Received update from client %s  (round %d, %d/%d clients)",
            data.get("client_id"), state["round"] + 1,
            len(state["pending"]), NUM_CLIENTS,
        )

        if len(state["pending"]) >= NUM_CLIENTS:
            # ── aggregate ──
            new_w, new_b = fedavg(state["pending"])
            state["weights"] = new_w
            state["bias"]    = new_b
            state["round"]  += 1

            avg_acc = np.mean([u["local_acc"] for u in state["pending"] if u["local_acc"] is not None])
            state["round_metrics"].append({"round": state["round"], "avg_local_acc": float(avg_acc)})

            logging.info("Round %d complete — avg local acc: %.4f", state["round"], avg_acc)
            state["pending"].clear()
            round_complete.set()
            round_complete.clear()

    return jsonify({"status": "ok", "round": state["round"]})


@app.get("/metrics")
def get_metrics():
    with lock:
        return jsonify(state["round_metrics"])


if __name__ == "__main__":
    logging.info("Server starting — waiting for %d clients, %d rounds", NUM_CLIENTS, ROUNDS)
    app.run(host="0.0.0.0", port=5000, threaded=True)
