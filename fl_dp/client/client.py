"""
Federated Learning Client with DP-SGD
---------------------------------------
Implements Differentially Private Stochastic Gradient Descent
(Abadi et al., "Deep Learning with Differential Privacy", CCS 2016).

Core DP-SGD steps per mini-batch:
  1. Compute per-sample gradients (not batch-averaged).
  2. Clip each per-sample gradient: g_i ← g_i / max(1, ‖g_i‖₂ / C)
     This bounds the sensitivity — one sample can shift the sum by at most C.
  3. Sum clipped gradients, then add Gaussian noise scaled to (C · σ):
       g̃ = Σ clip(g_i) + 𝒩(0, (Cσ)²I)
  4. Divide by batch size to get the noisy mean gradient.

Privacy accounting (Gaussian mechanism):
  Each step consumes roughly ε ≈ σ⁻¹ · √(2 ln(1/δ)) of the budget.
  Full accounting uses RDP (Rényi DP) compositors; here we print the
  naive per-step estimate so the result is interpretable in a demo.
"""

import os, time, math, logging
import numpy as np
import requests
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [CLIENT-%(client_id)s] %(message)s")

# ── hyperparameters ────────────────────────────────────────
SERVER_URL      = os.getenv("SERVER_URL",      "http://server:5000")
CLIENT_ID       = int(os.getenv("CLIENT_ID",   "0"))
NUM_CLIENTS     = int(os.getenv("NUM_CLIENTS", "3"))
ROUNDS          = int(os.getenv("ROUNDS",      "10"))

# Optimisation
LEARNING_RATE   = float(os.getenv("LR",          "0.05"))
LOCAL_EPOCHS    = int(os.getenv("LOCAL_EPOCHS",  "3"))
BATCH_SIZE      = int(os.getenv("BATCH_SIZE",    "32"))

# DP-SGD privacy parameters
CLIP_NORM       = float(os.getenv("CLIP_NORM",        "1.0"))   # C
NOISE_MULTIPLIER= float(os.getenv("NOISE_MULTIPLIER", "1.1"))   # σ
DELTA           = float(os.getenv("DELTA",            "1e-5"))  # δ for (ε,δ)-DP

log = logging.getLogger(__name__)
log.info = lambda msg, *a: logging.getLogger("CLIENT").info(msg, *a)


# ── model ─────────────────────────────────────────────────
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def predict_proba(X, w, b):
    return sigmoid(X @ w + b)

def accuracy(X, y, w, b):
    return (predict_proba(X, w, b).round() == y).mean()


# ── DP-SGD ─────────────────────────────────────────────────
def per_sample_gradients(X_b, y_b, w, b):
    """Return per-sample gradient arrays: (n, d) for w, (n,) for b."""
    p      = sigmoid(X_b @ w + b)          # (n,)
    errors = p - y_b                        # (n,)
    grad_w = errors[:, None] * X_b         # (n, d)
    grad_b = errors                         # (n,)
    return grad_w, grad_b

def clip_per_sample(grad_w, grad_b, C):
    """
    Clip each sample's gradient so its L2 norm ≤ C.
    Combined norm = sqrt(‖gw‖² + gb²).
    """
    norms  = np.sqrt((grad_w ** 2).sum(axis=1) + grad_b ** 2 + 1e-12)  # (n,)
    scales = np.minimum(1.0, C / norms)                                  # (n,)
    return grad_w * scales[:, None], grad_b * scales

def add_noise(sum_grad_w, sum_grad_b, C, sigma, n):
    """
    Add calibrated Gaussian noise to the *summed* clipped gradients,
    then normalise by batch size.

    Noise std for each coordinate = C · σ  (the Gaussian mechanism).
    """
    noise_w = np.random.normal(0.0, C * sigma, size=sum_grad_w.shape)
    noise_b = np.random.normal(0.0, C * sigma)
    return (sum_grad_w + noise_w) / n, (sum_grad_b + noise_b) / n

def dp_sgd_epoch(X, y, w, b, C, sigma, lr, batch_size):
    """One full epoch of DP-SGD over the local dataset."""
    idx = np.random.permutation(len(X))
    for start in range(0, len(X), batch_size):
        batch = idx[start : start + batch_size]
        X_b, y_b = X[batch], y[batch]

        gw, gb   = per_sample_gradients(X_b, y_b, w, b)
        gw, gb   = clip_per_sample(gw, gb, C)
        mgw, mgb = add_noise(gw.sum(0), gb.sum(), C, sigma, len(batch))

        w = w - lr * mgw
        b = b - lr * mgb
    return w, b

def privacy_spent(steps, sigma, C, delta, n, batch_size):
    """
    Naive (conservative) per-step Gaussian mechanism ε estimate.
    Each step contributes ≈ (C/σ) · sqrt(2·ln(1/δ)) to epsilon.
    For rigorous accounting, use google/dp-accounting or autodp.
    """
    q   = batch_size / n                             # sampling rate
    eps_per_step = (C / sigma) * math.sqrt(2 * math.log(1 / delta))
    return steps * q * eps_per_step


# ── data partitioning ──────────────────────────────────────
def local_dataset(client_id, num_clients):
    """
    Horizontal FL: every client has the same 30 features
    but a disjoint subset of patients.
    Breast-cancer dataset is a clean proxy for health data
    (binary label: malignant / benign).
    """
    X, y = load_breast_cancer(return_X_y=True)
    sc   = StandardScaler()
    X    = sc.fit_transform(X)

    # Deterministic partition by client index
    mask = np.arange(len(X)) % num_clients == client_id
    Xi, yi = X[mask], y[mask]
    return train_test_split(Xi, yi, test_size=0.2, random_state=42)


# ── federated training loop ────────────────────────────────
def wait_for_server(url, retries=20, delay=2):
    for _ in range(retries):
        try:
            r = requests.get(f"{url}/model", timeout=3)
            if r.ok:
                return
        except Exception:
            pass
        time.sleep(delay)
    raise RuntimeError(f"Server not reachable at {url}")


def run():
    X_tr, X_te, y_tr, y_te = local_dataset(CLIENT_ID, NUM_CLIENTS)
    logging.info(
        "[Client %d] Dataset: %d train / %d test samples",
        CLIENT_ID, len(X_tr), len(X_te),
    )

    wait_for_server(SERVER_URL)

    total_steps  = 0
    last_round   = -1

    for rnd in range(ROUNDS):
        # ── 1. pull global model ──────────────────────────────
        resp = requests.get(f"{SERVER_URL}/model", timeout=10).json()
        w    = np.array(resp["weights"])
        b    = float(resp["bias"])

        if resp["round"] == last_round:
            # server hasn't aggregated yet — back off and retry
            time.sleep(1)

        # ── 2. local training with DP-SGD ────────────────────
        for _ in range(LOCAL_EPOCHS):
            w, b  = dp_sgd_epoch(X_tr, y_tr, w, b,
                                  CLIP_NORM, NOISE_MULTIPLIER,
                                  LEARNING_RATE, BATCH_SIZE)
            total_steps += math.ceil(len(X_tr) / BATCH_SIZE)

        acc   = accuracy(X_te, y_te, w, b)
        eps   = privacy_spent(total_steps, NOISE_MULTIPLIER,
                               CLIP_NORM, DELTA, len(X_tr), BATCH_SIZE)

        logging.info(
            "[Client %d] Round %d — local acc: %.4f | ε spent so far: %.4f (δ=%.0e)",
            CLIENT_ID, rnd + 1, acc, eps, DELTA,
        )

        # ── 3. push local update to server ───────────────────
        requests.post(f"{SERVER_URL}/update", json={
            "weights":   w.tolist(),
            "bias":      float(b),
            "n_samples": len(X_tr),
            "client_id": CLIENT_ID,
            "local_acc": float(acc),
        }, timeout=10)

        last_round = rnd
        time.sleep(0.5)   # stagger clients slightly

    logging.info("[Client %d] Training complete.", CLIENT_ID)


if __name__ == "__main__":
    run()
