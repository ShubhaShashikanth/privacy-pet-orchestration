"""
Federated Learning Client with DP-SGD
---------------------------------------
Implements Differentially Private Stochastic Gradient Descent
(Abadi et al., "Deep Learning with Differential Privacy", CCS 2016).

Core DP-SGD steps per mini-batch:
  1. Poisson sampling: each local record joins the batch independently
     with probability q = BATCH_SIZE / n  (needed for the accountant below).
  2. Compute per-sample gradients (not batch-averaged).
  3. Clip each per-sample gradient: g_i ← g_i / max(1, ‖g_i‖₂ / C)
     This bounds the sensitivity — one record can shift the sum by at most C.
  4. Sum clipped gradients, then add Gaussian noise scaled to (C · σ):
       g̃ = Σ clip(g_i) + 𝒩(0, (Cσ)²I)
  5. Divide by the *expected* batch size (q · n) to get the noisy mean.

Privacy accounting:
  Every step is a Poisson-subsampled Gaussian mechanism with sampling
  rate q and noise multiplier σ. Steps are composed with a Rényi-DP
  accountant (Google's dp-accounting library, Mironov et al. 2017/2019)
  and converted to an (ε, δ) guarantee. The ε reported is record-level
  DP for this client's local dataset, covering all rounds so far.
"""

import os, time, logging
import numpy as np
import requests
from dp_accounting import dp_event
from dp_accounting.rdp import rdp_privacy_accountant
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# ── hyperparameters ────────────────────────────────────────
SERVER_URL      = os.getenv("SERVER_URL",      "http://server:5000")
CLIENT_ID       = int(os.getenv("CLIENT_ID",   "0"))
NUM_CLIENTS     = int(os.getenv("NUM_CLIENTS", "3"))
ROUNDS          = int(os.getenv("ROUNDS",      "10"))

# Optimisation
LEARNING_RATE   = float(os.getenv("LR",          "0.05"))
LOCAL_EPOCHS    = int(os.getenv("LOCAL_EPOCHS",  "3"))
BATCH_SIZE      = int(os.getenv("BATCH_SIZE",    "16"))

# DP-SGD privacy parameters
CLIP_NORM       = float(os.getenv("CLIP_NORM",        "1.0"))   # C
NOISE_MULTIPLIER= float(os.getenv("NOISE_MULTIPLIER", "3.0"))   # σ
DELTA           = float(os.getenv("DELTA",            "1e-5"))  # δ for (ε,δ)-DP


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

def add_noise(sum_grad_w, sum_grad_b, C, sigma, expected_batch):
    """
    Add calibrated Gaussian noise to the *summed* clipped gradients,
    then normalise by the expected batch size.

    Noise std for each coordinate = C · σ  (the Gaussian mechanism).
    Dividing by the expected (public) batch size rather than the actual
    sampled size keeps the batch size itself from leaking information.
    """
    noise_w = np.random.normal(0.0, C * sigma, size=sum_grad_w.shape)
    noise_b = np.random.normal(0.0, C * sigma)
    return (sum_grad_w + noise_w) / expected_batch, (sum_grad_b + noise_b) / expected_batch

def steps_per_epoch(n, batch_size):
    return max(1, round(n / batch_size))

def dp_sgd_epoch(X, y, w, b, C, sigma, lr, batch_size):
    """
    One epoch of DP-SGD over the local dataset, using Poisson sampling.
    Returns the updated model and the number of noisy steps taken.
    """
    n = len(X)
    q = batch_size / n
    steps = steps_per_epoch(n, batch_size)
    for _ in range(steps):
        mask = np.random.random(n) < q          # each record in w.p. q
        X_b, y_b = X[mask], y[mask]

        if mask.any():
            gw, gb = per_sample_gradients(X_b, y_b, w, b)
            gw, gb = clip_per_sample(gw, gb, C)
            sum_w, sum_b = gw.sum(0), gb.sum()
        else:                                   # empty batch: noise only
            sum_w, sum_b = np.zeros_like(w), 0.0

        mgw, mgb = add_noise(sum_w, sum_b, C, sigma, q * n)
        w = w - lr * mgw
        b = b - lr * mgb
    return w, b, steps

def privacy_spent(steps, sigma, delta, n, batch_size):
    """
    ε for `steps` compositions of the Poisson-subsampled Gaussian
    mechanism (sampling rate q = batch_size / n, noise multiplier σ),
    computed with a Rényi-DP accountant and converted to (ε, δ)-DP.
    """
    accountant = rdp_privacy_accountant.RdpAccountant()
    event = dp_event.PoissonSampledDpEvent(
        sampling_probability=batch_size / n,
        event=dp_event.GaussianDpEvent(noise_multiplier=sigma),
    )
    accountant.compose(event, steps)
    return accountant.get_epsilon(target_delta=delta)


# ── data partitioning ──────────────────────────────────────
def local_dataset(client_id, num_clients):
    """
    Horizontal FL: every client has the same 30 features
    but a disjoint subset of patients.
    Breast-cancer dataset is a clean proxy for health data
    (binary label: malignant / benign).
    """
    X, y = load_breast_cancer(return_X_y=True)

    # Deterministic partition by client index. (In this simulation every
    # container loads the public dataset and keeps only its own slice; in
    # a real deployment each institution would hold only its own records.)
    mask = np.arange(len(X)) % num_clients == client_id
    Xi, yi = X[mask], y[mask]
    X_tr, X_te, y_tr, y_te = train_test_split(Xi, yi, test_size=0.2, random_state=42)

    # Feature scaling is fitted on this client's own training data only,
    # so no statistics from other clients' records are used.
    sc = StandardScaler().fit(X_tr)
    return sc.transform(X_tr), sc.transform(X_te), y_tr, y_te


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

    for rnd in range(ROUNDS):
        # ── 1. pull global model for this round ───────────────
        # Wait until the server has aggregated the previous round,
        # so every client starts round `rnd` from the same global model.
        while True:
            resp = requests.get(f"{SERVER_URL}/model", timeout=10).json()
            if resp["round"] >= rnd:
                break
            time.sleep(0.5)
        w = np.array(resp["weights"])
        b = float(resp["bias"])

        # ── 2. local training with DP-SGD ────────────────────
        for _ in range(LOCAL_EPOCHS):
            w, b, steps = dp_sgd_epoch(X_tr, y_tr, w, b,
                                       CLIP_NORM, NOISE_MULTIPLIER,
                                       LEARNING_RATE, BATCH_SIZE)
            total_steps += steps

        acc   = accuracy(X_te, y_te, w, b)
        eps   = privacy_spent(total_steps, NOISE_MULTIPLIER,
                               DELTA, len(X_tr), BATCH_SIZE)

        logging.info(
            "[Client %d] Round %d — local acc: %.4f | ε spent so far: %.2f (δ=%.0e)",
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

        time.sleep(0.5)   # stagger clients slightly

    logging.info("[Client %d] Training complete.", CLIENT_ID)


if __name__ == "__main__":
    run()
