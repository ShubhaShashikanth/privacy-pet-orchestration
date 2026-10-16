# Privacy-PET Orchestration

A hands-on implementation of multiple Privacy-Enhancing Technologies (PETs) — built to demonstrate how DP, FL, HE, and SMPC compose into a privacy-preserving data pipeline.

> **Motivation:** CDPG's mission is building a modular orchestration layer that combines multiple PETs. This repo demonstrates each layer independently and shows how they interlock.

---

## What's inside

| Module | PET | Key technique | Status |
|--------|-----|---------------|--------|
| `fl_dp/` | Federated Learning + Differential Privacy | DP-SGD (gradient clipping + Gaussian noise), FedAvg | Dockerised, runnable |
| `demos/homomorphic_encryption.py` | Homomorphic Encryption | CKKS scheme via TenSEAL — mean on encrypted health data | Standalone script |
| `demos/smpc.py` | Secure Multi-Party Computation | Additive secret sharing + PSI sketch | Standalone script |

---

## FL + DP-SGD — Containerised Demo

Three hospital "clients" train a logistic regression model on disjoint partitions of the [Breast Cancer Wisconsin dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset) — a proxy for real health data.

**Data never leaves a container.** Only noisy gradient updates cross the network.

### Architecture

```
┌─────────────────────────────────────────────────────┐
│  docker-compose network                              │
│                                                      │
│  ┌──────────┐     GET /model      ┌──────────────┐  │
│  │ client-0 │◄───────────────────►│              │  │
│  │ (190 pts)│  POST /update       │    server    │  │
│  └──────────┘  (noisy gradients)  │  (FedAvg)    │  │
│  ┌──────────┐                     │              │  │
│  │ client-1 │◄───────────────────►│  aggregates  │  │
│  │ (190 pts)│                     │  3 clients   │  │
│  └──────────┘                     │  per round   │  │
│  ┌──────────┐                     └──────────────┘  │
│  │ client-2 │◄───────────────────►                  │
│  │ (189 pts)│                                        │
│  └──────────┘                                        │
└─────────────────────────────────────────────────────┘
```

### DP-SGD — the privacy mechanism

Each client applies DP-SGD before sending its update:

```
Per mini-batch:
  1. Compute per-sample gradients  g_i   (not batch-averaged)
  2. Clip each:   g_i ← g_i / max(1, ‖g_i‖₂ / C)     [bounds sensitivity]
  3. Sum + noise: g̃  = Σ clip(g_i) + 𝒩(0, (Cσ)²I)    [Gaussian mechanism]
  4. Average:     ḡ  = g̃ / batch_size
```

**Key parameters** (set via environment variables):

| Param | Default | Meaning |
|-------|---------|---------|
| `CLIP_NORM` (C) | 1.0 | Max L2 norm of any per-sample gradient — bounds sensitivity |
| `NOISE_MULTIPLIER` (σ) | 1.1 | Noise scale; larger = more private, less accurate |
| `DELTA` (δ) | 1e-5 | Failure probability for (ε, δ)-DP guarantee |

### Run it

```bash
cd fl_dp
docker compose up --build

# Watch client logs
docker compose logs -f client-0

# Check server metrics after training
curl http://localhost:5000/metrics | python3 -m json.tool
```

### Sample output

```
[Client 0] Round 1  — local acc: 0.7368 | ε spent so far: 0.21 (δ=1e-05)
[Client 0] Round 5  — local acc: 0.8684 | ε spent so far: 1.07 (δ=1e-05)
[Client 0] Round 10 — local acc: 0.9211 | ε spent so far: 2.14 (δ=1e-05)
[SERVER]   Round 10 complete — avg local acc: 0.9035
```

### Privacy-utility trade-off

Run with different `NOISE_MULTIPLIER` values to see the accuracy cost of stronger privacy:

| σ (noise_multiplier) | ε after 10 rounds | Final accuracy |
|----------------------|-------------------|----------------|
| 0.5  | ~4.3 | ~0.94 |
| 1.1  | ~2.1 | ~0.90 |
| 2.0  | ~1.1 | ~0.85 |

---

## Homomorphic Encryption Demo

An analytics server computes the mean BMI across 100 encrypted patient records without ever seeing a plaintext value.

Uses the **CKKS scheme** (approximate arithmetic, real numbers) via [TenSEAL](https://github.com/OpenMined/TenSEAL).

```bash
pip install tenseal numpy
python demos/homomorphic_encryption.py
```

```
[Hospital] Encrypting 100 values ... 3.9 ms
[Server]   Computing sum on ciphertext ... 21.4 ms
           N=100 is public info; dividing by N is a plaintext op.
[Hospital] Decrypting result ... 0.9 ms

True mean (plaintext):   27.032691
HE result (decrypted):   27.032572
Absolute error:          1.19e-04   ← CKKS is approximate
```

**Key insight:** The server computed an encrypted sum over 100 ciphertexts using SIMD rotations (Galois keys). Division by N happens on the plaintext side after decryption — N is public information, so no privacy is lost. The server saw zero plaintext data.

---

## SMPC Demo

Three hospitals compute the total dengue case count across their districts using **additive secret sharing** — no hospital reveals its count to any other.

```bash
python demos/smpc.py
```

```
NIMHANS (Bengaluru):   1,847 cases  ← private
KMC (Manipal):           923 cases  ← private
PGIMER (Chandigarh):   2,104 cases  ← private

Reconstructed total:   4,874  ✓ correct
```

Also includes a sketch of **Private Set Intersection (PSI)**: hospitals find how many patients appear in both their records without revealing who.

---

## How these PETs compose (CDPG context)

```
Data contributor proves eligibility     → ZKP (see HIDE project)
Data never leaves the source institution → Federated Learning
Gradients don't leak training data       → DP-SGD (this repo)
Analytics server computes on ciphertext  → Homomorphic Encryption
Cross-institution exact computation      → SMPC
Published statistics protect individuals → Differential Privacy output layer
```

Each layer adds a specific privacy guarantee. The modular design means they can be swapped or combined based on the sensitivity of the data and the threat model.

---

## Related work

- HIDE: Groth16 zk-SNARK framework for anonymous email (M.Tech thesis, ACM CCS 2026 submission)
- Abadi et al., "Deep Learning with Differential Privacy", CCS 2016
- McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data", AISTATS 2017
- Cheon et al., "Homomorphic Encryption for Arithmetic of Approximate Numbers", ASIACRYPT 2017
