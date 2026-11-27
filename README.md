# Privacy-PET Orchestration

A hands-on implementation of multiple Privacy-Enhancing Technologies (PETs) — built to demonstrate how DP, FL, HE, and SMPC compose into a privacy-preserving data pipeline.

> **Motivation:** CDPG's mission is building a modular orchestration layer that combines multiple PETs. This repo demonstrates each layer independently and shows how they interlock.

---

## What's inside

| Module | PET | Key technique | Status |
|--------|-----|---------------|--------|
| `fl_dp/` | Federated Learning + Differential Privacy | DP-SGD (Poisson sampling, gradient clipping, Gaussian noise), RDP privacy accounting, FedAvg | Dockerised, runnable |
| `demos/homomorphic_encryption.py` | Homomorphic Encryption | CKKS scheme via TenSEAL — mean on encrypted health data | Standalone script |
| `demos/smpc.py` | Secure Multi-Party Computation | Additive secret sharing + Diffie-Hellman PSI cardinality | Standalone script |

---

## FL + DP-SGD — Containerised Demo

Three hospital "clients" train a logistic regression model on disjoint partitions of the [Breast Cancer Wisconsin dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset) — a proxy for real health data.

**Raw records never leave a client.** Only model weights trained with DP-SGD cross the network. (In this simulation each container loads the public dataset and keeps only its own slice; feature scaling is fitted on each client's own data.)

### Architecture

```
┌─────────────────────────────────────────────────────┐
│  docker-compose network                              │
│                                                      │
│  ┌──────────┐     GET /model      ┌──────────────┐  │
│  │ client-0 │◄───────────────────►│              │  │
│  │ (190 pts)│  POST /update       │    server    │  │
│  └──────────┘  (DP-SGD weights)   │  (FedAvg)    │  │
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
Per step:
  1. Sample batch: each record joins independently w.p. q = B / n   [Poisson sampling]
  2. Compute per-sample gradients  g_i   (not batch-averaged)
  3. Clip each:   g_i ← g_i / max(1, ‖g_i‖₂ / C)     [bounds sensitivity]
  4. Sum + noise: g̃  = Σ clip(g_i) + 𝒩(0, (Cσ)²I)    [Gaussian mechanism]
  5. Average:     ḡ  = g̃ / (q·n)                     [expected batch size]
```

**Privacy accounting.** Each step is a Poisson-subsampled Gaussian mechanism. All steps across all rounds are composed with a Rényi-DP accountant ([Google `dp-accounting`](https://github.com/google/differential-privacy/tree/main/python)) and converted to an (ε, δ) guarantee. The ε printed by each client is record-level DP for that client's local dataset. Poisson sampling is used (rather than shuffling into fixed batches) because that is what the accountant's privacy-amplification analysis assumes.

**Key parameters** (set via environment variables):

| Param | Default | Meaning |
|-------|---------|---------|
| `CLIP_NORM` (C) | 1.0 | Max L2 norm of any per-sample gradient — bounds sensitivity |
| `NOISE_MULTIPLIER` (σ) | 3.0 | Noise scale; larger = more private, less accurate |
| `BATCH_SIZE` (B) | 16 | Expected batch size; sampling rate q = B / n |
| `LOCAL_EPOCHS` | 3 | Local DP-SGD epochs per federated round |
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

Actual output from `docker compose up --build` with the defaults (accuracy varies slightly run to run because of the DP noise):

```
[Client 0] Dataset: 152 train / 38 test samples
[Client 0] Round 1 — local acc: 0.8684 | ε spent so far: 0.89 (δ=1e-05)
[Client 0] Round 5 — local acc: 0.8947 | ε spent so far: 1.99 (δ=1e-05)
[Client 0] Round 10 — local acc: 0.9737 | ε spent so far: 2.88 (δ=1e-05)
[SERVER] Round 10 complete — avg local acc: 0.9561
```

### Privacy-utility trade-off

Run with different `NOISE_MULTIPLIER` values (B = 16, 3 local epochs, 10 rounds, δ = 1e-5, 152 records per client). ε is from the RDP accountant; accuracy is the mean over 5 seeds:

| σ (noise_multiplier) | ε after 10 rounds | Final accuracy |
|----------------------|-------------------|----------------|
| 0.5  | 67.6 | 0.94 |
| 1.1  | 12.2 | 0.94 |
| 2.0  | 4.8  | 0.93 |
| 3.0  | 2.9  | 0.93 |

With only ~150 records per client, every step spends a noticeable share of the budget, so a fairly large σ is needed to reach single-digit ε. Logistic regression on this dataset is easy, so accuracy barely drops; on harder tasks the trade-off is much steeper.

---

## Homomorphic Encryption Demo

An analytics server computes the mean BMI across 100 encrypted patient records without ever seeing a plaintext value.

Uses the **CKKS scheme** (approximate arithmetic, real numbers) via [TenSEAL](https://github.com/OpenMined/TenSEAL).

```bash
pip install tenseal numpy
python demos/homomorphic_encryption.py
```

```
[Hospital] Encrypting 100 values ... 5.0 ms
[Server]   Computing sum on ciphertext ... 28.0 ms
[Server]   Tried to decrypt → refused: no secret key on the server.
           N=100 is public info; dividing by N is a plaintext op.
[Hospital] Decrypting result ... 2.0 ms

True mean (plaintext):   27.032691
HE result (decrypted):   27.032691
Absolute error:          4.70e-08   ← CKKS is approximate
```

**Key insight:** The server receives only serialized bytes — the ciphertext and a context with the secret key stripped (`save_secret_key=False`). It computed an encrypted sum over 100 ciphertexts using SIMD rotations (Galois keys). Division by N happens on the plaintext side after decryption — N is public information, so no privacy is lost. The server saw zero plaintext data.

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

Also includes **Private Set Intersection cardinality** using the Diffie-Hellman commutative-blinding protocol over the RFC 3526 2048-bit group: two hospitals learn *how many* patients they share without revealing their patient lists. Semi-honest security under the DDH assumption; malicious-party protections are out of scope for this demo.

```
PSI result: 3 common patients  (✓ matches plaintext check: 3)
A learns only the overlap COUNT; B learns only |A|.
```

---

## How these PETs compose 

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
