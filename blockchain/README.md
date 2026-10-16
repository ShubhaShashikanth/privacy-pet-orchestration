# Blockchain & On-Chain Privacy

Smart contract implementations of privacy-preserving protocols on Ethereum (Solidity 0.8+).

| Module | Pattern | Key technique |
|--------|---------|---------------|
| `zkp_anonymous_proof/` | Anonymous membership | Groth16 on-chain verifier, BN128 pairings, nullifier |
| `commit_reveal_vote/`  | Private ballot | Commit-reveal scheme, keccak256 commitments |

---

## zkp_anonymous_proof — Groth16 On-Chain Verifier

Proves set membership **without revealing identity**.

A user holds an identity secret. An admin publishes a Sparse Merkle Tree root of all authorised members. The user generates a Groth16 proof (off-chain, using circom + snarkjs) that their leaf is inside the tree — without revealing which leaf. The on-chain contract verifies the proof using BN128 pairing precompiles and marks a **nullifier** so the same identity cannot act twice.

**Same curve, same proof system as HIDE** (BN128 / Groth16). This contract is the on-chain equivalent of HIDE's email authentication layer.

→ [Details & setup](zkp_anonymous_proof/README.md)

---

## commit_reveal_vote — Privacy-Preserving Ballot

Prevents vote observation and strategic late voting in on-chain elections.

Voters submit `keccak256(choice || secret || address)` during the commit phase — all votes are hidden. After the commit deadline, they reveal their plaintext choice; the contract verifies and counts. Commitments are locked before any reveal is accepted, so no one can see results early or change their vote strategically.

→ [Details, Remix demo & security analysis](commit_reveal_vote/README.md)

---

## How blockchain privacy connects to the rest of this repo

```
ZKP membership proof (this folder)   → proves eligibility without identity disclosure
Federated Learning (fl_dp/)          → keeps raw data at the source institution  
DP-SGD (fl_dp/)                      → prevents gradient inversion on model updates
Homomorphic Encryption (demos/)      → server computes on ciphertext, sees no plaintext
SMPC (demos/)                        → aggregate across parties without sharing inputs
```

Each layer closes a different attack surface. The blockchain layer adds **verifiability** — any party can check that the computation was authorised without a trusted intermediary.
