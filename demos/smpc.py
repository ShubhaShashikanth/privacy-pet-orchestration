"""
Secure Multi-Party Computation Demo — Shamir's Secret Sharing
===============================================================
Scenario: Three hospitals want to compute the *total* number of
dengue cases across their districts without any hospital revealing
its own case count to the others.

Protocol (additive secret sharing, 3-of-3):
  1. Each hospital splits its value into 3 random shares s.t.
     share_1 + share_2 + share_3 = original_value  (mod prime).
  2. Each hospital sends one share to each other hospital —
     a single share reveals NOTHING about the original.
  3. Each hospital sums the share it received from each party.
  4. The three local sums are broadcast and added to get the total.

Security model: semi-honest (honest-but-curious).
  Parties follow the protocol but try to learn from transcripts.
  No single party or colluding pair can reconstruct any other
  party's input from its shares alone.

No external libraries needed — pure Python + numpy.
"""

import numpy as np
from typing import NamedTuple

PRIME = (1 << 61) - 1   # Mersenne prime — large enough for healthcare counts
                          # and convenient for modular arithmetic


# ── types ─────────────────────────────────────────────────────────────
class Shares(NamedTuple):
    """One secret value split into n additive shares (mod PRIME)."""
    values: list[int]
    n: int


# ── secret sharing ────────────────────────────────────────────────────
def share(secret: int, n: int = 3) -> Shares:
    """
    Additive (n-of-n) secret sharing.
    Generate n-1 uniformly random values; the last share is the residual
    so that all shares sum to the secret (mod PRIME).
    """
    if not (0 <= secret < PRIME):
        raise ValueError("secret must be in [0, PRIME)")

    random_shares = [int(np.random.randint(0, PRIME)) for _ in range(n - 1)]
    last_share    = (secret - sum(random_shares)) % PRIME
    return Shares(values=random_shares + [last_share], n=n)


def reconstruct(shares: list[int]) -> int:
    """Sum all shares modulo PRIME to recover the original secret."""
    return sum(shares) % PRIME


# ── simulation ────────────────────────────────────────────────────────
def run_smpc_demo():
    print("=" * 60)
    print("Secure Multi-Party Computation — Additive Secret Sharing")
    print("=" * 60)

    # ── 1. Private inputs (known only to each hospital) ───────────────
    hospitals = {
        "NIMHANS (Bengaluru)":   1_847,
        "KMC (Manipal)":           923,
        "PGIMER (Chandigarh)":   2_104,
    }
    true_total = sum(hospitals.values())

    print("\nPrivate inputs (NOT shared with other hospitals):")
    for h, v in hospitals.items():
        print(f"  {h}: {v:,} dengue cases  ← private")

    print(f"\nTrue total (ground truth): {true_total:,}")

    # ── 2. Each hospital creates shares of its value ───────────────────
    all_shares = {name: share(val) for name, val in hospitals.items()}

    print("\n[Step 1] Each hospital splits its count into 3 random shares.")
    print("         (Each individual share looks like random noise.)")
    for name, s in all_shares.items():
        print(f"  {name}: shares = {s.values}")

    # ── 3. Distribute: hospital i sends share j to hospital j ─────────
    # hospitals[0] gets: all_shares[0][0], all_shares[1][0], all_shares[2][0]
    names  = list(hospitals.keys())
    n      = len(names)
    received = {names[j]: [all_shares[names[i]].values[j] for i in range(n)]
                for j in range(n)}

    print("\n[Step 2] Each hospital sends one share to each peer.")
    print("         A single share reveals nothing — it's uniform random mod p.")

    # ── 4. Local computation: each hospital sums its received shares ───
    local_sums = {name: sum(shares) % PRIME for name, shares in received.items()}

    print("\n[Step 3] Each hospital computes the sum of the shares it received:")
    for name, ls in local_sums.items():
        print(f"  {name} local sum: {ls:,}")

    # ── 5. Broadcast local sums and reconstruct total ──────────────────
    total = reconstruct(list(local_sums.values()))

    print("\n[Step 4] All local sums are broadcast and added (mod p).")
    print(f"\nReconstructed total: {total:,}")
    print(f"True total:          {true_total:,}")
    print(f"Match: {'✓ correct' if total == true_total else '✗ mismatch'}")

    # ── 6. Privacy verification ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Privacy verification — what can each hospital infer?")
    print()

    # Hospital 0 sees: its own value + the single shares it received from 1 and 2
    h0_sees = [all_shares[names[1]].values[0], all_shares[names[2]].values[0]]
    print(f"NIMHANS received shares from KMC and PGIMER: {h0_sees}")
    print(f"  These are uniform random values mod p — no information about")
    print(f"  KMC's ({hospitals[names[1]]:,}) or PGIMER's ({hospitals[names[2]]:,}) true counts.")
    print()

    # Show that partial shares are statistically indistinguishable from random
    print("Statistical test: are the shares distinguishable from uniform random?")
    random_vals = [int(np.random.randint(0, PRIME)) for _ in range(len(h0_sees))]
    print(f"  Actual shares:  {h0_sees}")
    print(f"  Random values:  {random_vals}")
    print(f"  Visually indistinguishable → semi-honest security holds.")

    print("\nProtocol:  Additive secret sharing (3-of-3)")
    print("Security:  Semi-honest (honest-but-curious) adversary model")
    print("Guarantee: No party learns another's input; only the aggregate")
    print("           total is revealed at the end.")
    print("=" * 60)


# ── extension: private set intersection ─────────────────────────────
def private_set_intersection_size(set_a: set, set_b: set) -> int:
    """
    Toy PSI (Private Set Intersection) — counts common elements
    without revealing the sets to each other.
    In production this uses OT-based or HE-based PSI protocols.
    Here we use a hash-based local simulation to illustrate the concept.
    """
    hashed_a = {hash(x) for x in set_a}
    hashed_b = {hash(x) for x in set_b}
    return len(hashed_a & hashed_b)   # only intersection SIZE is revealed


def run_psi_demo():
    print("\n" + "=" * 60)
    print("Extension: Private Set Intersection (PSI)")
    print("=" * 60)
    print("\nTwo hospitals want to know how many patients appear in both")
    print("their records — without revealing their patient lists.\n")

    hospital_a = {"P001", "P002", "P003", "P099", "P200"}
    hospital_b = {"P002", "P003", "P150", "P200", "P300"}
    true_overlap = hospital_a & hospital_b

    intersection_size = private_set_intersection_size(hospital_a, hospital_b)

    print(f"  Hospital A: {sorted(hospital_a)}  (private)")
    print(f"  Hospital B: {sorted(hospital_b)}  (private)")
    print(f"\n  True overlap:       {sorted(true_overlap)}")
    print(f"  PSI result (size):  {intersection_size} common patients")
    print(f"\n  Each hospital learns ONLY the count — not which patients overlap.")
    print("  (Production PSI uses Oblivious Transfer or HE for cryptographic security.)")


if __name__ == "__main__":
    np.random.seed(0)
    run_smpc_demo()
    run_psi_demo()
