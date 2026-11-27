"""
Secure Multi-Party Computation Demo — Additive Secret Sharing + PSI
====================================================================
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
  Any n-1 shares of a value are independent and uniformly random,
  so no single party (or colluding pair) learns another party's
  input from the shares. Only the final total is revealed — and,
  as with any aggregate, colluders can subtract their own inputs
  from it.

Shares are drawn with Python's `secrets` module (a CSPRNG); a
non-cryptographic RNG such as numpy's would make shares predictable.

No external libraries needed — pure Python standard library.
"""

import hashlib
import random
import secrets
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

    random_shares = [secrets.randbelow(PRIME) for _ in range(n - 1)]
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
    random_vals = [secrets.randbelow(PRIME) for _ in range(len(h0_sees))]
    print(f"  Actual shares:  {h0_sees}")
    print(f"  Random values:  {random_vals}")
    print(f"  Each share is uniform mod p by construction, so the two lists")
    print(f"  have exactly the same distribution — that is the security argument.")

    print("\nProtocol:  Additive secret sharing (3-of-3)")
    print("Security:  Semi-honest (honest-but-curious) adversary model")
    print("Guarantee: No party learns another's input; only the aggregate")
    print("           total is revealed at the end.")
    print("=" * 60)


# ── extension: private set intersection (cardinality) ───────────────
#
# Diffie-Hellman PSI-cardinality (Meadows 1986; Huberman et al. 1999;
# the same idea underlies Google's Private Join and Compute).
#
#   Group: quadratic residues modulo the RFC 3526 2048-bit safe prime p
#          (prime order q = (p-1)/2), where DDH is believed hard.
#   H(x):  hash the identifier to 2048 bits, reduce mod p, square
#          → a pseudo-random element of the QR subgroup.
#
#   1. A picks secret α, sends B the shuffled set { H(a)^α }.
#   2. B picks secret β, returns the shuffled set { (H(a)^α)^β }
#      and also sends { H(b)^β }.
#   3. A computes { (H(b)^β)^α } and counts matches with step 2.
#      Since exponentiation commutes, H(x)^(αβ) matches iff x is in both.
#
#   Semi-honest security under DDH: A learns only |A ∩ B| (B's shuffle
#   hides *which* of A's elements matched); B learns only |A|.
#   Not covered here: malicious parties, or limiting what B can learn
#   by choosing which identifiers to test — production systems add
#   those protections.

RFC3526_P = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74"
    "020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F1437"
    "4FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF05"
    "98DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB"
    "9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
    "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF695581718"
    "3995497CEA956AE515D2261898FA051015728E5A8AACAA68FFFFFFFFFFFFFFFF",
    16,
)
Q = (RFC3526_P - 1) // 2


def hash_to_group(x: str) -> int:
    digest = hashlib.shake_256(x.encode()).digest(256)        # 2048 bits
    return pow(int.from_bytes(digest, "big") % RFC3526_P, 2, RFC3526_P)


def blind(elements, key: int) -> list[int]:
    """Raise every group element to the secret key, then shuffle."""
    out = [pow(e, key, RFC3526_P) for e in elements]
    random.SystemRandom().shuffle(out)
    return out


def psi_cardinality(set_a: set, set_b: set) -> int:
    """Run both parties' steps of DH PSI-cardinality; returns |A ∩ B|."""
    alpha = 1 + secrets.randbelow(Q - 1)        # A's secret
    beta  = 1 + secrets.randbelow(Q - 1)        # B's secret

    a_to_b   = blind([hash_to_group(x) for x in set_a], alpha)   # step 1 (A → B)
    a_double = blind(a_to_b, beta)                               # step 2 (B → A)
    b_to_a   = blind([hash_to_group(x) for x in set_b], beta)    # step 2 (B → A)
    b_double = blind(b_to_a, alpha)                              # step 3 (at A)

    return len(set(a_double) & set(b_double))


def run_psi_demo():
    print("\n" + "=" * 60)
    print("Extension: Private Set Intersection cardinality (DH-PSI)")
    print("=" * 60)
    print("\nTwo hospitals want to know how many patients appear in both")
    print("their records — without revealing their patient lists.\n")

    hospital_a = {"P001", "P002", "P003", "P099", "P200"}
    hospital_b = {"P002", "P003", "P150", "P200", "P300"}

    size = psi_cardinality(hospital_a, hospital_b)
    expected = len(hospital_a & hospital_b)

    print(f"  Hospital A holds {len(hospital_a)} IDs, Hospital B holds {len(hospital_b)} IDs (private).")
    print("  Messages exchanged: blinded 2048-bit group elements only.")
    print(f"\n  PSI result: {size} common patients  "
          f"({'✓ matches' if size == expected else '✗ differs from'} plaintext check: {expected})")
    print("\n  A learns only the overlap COUNT; B learns only |A|.")
    print("  Security: semi-honest, under the Decisional Diffie-Hellman assumption.")


if __name__ == "__main__":
    run_smpc_demo()
    run_psi_demo()
