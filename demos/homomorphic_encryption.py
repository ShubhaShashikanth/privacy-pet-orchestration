"""
Homomorphic Encryption Demo — CKKS scheme via TenSEAL
========================================================
Scenario: A hospital sends *encrypted* patient BMI records to
an analytics server. The server computes the mean BMI without
ever seeing individual values. The hospital decrypts the result.

This uses the CKKS scheme (Cheon-Kim-Kim-Song, 2017), which
supports approximate arithmetic on real-valued vectors — ideal
for ML and statistical computations.

Install:  pip install tenseal numpy

CKKS key concepts used here:
  poly_modulus_degree : controls security level and noise budget
                        (8192 → ~128-bit security)
  coeff_mod_bit_sizes : bit sizes of coefficient moduli; the last
                        value is the scale, rest are for bootstrapping
  global_scale        : precision of the fixed-point encoding (2^26)
"""

import tenseal as ts
import numpy as np
import time


# ── 1. Key generation (hospital / data owner) ────────────────────────
def create_he_context() -> ts.Context:
    """
    Create a CKKS context.
    Equivalent to a trusted setup — whoever holds the secret key
    can decrypt; the evaluator (server) only gets the public key.
    """
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    ctx.global_scale = 2 ** 40   # matches the 40-bit middle primes
    ctx.generate_galois_keys()   # required for vector rotation (dot products)
    return ctx


# ── 2. Simulated patient data ────────────────────────────────────────
np.random.seed(42)
N_PATIENTS = 100
true_bmis  = np.random.normal(loc=27.5, scale=4.5, size=N_PATIENTS)
true_mean  = true_bmis.mean()

print("=" * 60)
print("Homomorphic Encryption — CKKS scheme (TenSEAL)")
print("=" * 60)
print(f"\nHospital has {N_PATIENTS} patient BMI records.")
print(f"True mean BMI (plaintext):  {true_mean:.4f}")


# ── 3. Encrypt (hospital side) ───────────────────────────────────────
ctx    = create_he_context()
t0     = time.perf_counter()
enc_v  = ts.ckks_vector(ctx, true_bmis.tolist())
t_enc  = time.perf_counter() - t0

print(f"\n[Hospital] Encrypting {N_PATIENTS} values ... {t_enc*1000:.1f} ms")
print("[Hospital] Sending encrypted vector to analytics server.")
print("           ⚠  Server receives ciphertext — no plaintext visible.")


# What actually crosses the wire: a public-only context (no secret key)
# and the serialized ciphertext.
public_ctx_bytes = ctx.serialize(save_secret_key=False)
ciphertext_bytes = enc_v.serialize()


# ── 4. Compute on encrypted data (server side) ───────────────────────
# The server rebuilds everything from bytes, so it holds only the public
# key, relinearisation and Galois keys — never the secret key.
server_ctx = ts.context_from(public_ctx_bytes)
assert not server_ctx.has_secret_key()
server_vec = ts.ckks_vector_from(server_ctx, ciphertext_bytes)

t0 = time.perf_counter()

# Compute the encrypted sum of all elements using SIMD rotations.
# TenSEAL's .sum() uses the Galois keys to rotate the ciphertext vector
# and accumulate — no decryption needed.
server_sum = server_vec.sum()            # homomorphic sum (still encrypted)

t_compute = time.perf_counter() - t0
print(f"\n[Server]   Computing sum on ciphertext ... {t_compute*1000:.1f} ms")

try:                                     # prove the server cannot decrypt
    server_sum.decrypt()
    print("[Server]   ✗ unexpectedly able to decrypt!")
except Exception:
    print("[Server]   Tried to decrypt → refused: no secret key on the server.")

print("[Server]   Result is still encrypted.  Sending back to hospital.")
print(f"           N={N_PATIENTS} is public info; dividing by N is a plaintext op.")
result_bytes = server_sum.serialize()


# ── 5. Decrypt (hospital side, with secret key) ──────────────────────
t0             = time.perf_counter()
enc_sum        = ts.ckks_vector_from(ctx, result_bytes)
decrypted_sum  = enc_sum.decrypt()[0]
decrypted_mean = decrypted_sum / N_PATIENTS   # N is public — no privacy loss
t_dec          = time.perf_counter() - t0

print(f"\n[Hospital] Decrypting result ... {t_dec*1000:.1f} ms")
print(f"\nResults:")
print(f"  True mean (plaintext):   {true_mean:.6f}")
print(f"  HE result (decrypted):   {decrypted_mean:.6f}")
print(f"  Absolute error:          {abs(decrypted_mean - true_mean):.2e}   ← CKKS is approximate")


# ── 6. Show privacy guarantee ────────────────────────────────────────
print("\n" + "=" * 60)
print("Privacy guarantee:")
print("  The analytics server performed the mean computation on")
print("  100 encrypted values.  It never held the secret key and")
print("  never saw any plaintext BMI value — only ciphertexts.")
print("  This is the core HE promise: compute without decrypting.")
print()
print("Scheme:  CKKS (approximate arithmetic, real numbers)")
print("Security: ~128-bit (poly_modulus_degree=8192)")
print("Overhead: CKKS encryption is ~1000x slower than plaintext")
print("          arithmetic — best for offline batch analytics.")
print("=" * 60)
