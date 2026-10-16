// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

/**
 * @title AnonymousProver
 * @notice On-chain Groth16 verifier for anonymous set-membership proofs.
 *
 * Use-case: a user proves they belong to an authorised set (stored as a
 * Sparse Merkle Tree root on-chain) WITHOUT revealing which member they are.
 * A per-action nullifier prevents the same identity from acting twice.
 *
 * This mirrors the HIDE (Header-Identity Decoupling Engine) architecture:
 * - A Sparse Merkle Tree root represents the set of authorised identities.
 * - A Groth16 proof (BN128 curve) proves membership without disclosing it.
 * - A nullifier hash prevents replay / double-action by the same member.
 *
 * Circuit public inputs:
 *   [0] merkleRoot     — root of the on-chain identity tree
 *   [1] nullifierHash  — H(identity_secret || external_nullifier)
 *
 * Circuit private witness (never on-chain):
 *   identity_secret, merkle_path_siblings[], merkle_path_indices[]
 *
 * ── Toolchain to generate real proofs ───────────────────────────────────
 *   circom membership.circom --r1cs --wasm --sym
 *   snarkjs groth16 setup membership.r1cs pot15_final.ptau membership_0.zkey
 *   snarkjs zkey contribute membership_0.zkey membership_final.zkey
 *   snarkjs zkey export solidityverifier membership_final.zkey Verifier.sol
 *   snarkjs groth16 prove membership_final.zkey witness.wtns proof.json public.json
 *   snarkjs groth16 verify verification_key.json public.json proof.json
 *
 * The VK constants below are structural PLACEHOLDERS — replace every
 * uint256 literal with the hex values from your trusted setup output.
 * ────────────────────────────────────────────────────────────────────────
 * Curve:      BN128 (alt_bn128)
 * Precompiles used:
 *   0x06 — ecAdd   (G1 point addition)
 *   0x07 — ecMul   (G1 scalar multiplication)
 *   0x08 — ecPairing (BN128 pairing check)
 */

// ── BN128 pairing library ─────────────────────────────────────────────────
library Pairing {
    /// Field prime for BN128
    uint256 constant FIELD_PRIME =
        21888242871839275222246405745257275088696311157297823662689037894645226208583;

    struct G1Point {
        uint256 X;
        uint256 Y;
    }

    /// G2 point: coordinates are elements of F_p2 (extension field)
    struct G2Point {
        uint256[2] X; // [x.c1, x.c0]  (imaginary, real)
        uint256[2] Y; // [y.c1, y.c0]
    }

    /// Negate a G1 point (additive inverse)
    function negate(G1Point memory p) internal pure returns (G1Point memory) {
        if (p.X == 0 && p.Y == 0) return G1Point(0, 0);
        return G1Point(p.X, FIELD_PRIME - (p.Y % FIELD_PRIME));
    }

    /// BN128 G1 point addition (precompile 0x06)
    function addition(G1Point memory p1, G1Point memory p2)
        internal view returns (G1Point memory r)
    {
        uint256[4] memory input = [p1.X, p1.Y, p2.X, p2.Y];
        bool ok;
        assembly { ok := staticcall(gas(), 0x06, input, 0x80, r, 0x40) }
        require(ok, "BN128 ecAdd failed");
    }

    /// BN128 G1 scalar multiplication (precompile 0x07)
    function scalar_mul(G1Point memory p, uint256 s)
        internal view returns (G1Point memory r)
    {
        uint256[3] memory input = [p.X, p.Y, s];
        bool ok;
        assembly { ok := staticcall(gas(), 0x07, input, 0x60, r, 0x40) }
        require(ok, "BN128 ecMul failed");
    }

    /**
     * @dev BN128 multi-pairing check (precompile 0x08).
     * Returns true iff ∏ e(p1[i], p2[i]) == 1  in GT.
     * Input format per pair: [G1.x, G1.y, G2.x1, G2.x0, G2.y1, G2.y0]
     */
    function pairing(G1Point[] memory p1, G2Point[] memory p2)
        internal view returns (bool)
    {
        require(p1.length == p2.length, "Pairing: length mismatch");
        uint256 n = p1.length;
        uint256[] memory input = new uint256[](n * 6);
        for (uint256 i = 0; i < n; i++) {
            input[i * 6]     = p1[i].X;
            input[i * 6 + 1] = p1[i].Y;
            input[i * 6 + 2] = p2[i].X[0];
            input[i * 6 + 3] = p2[i].X[1];
            input[i * 6 + 4] = p2[i].Y[0];
            input[i * 6 + 5] = p2[i].Y[1];
        }
        uint256[1] memory out;
        bool ok;
        assembly {
            ok := staticcall(gas(), 0x08, add(input, 0x20), mul(n, 0xC0), out, 0x20)
        }
        require(ok, "BN128 pairing precompile failed");
        return out[0] == 1;
    }
}

// ── Main verifier contract ────────────────────────────────────────────────
contract AnonymousProver {
    using Pairing for *;

    // ── Groth16 verification key (replace with snarkjs output) ───────────
    // These are example BN128 generator point values — not a real VK.
    // After running: snarkjs zkey export solidityverifier
    // copy the IC, alpha, beta, gamma, delta arrays here.
    struct VerifyingKey {
        Pairing.G1Point alpha;
        Pairing.G2Point beta;
        Pairing.G2Point gamma;
        Pairing.G2Point delta;
        Pairing.G1Point[] IC; // Input Commitments: IC[0] + Σ input[i]*IC[i+1]
    }

    struct Proof {
        Pairing.G1Point A;
        Pairing.G2Point B;
        Pairing.G1Point C;
    }

    // ── on-chain state ────────────────────────────────────────────────────
    bytes32 public merkleRoot;
    mapping(uint256 => bool) public nullifierUsed; // prevents replay
    address public admin;

    event ProofVerified(uint256 indexed nullifierHash, address indexed verifier);
    event MerkleRootUpdated(bytes32 oldRoot, bytes32 newRoot);

    error NullifierAlreadyUsed(uint256 nullifierHash);
    error MerkleRootMismatch(bytes32 expected, bytes32 got);
    error InvalidProof();

    modifier onlyAdmin() {
        require(msg.sender == admin, "Not admin");
        _;
    }

    constructor(bytes32 _merkleRoot) {
        merkleRoot = _merkleRoot;
        admin = msg.sender;
    }

    // ── public interface ──────────────────────────────────────────────────

    /**
     * @notice Verify an anonymous membership proof.
     * @param a       Proof.A  (G1 point)
     * @param b       Proof.B  (G2 point)
     * @param c       Proof.C  (G1 point)
     * @param input   Public inputs: [merkleRoot_as_uint, nullifierHash]
     * @return true if the proof is valid and the nullifier is fresh
     *
     * The caller proves: "I know an identity_secret whose leaf is inside
     * the Merkle tree with root `merkleRoot`, and I have not acted before
     * under nullifier H(identity_secret || external_nullifier)."
     */
    function verifyMembership(
        uint256[2]    calldata a,
        uint256[2][2] calldata b,
        uint256[2]    calldata c,
        uint256[2]    calldata input
    ) external returns (bool) {
        // Validate public inputs against on-chain state
        if (bytes32(input[0]) != merkleRoot)
            revert MerkleRootMismatch(merkleRoot, bytes32(input[0]));

        uint256 nullifierHash = input[1];
        if (nullifierUsed[nullifierHash])
            revert NullifierAlreadyUsed(nullifierHash);

        // Run Groth16 verification
        if (!_groth16Verify(a, b, c, input))
            revert InvalidProof();

        // Mark nullifier consumed
        nullifierUsed[nullifierHash] = true;
        emit ProofVerified(nullifierHash, msg.sender);
        return true;
    }

    /// @notice Allow admin to rotate the Merkle root (e.g. new members added)
    function updateMerkleRoot(bytes32 _newRoot) external onlyAdmin {
        emit MerkleRootUpdated(merkleRoot, _newRoot);
        merkleRoot = _newRoot;
    }

    // ── Groth16 core ──────────────────────────────────────────────────────

    function _verifyingKey() internal pure returns (VerifyingKey memory vk) {
        // ── REPLACE with your circuit's VK from snarkjs ──────────────────
        // Example structure (BN128 generator point used as placeholder):
        vk.alpha = Pairing.G1Point(
            0x0000000000000000000000000000000000000000000000000000000000000001, // placeholder
            0x0000000000000000000000000000000000000000000000000000000000000002  // placeholder
        );
        vk.beta = Pairing.G2Point(
            [uint256(0x00), uint256(0x00)], // placeholder
            [uint256(0x00), uint256(0x00)]
        );
        vk.gamma = Pairing.G2Point(
            [uint256(0x00), uint256(0x00)], // placeholder
            [uint256(0x00), uint256(0x00)]
        );
        vk.delta = Pairing.G2Point(
            [uint256(0x00), uint256(0x00)], // placeholder
            [uint256(0x00), uint256(0x00)]
        );
        // IC length = number of public inputs + 1
        vk.IC = new Pairing.G1Point[](3);
        vk.IC[0] = Pairing.G1Point(0x00, 0x00); // placeholder
        vk.IC[1] = Pairing.G1Point(0x00, 0x00); // placeholder
        vk.IC[2] = Pairing.G1Point(0x00, 0x00); // placeholder
    }

    /**
     * @dev Groth16 verification equation:
     *
     *   e(A, B) == e(α, β) · e(vk_x, γ) · e(C, δ)
     *
     * Rearranged (negate A so the pairing precompile checks product == 1):
     *   e(-A, B) · e(α, β) · e(vk_x, γ) · e(C, δ) == 1
     *
     * vk_x = IC[0] + Σ_i (input[i] · IC[i+1])
     */
    function _groth16Verify(
        uint256[2]    calldata a,
        uint256[2][2] calldata b,
        uint256[2]    calldata c,
        uint256[2]    calldata input
    ) internal view returns (bool) {
        VerifyingKey memory vk = _verifyingKey();
        require(vk.IC.length == input.length + 1, "IC length mismatch");

        Proof memory proof;
        proof.A = Pairing.G1Point(a[0], a[1]);
        proof.B = Pairing.G2Point([b[0][0], b[0][1]], [b[1][0], b[1][1]]);
        proof.C = Pairing.G1Point(c[0], c[1]);

        // Compute vk_x = IC[0] + input[0]*IC[1] + input[1]*IC[2]
        Pairing.G1Point memory vk_x = vk.IC[0];
        for (uint256 i = 0; i < input.length; i++) {
            vk_x = Pairing.addition(vk_x, Pairing.scalar_mul(vk.IC[i + 1], input[i]));
        }

        // Four-pairing check
        Pairing.G1Point[] memory p1 = new Pairing.G1Point[](4);
        Pairing.G2Point[] memory p2 = new Pairing.G2Point[](4);

        p1[0] = Pairing.negate(proof.A);  p2[0] = proof.B;
        p1[1] = vk.alpha;                  p2[1] = vk.beta;
        p1[2] = vk_x;                      p2[2] = vk.gamma;
        p1[3] = proof.C;                   p2[3] = vk.delta;

        return Pairing.pairing(p1, p2);
    }
}
