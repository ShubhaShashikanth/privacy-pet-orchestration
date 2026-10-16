// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

/**
 * @title CommitRevealVote
 * @notice Privacy-preserving on-chain vote using a commit-reveal scheme.
 *
 * Problem with naive on-chain voting: Ethereum is public.
 * A voter who sends `vote(candidateId)` reveals their choice immediately.
 * Later voters can see early results and change their vote strategically.
 *
 * Solution — commit-reveal in two phases:
 *
 *   Phase 1 COMMIT  (deadline: commitEnd)
 *     Voter submits: keccak256(abi.encodePacked(choice, secret, msg.sender))
 *     On-chain only stores the hash — no one knows the actual vote.
 *
 *   Phase 2 REVEAL  (deadline: revealEnd)
 *     Voter submits their plaintext choice + secret.
 *     Contract re-hashes and checks against the stored commitment.
 *     Only then is the vote counted.
 *
 * Privacy guarantee: during the commit phase, all votes look like random
 * 32-byte hashes. Neither other voters nor the contract admin can determine
 * anyone's choice until the reveal phase — at which point all commitments
 * are already locked in, so late reveals cannot be strategically withheld
 * to swing the outcome without losing your committed slot.
 *
 * Security notes:
 *   - The secret must be kept private by the voter off-chain.
 *   - A weak secret (e.g. "password123") allows brute-force preimage attacks.
 *     Recommend: secret = random 32-byte value generated client-side.
 *   - Voters who commit but do not reveal are counted as abstentions.
 *   - msg.sender is mixed into the commitment to prevent commitment stealing.
 */
contract CommitRevealVote {

    // ── types ─────────────────────────────────────────────────────────────
    struct Candidate {
        string name;
        uint256 voteCount;
    }

    struct VoterRecord {
        bytes32 commitment;   // keccak256(choice || secret || voter_address)
        bool    hasCommitted;
        bool    hasRevealed;
    }

    // ── state ─────────────────────────────────────────────────────────────
    address public admin;

    uint256 public commitEnd;  // unix timestamp: commit phase closes
    uint256 public revealEnd;  // unix timestamp: reveal phase closes

    Candidate[] public candidates;
    mapping(address => VoterRecord) public voters;
    mapping(address => bool) public registered; // whitelist (optional)

    bool public whitelistEnabled;

    uint256 public totalCommits;
    uint256 public totalReveals;

    // ── events ────────────────────────────────────────────────────────────
    event Committed(address indexed voter);
    event Revealed(address indexed voter, uint8 choice);
    event WinnerDeclared(uint8 indexed candidateId, string name, uint256 votes);

    // ── errors ────────────────────────────────────────────────────────────
    error NotInCommitPhase();
    error NotInRevealPhase();
    error AlreadyCommitted();
    error AlreadyRevealed();
    error NoCommitmentFound();
    error InvalidReveal();
    error InvalidCandidate(uint8 choice, uint256 numCandidates);
    error NotRegistered(address voter);
    error VotingStillOpen();

    // ── constructor ───────────────────────────────────────────────────────
    /**
     * @param _candidateNames  e.g. ["Alice", "Bob", "Carol"]
     * @param _commitDuration  seconds from now until commit phase ends
     * @param _revealDuration  seconds from commitEnd until reveal phase ends
     * @param _whitelistEnabled  if true, only pre-registered addresses may vote
     */
    constructor(
        string[] memory _candidateNames,
        uint256 _commitDuration,
        uint256 _revealDuration,
        bool    _whitelistEnabled
    ) {
        require(_candidateNames.length >= 2, "Need at least 2 candidates");
        require(_commitDuration > 0 && _revealDuration > 0, "Durations must be > 0");

        admin            = msg.sender;
        commitEnd        = block.timestamp + _commitDuration;
        revealEnd        = commitEnd + _revealDuration;
        whitelistEnabled = _whitelistEnabled;

        for (uint256 i = 0; i < _candidateNames.length; i++) {
            candidates.push(Candidate({name: _candidateNames[i], voteCount: 0}));
        }
    }

    // ── voter registration (optional whitelist) ───────────────────────────
    function registerVoter(address _voter) external {
        require(msg.sender == admin, "Only admin");
        registered[_voter] = true;
    }

    function registerVotersBatch(address[] calldata _voters) external {
        require(msg.sender == admin, "Only admin");
        for (uint256 i = 0; i < _voters.length; i++) {
            registered[_voters[i]] = true;
        }
    }

    // ── phase 1: commit ───────────────────────────────────────────────────

    /**
     * @notice Submit a blinded commitment to your vote.
     * @param _commitment  keccak256(abi.encodePacked(choice, secret, msg.sender))
     *
     * Generate off-chain:
     *   choice  = uint8(candidateId)   e.g. 0, 1, 2
     *   secret  = ethers.utils.randomBytes(32)
     *   commitment = ethers.utils.solidityKeccak256(
     *                  ["uint8","bytes32","address"],
     *                  [choice, secret, voterAddress])
     */
    function commit(bytes32 _commitment) external {
        if (block.timestamp >= commitEnd) revert NotInCommitPhase();
        if (whitelistEnabled && !registered[msg.sender]) revert NotRegistered(msg.sender);

        VoterRecord storage v = voters[msg.sender];
        if (v.hasCommitted) revert AlreadyCommitted();

        v.commitment   = _commitment;
        v.hasCommitted = true;
        totalCommits++;

        emit Committed(msg.sender);
    }

    /// @notice Convenience: compute a commitment hash (call off-chain via eth_call)
    function computeCommitment(uint8 _choice, bytes32 _secret)
        external view returns (bytes32)
    {
        return keccak256(abi.encodePacked(_choice, _secret, msg.sender));
    }

    // ── phase 2: reveal ───────────────────────────────────────────────────

    /**
     * @notice Reveal your vote. Must match the earlier commitment exactly.
     * @param _choice  the candidate index you committed to
     * @param _secret  the random secret you used when committing
     */
    function reveal(uint8 _choice, bytes32 _secret) external {
        if (block.timestamp < commitEnd)  revert NotInCommitPhase();  // commit still open
        if (block.timestamp >= revealEnd) revert NotInRevealPhase();  // reveal closed

        if (_choice >= candidates.length)
            revert InvalidCandidate(_choice, candidates.length);

        VoterRecord storage v = voters[msg.sender];
        if (!v.hasCommitted) revert NoCommitmentFound();
        if (v.hasRevealed)   revert AlreadyRevealed();

        // Recompute and compare — this is the integrity check
        bytes32 expected = keccak256(abi.encodePacked(_choice, _secret, msg.sender));
        if (expected != v.commitment) revert InvalidReveal();

        v.hasRevealed = true;
        candidates[_choice].voteCount++;
        totalReveals++;

        emit Revealed(msg.sender, _choice);
    }

    // ── results ───────────────────────────────────────────────────────────

    /// @notice Returns all candidates with their vote counts (only after reveal ends)
    function results()
        external view
        returns (string[] memory names, uint256[] memory votes)
    {
        if (block.timestamp < revealEnd) revert VotingStillOpen();

        names = new string[](candidates.length);
        votes = new uint256[](candidates.length);
        for (uint256 i = 0; i < candidates.length; i++) {
            names[i] = candidates[i].name;
            votes[i] = candidates[i].voteCount;
        }
    }

    /// @notice Declare and emit the winner (only after reveal phase ends)
    function declareWinner() external returns (uint8 winnerId) {
        if (block.timestamp < revealEnd) revert VotingStillOpen();

        uint256 maxVotes = 0;
        for (uint256 i = 0; i < candidates.length; i++) {
            if (candidates[i].voteCount > maxVotes) {
                maxVotes  = candidates[i].voteCount;
                winnerId  = uint8(i);
            }
        }
        emit WinnerDeclared(winnerId, candidates[winnerId].name, maxVotes);
    }

    // ── helpers ───────────────────────────────────────────────────────────
    function candidateCount() external view returns (uint256) {
        return candidates.length;
    }

    function currentPhase() external view returns (string memory) {
        if (block.timestamp < commitEnd)  return "COMMIT";
        if (block.timestamp < revealEnd)  return "REVEAL";
        return "ENDED";
    }

    function participation()
        external view returns (uint256 commits, uint256 reveals)
    {
        return (totalCommits, totalReveals);
    }
}
