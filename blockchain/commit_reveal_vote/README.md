# Commit-Reveal Vote

Privacy-preserving on-chain voting using a **commit-reveal scheme** in Solidity.

## The problem with naive on-chain voting

Ethereum is fully public. If voters call `vote(candidateId)` directly, their choice is visible to everyone in the mempool before the transaction is even mined. Later voters can see the running tally and vote strategically. This breaks ballot secrecy.

## The solution: commit-reveal

Split voting into two phases:

### Phase 1 — Commit (deadline: `commitEnd`)
Voter submits: `keccak256(abi.encodePacked(choice, secret, voterAddress))`

On-chain only stores the **hash** — the actual vote is invisible. All committed votes look like random 32-byte strings.

### Phase 2 — Reveal (deadline: `revealEnd`)
Voter submits their plaintext `choice` + `secret`. The contract re-hashes and verifies. Only then is the vote counted.

Since all commitments are locked in before any reveal is accepted, no one can strategically change their vote after seeing early results. A voter who fails to reveal is counted as an abstention.

## Run it (Remix IDE — quickest demo)

1. Open [remix.ethereum.org](https://remix.ethereum.org)
2. Create `CommitRevealVote.sol` and paste the contract
3. Compile with Solidity 0.8.17+
4. Deploy with:
   ```
   _candidateNames: ["Alice", "Bob", "Carol"]
   _commitDuration: 300    (5 minutes)
   _revealDuration: 300    (5 minutes)
   _whitelistEnabled: false
   ```

### Commit a vote
```js
// In Remix console or ethers.js:
const choice  = 0;                          // voting for Alice
const secret  = ethers.utils.randomBytes(32);
const commitment = ethers.utils.solidityKeccak256(
    ["uint8", "bytes32", "address"],
    [choice, secret, voterAddress]
);
// Call: commit(commitment)
```

### Reveal the vote
```js
// After commitEnd, before revealEnd:
// Call: reveal(choice, secret)
// Contract verifies and counts the vote
```

### Check results
```js
// After revealEnd:
// Call: results()   → returns candidate names + vote counts
// Call: declareWinner()  → emits WinnerDeclared event
```

## Security analysis

| Attack | Defence |
|--------|---------|
| See votes early | Commits are hashes — choice is hidden until reveal phase |
| Strategic late voting | Commit deadline locks all ballots before any reveal |
| Commitment stealing | `msg.sender` is mixed into the hash — can't reuse another's commitment |
| Brute-force preimage | 32-byte random secret → 2²⁵⁶ preimage search space |
| Vote twice | One commitment per address; `hasRevealed` flag blocks second reveal |
| Withhold reveal to change outcome | Non-revealers counted as abstentions — cannot swing result without losing their commit |

## Limitations (honest)

- **Not fully anonymous**: the *reveal* transaction links your address to your choice on-chain. For full anonymity, combine with ZKP (see `../zkp_anonymous_proof/`).
- **Trusted reveal period**: voters must come back online to reveal. If a voter loses their `secret`, their commit is wasted.
- **Semi-honest model**: the contract enforces correct counting, but a malicious admin could front-run registrations. Use a public Merkle whitelist to remove admin trust.
