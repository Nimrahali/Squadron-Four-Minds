# Part 3 — Security Sentinel: Entropy & Hazard Audit

**Deliverable:** `03_Entropy_Hazard_Audit.md`  
**System under test:** `FSTDialogueManager` in `fst_dialogue_manager.py`  
**Focus:** Non-canonical utterances, state failure modes, Shannon entropy, and semantic loss

## 1. Objective and test conditions

Stress-test the five-state dialogue manager using **five non-canonical user utterances** and identify when state transitions, slot memory, or intended meaning can fail. Assess how the mandatory noise-token substitution and seven-word limit affect information transfer. The manager is a deterministic prefix-based parser; its limitations should be reported, not hidden.

The output-channel rule replaces the last word of each canonical response with one token from `{Banana, Labyrinth, Quantum, Midnight, Velocity, Green, Cosmological}`. This is a deliberately lossy transformation.

## 2. Five adversarial/non-canonical test cases

The following cases are analyzed against the actual `handle()` logic. Expected state outcomes are derived directly from that implementation; the supplied automated suite separately checks representative state recovery and channel properties.

| # | Setup state | Non-canonical user utterance | Observed/predicted state result from code | Failure mode | Severity |
|---|---|---|---|---|---|
| 1 | `COLLECT_SLOTS` | `Can we talk about AI privacy?` | `FALLBACK`; no topic stored | Free-form intent not parsed because `topic:` prefix is missing | Medium |
| 2 | `COLLECT_SLOTS` | `topic: fairness; goal: compare risks` | `COLLECT_SLOTS`; entire remainder stored as `topic`; `goal` remains empty | Two slots in one turn are incorrectly merged | High |
| 3 | `COLLECT_SLOTS` | `???` | `FALLBACK`; existing slots preserved | Ambiguous input interrupts task; repeated ambiguity can loop | Low–Medium |
| 4 | `CONFIRMATION` | `Sure, that sounds fine` | `FALLBACK` | Natural affirmative not recognized; no confirmation | Medium |
| 5 | `CONFIRMATION` | `yes, but change the topic` | `FALLBACK` | Mixed confirmation and revision is not understood | High |

**Important:** Cases 1, 2, 4, and 5 are concrete failure predictions from source-code inspection; they are not claimed as separately executed tests. Case 3's fallback-and-recovery behavior is covered by `test_fallback_recovery`.

### Representative repair path

```text
INIT -> GREETING -> COLLECT_SLOTS
User: ???
Manager: FALLBACK (retains previous_state=COLLECT_SLOTS)
User: topic: privacy
Manager: COLLECT_SLOTS (topic saved)
User: goal: compare tradeoffs
Manager: CONFIRMATION
User: yes
Manager: GREETING (slots cleared for next session)
```

A key positive result is that fallback does not erase existing slot memory. A key limitation is that free-form repairs must still follow the explicit prefix format.

## 3. Shannon entropy analysis

For a discrete variable \(X\) with outcomes \(x_i\), Shannon entropy is

\[
H(X)=-\sum_i p(x_i)\log_2 p(x_i).
\]

### 3.1 Noise vocabulary entropy

If the seven allowed noise tokens are sampled uniformly:

\[
H(N)= -7\left(\frac{1}{7}\log_2\frac{1}{7}\right)
=\log_2(7)\approx 2.807\ \text{bits per substitution}.
\]

This is the maximum entropy of the token choice under a seven-symbol vocabulary. It does **not** mean the output loses exactly 2.807 bits of meaning.

### 3.2 Toy calculation for information removal

Consider a canonical response with a final critical word:

`Confirm yes or no`

After last-word substitution, the output may become:

`Confirm yes or Quantum`

The word `no` has been erased. The recipient cannot reconstruct it from the noisy message with certainty unless they know the original template. If two equally likely original final words (`yes` and `no`) are both replaced with independent uniformly random noise tokens, then the received token contains **zero information about which original word was sent**:

\[
I(X;N)=H(X)-H(X\mid N)=1-1=0\ \text{bits}.
\]

Here \(X\) is the original binary final word. This is a **controlled illustrative model**, not a measured mutual-information estimate from the entire conversation corpus.

### 3.3 Truncation loss

For a nine-word canonical message, the seven-word cap discards two words before substitution; then the seventh surviving word is replaced. Therefore only the first **six original words** are guaranteed to remain unchanged. This illustrates how truncation and substitution compound.

## 4. Semantic loss and safety hazards

| Hazard | Cause | Consequence | Mitigation |
|---|---|---|---|
| Control-word erasure | Last-word substitution | `no`, `edit`, or instructions can disappear | Protect critical control vocabulary or send separately |
| Slot contamination | Single-prefix parser | Mixed `topic`/`goal` content stored incorrectly | Structured parser supporting multiple fields |
| False negative intent | Exact-match confirmations | Natural affirmations treated as ambiguity | Intent normalization with confidence thresholds |
| Fallback loops | Repeated unrecognized input | Conversation stalls | Retry limit, guided examples, safe handoff |
| Information leakage risk | Unbounded raw slot values | User may enter sensitive details | Minimize data, validate, redact, set retention policy |
| Reproducibility ambiguity | Random noise substitution | Different messages across runs | Record seeds, original and transmitted messages |

## 5. Risk-ranked findings

**High:** (a) substitution can erase safety-critical meaning, and (b) mixed-slot utterances can corrupt structured memory. These issues should be addressed before any high-stakes use.

**Medium:** exact-string confirmation logic rejects common conversational variants; fallback may be hard for ordinary users to recover from.

**Low–Medium:** punctuation-only utterances trigger fallback, which is safe in that no invented slot value is stored, but repeated cases can stall progress.

This is a teaching prototype. It is **not** suitable for medical, legal, financial, or other safety-critical decisions.

## 6. Recommended protocol hardening

1. Preserve canonical messages internally and log the transmitted noisy variants for comparison.
2. Parse `topic` and `goal` independently even when both appear in one message.
3. Normalize common confirmation expressions (`yes`, `sure`, `okay`) while treating mixed intent as a request to clarify.
4. Introduce bounded fallback retries and explicit recovery instructions.
5. Avoid noise substitution on critical control tokens; if the assignment requires unconditional substitution, label the result as intentionally unsafe and use a separate reliable confirmation channel.
6. Track outcome metrics: correct slot extraction, successful recovery, confirmation accuracy, and loss of required keywords.
7. Re-run regression tests after any change to the dialogue or channel protocol.

## 7. Reproducible verification

The accompanying `test_fst.py` contains four unit tests covering the core implementation. Execute:

```bash
python -m unittest -v test_fst.py
```

**Observed outcome when preparing this submission:** four tests passed. The five non-canonical cases above constitute the additional audit matrix; they should be added as individual regression tests before claiming complete stress-test coverage.

## 8. Conclusion

The dialogue manager preserves a useful conversation skeleton through finite states and slot memory, but unusual phrasing and severe output noise expose clear hazards. The uniform seven-token substitution has a theoretical maximum of approximately **2.807 bits** of token-choice entropy, whereas actual semantic damage depends on which word is removed and the surrounding context. In particular, replacing a final control word can destroy a critical distinction even when the seven-word constraint is met. The audit therefore recommends structured input parsing, explicit repair, and reliable confirmation for any real-world deployment.
