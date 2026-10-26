# Part 4: Security Sentinel - Memory Hierarchy & Long-Context Safety Audit

## 1. Scope and assumptions

We analyze exact scaled dot-product attention, FlashAttention-style SRAM tiling, and inference-time KV-cache exhaustion. Let sequence length be N, head dimension d, fast on-chip memory capacity M scalar elements, and scalar size b bytes. Unless stated otherwise, N >= d and d <= M <= Nd; constants, layout overhead, and the cost of reading the input/output once are suppressed. Attention is computed exactly up to floating-point rounding, without materializing the N x N score or probability matrix in HBM.

## 2. Baseline attention: arithmetic and memory

For Q,K,V in R^(N x d), attention is O = softmax(Q K^T / sqrt(d)) V. Matrix multiplication requires Theta(N^2 d) arithmetic. A conventional implementation materializes scores S and/or probabilities P, each Theta(N^2) elements, and transfers Theta(N^2 + Nd) elements between HBM and SRAM, ignoring kernel-specific fusion and cache effects. Its attention intermediates can occupy Theta(N^2 b) bytes of HBM. This is separate from the Theta(Nd) bytes needed for Q,K,V,O.

## 3. FlashAttention SRAM tiling: formal IO derivation

Partition K and V into column blocks of Bc rows and Q into row blocks of Br rows. The algorithm keeps a Q block, a K/V block, an output accumulator, row-wise maximum m and normalizer l, and the Br x Bc score tile in SRAM. A sufficient storage condition (in scalar elements) is

    c1*Br*d + c2*Bc*d + c3*Br*Bc + c4*Br <= M,

for positive implementation constants c_i. Choose Bc = Theta(M/d) and Br = Theta(min(M/d, d)) with sufficiently small constant factors so the tile fits. In the common regime M >= Theta(d^2), Br = Theta(d), Bc = Theta(M/d). The K/V tiles are streamed once per Q tile, and Q/output blocks are read/written O(1) times per Q tile.

There are ceil(N/Br) Q blocks and ceil(N/Bc) K/V blocks. Each pair reads Theta(Bc*d) K/V elements, so total K/V traffic is

    Theta((N/Br)*(N/Bc)*(Bc*d)) = Theta(N^2*d/Br).

Reading and writing Q/output costs O(Nd) in a tiled schedule. With Br = Theta(d), the above naive loop ordering gives Theta(N^2) traffic, not the desired Theta(N^2*d^2/M). To attain the standard FlashAttention bound, use the original IO-optimal schedule with Bc = Theta(M/d), Br = Theta(min(M/d,d)), and *outer-loop over K/V tiles*: load each K/V tile once, then process every Q block, loading and updating Q, O, m, l for each pair. K/V traffic is O(Nd); Q/output/state traffic is

    Theta((N/Bc)*N*d) = Theta(N^2*d^2/M).

Thus the total HBM IO is

    T_IO = O(Nd + N^2*d^2/M) scalar transfers,

under the stated memory regime, and O(b*(Nd + N^2*d^2/M)) bytes. In the usual nondegenerate regime where N*d >= M, the quadratic term dominates Nd, yielding O(N^2*d^2/M). The established asymptotic IO optimality result applies in the specified SRAM range and computational model; it is not a universal lower bound for every architecture or attention variant. Arithmetic remains Theta(N^2*d). Practical FlashAttention implementations may choose different tile sizes and loop orders for hardware efficiency.

## 4. Numerically stable online softmax

For one query row, suppose previously processed keys have maximum m, denominator l = sum_j exp(s_j-m), and unnormalized output accumulator a = sum_j exp(s_j-m)*v_j. For a new score block with maximum mt, local denominator lt and local weighted sum at, define

    mnew = max(m, mt)
    lnew = exp(m-mnew)*l + exp(mt-mnew)*lt
    anew = exp(m-mnew)*a + exp(mt-mnew)*at
    output = anew/lnew.

Initialize m=-infinity, l=0, a=0. The rescaling is exact in real arithmetic and avoids writing the N x N softmax matrix. For causal attention, mask future-key scores before computing block maxima; fully masked tiles are skipped. The online update is associative at the level of the represented sums (subject to floating-point roundoff).

## 5. KV-cache capacity and denial-of-service model

For batch/concurrent sequences indexed by i, with live token counts Li, Hkv KV heads, per-head width dk, and cache element size b, ideal cache payload is

    KV_bytes = 2*b*Hkv*dk*sum_i Li.

With Hkv=Hq and dk=dmodel/Hq, this is 2*b*dmodel*sum_i Li for standard MHA; GQA reduces it by Hkv/Hq. The actual VRAM budget also includes weights, activations, workspaces, allocator metadata, fragmentation, and reserved headroom. A malicious or simply oversized workload can exhaust the cache via long prompts, large output-token reservations, high concurrency, or many stalled sessions. Consequences include admission failures, out-of-memory exceptions, latency spikes, or eviction/recomputation overhead. FlashAttention reduces attention intermediate traffic; it does not by itself eliminate the persistent KV-cache growth with context length.

Example (payload only): Hkv=8, dk=128, b=2 bytes => 4096 bytes/token/sequence (4 KiB). A 128K-token context needs 512 MiB per sequence; 16 such sequences require 8 GiB of cache payload before other VRAM uses. These are illustrative calculations, not measurements.

## 6. PagedAttention versus MLA: defensive evaluation

PagedAttention stores KV in fixed-size logical blocks mapped to noncontiguous physical blocks. It reduces internal/external fragmentation, permits block sharing in suitable workloads, and supports flexible allocation/reclamation. It does NOT change the fundamental Theta(sum Li) KV payload growth, so an attacker can still exhaust physical pages. Mitigations must include quotas and admission control.

Multi-head Latent Attention (MLA) caches a lower-dimensional latent representation rather than full per-head K/V tensors, subject to the architecture's reconstruction/position-encoding design. If the cache stores r latent values plus p additional positional values per token, its ideal payload is b*(r+p)*sum Li, compared with 2*b*Hkv*dk*sum Li for conventional KV. The saving factor is approximately (2*Hkv*dk)/(r+p), provided r+p is smaller. MLA requires model support and can add projection/reconstruction compute; it is not a drop-in retrofit for arbitrary trained MHA weights. Cache remains linear in total tokens, so compression delays but cannot prevent resource exhaustion.

| Mechanism | Primary benefit | What it does not solve |
|---|---|---|
| FlashAttention | Avoids N^2 attention intermediates and reduces HBM IO | Persistent KV-cache growth |
| PagedAttention | Better physical page utilization, sharing, reclamation | Unlimited logical token demand |
| MLA | Smaller per-token KV representation in compatible models | Unbounded total tokens or concurrency |
| GQA/MQA | Fewer KV heads per token | Unbounded admissions |
| Quotas + scheduler | Enforces finite per-tenant and aggregate resource budgets | Does not reduce underlying per-token compute |

## 7. Defense-in-depth policy

1. Enforce authenticated per-tenant limits on input tokens, reserved output tokens, concurrent sequences, and aggregate live KV tokens. Reserve output capacity at admission; charge actual usage during decode.
2. Estimate available physical KV blocks from free VRAM after weights, workspace, safety margin, and fragmentation allowance. Reject or queue requests if projected pages exceed budget; never rely on optimistic overcommit.
3. Use fair scheduling and per-tenant token-bucket or weighted admission limits so one tenant cannot monopolize memory. Apply backpressure, queue timeouts, and maximum session lifetimes.
4. Reclaim cache pages on completion, cancellation, disconnect, and timeout. Track reference counts for shared prefix blocks and avoid use-after-free or cross-tenant cache exposure.
5. For high-risk workloads, cap context length, use bounded prefix caching, consider GQA/MLA-compatible models, and offer explicit truncation/summarization policies rather than silently discarding important context.
6. Monitor live KV pages, reserved-versus-used tokens, free VRAM, allocation failures, preemption/recompute counts, p95/p99 latency, and tenant-level utilization. Alert before the hard limit.
7. Test adversarial but authorized load cases: many near-limit prompts, long decoding, cancellation storms, cache sharing, mixed-size batches, and simultaneous admissions. Verify isolation and graceful rejection rather than GPU crashes.

## 8. Audit acceptance criteria and limitations

PASS if: (a) a configured per-tenant token and concurrency quota is never exceeded; (b) worst-case admitted reservations fit the measured usable KV-block budget plus explicit headroom; (c) cancellation and expiry release all unshared pages; (d) no cross-tenant cache content is exposed; and (e) overload yields bounded queueing or explicit errors, not uncontrolled OOM. Measure these in the target deployment; this document presents a theoretical audit and threat model, not empirical benchmark evidence.

## 9. References

- Tri Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," NeurIPS 2022, arXiv:2205.14135.
- Woosuk Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023, arXiv:2309.06180.
- DeepSeek-AI et al., "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model," arXiv:2405.04434 (MLA architecture).
