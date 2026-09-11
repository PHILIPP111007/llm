# Sublinear Attention Computation in Ocean

## Overview

Ocean implements an experimental content-dependent sparse attention algorithm
for long-context autoregressive inference. The algorithm is designed to reduce
the amount of key/value data inspected by the attention operation while
preserving the full KV cache.

For every query, dense causal attention compares the query with every previous
key. Ocean instead divides the cached sequence into fixed-size blocks, creates
a compact summary for every block, and selects a small number of relevant
blocks using cosine similarity. Exact attention is then computed over every
token inside the selected blocks.

The algorithm is a routing approximation. It is not claimed to be the same as
any proprietary or external attention implementation, and it does not by
itself provide a 10-million-token context window.

The empirical results below come from a separate transparent PyTorch
Pythia-1B prototype. That prototype is useful for validating the routing idea,
but it must not be confused with the hierarchical/CUDA implementation
described in the prefill sections: the Pythia prototype still uses dense
prefill, batch size 1, a full KV cache, Python-side route construction, and
`torch.cat` when extending the cache.

## Configuration

The current GPT-2-style implementation uses:

```text
summary window W       = 100 recent tokens
route refresh interval = 50 decoded tokens
block size S           = 64 tokens
semantic blocks        = 5
mandatory local blocks = 2 latest blocks
exploration blocks     = 1
route width            = 2 + 5 + 1 = 8 blocks
maximum selected keys  = 8 * 64 = 512 tokens
```

The parameters are configurable in the Tensor/runtime API, although the
current GPT-2 model uses the values above. If the active context is shorter
than 512 tokens, the effective number of selected tokens is smaller.

## Data structures

For every Transformer layer, Ocean stores:

```text
K cache       [batch, heads, context, head_dim]
V cache       [batch, heads, context, head_dim]
block summary [batch, heads, summary_blocks, head_dim]
route         [batch, heads, 8]
```

The summary of block `j` is the mean of the key vectors in that block:

```text
summary[j] = mean(K[j * S : (j + 1) * S])
```

The final block may contain fewer than `S` tokens and is averaged using its
actual number of tokens.

Summaries and routes are persistent inference state. They are maintained
separately for every attention layer and head.

## Route construction

For an active sequence ending at position `t`, the algorithm performs the
following steps.

### 1. Build a recent-query summary

The last `W` available key vectors are averaged:

```text
recent = mean(K[max(0, t - W) : t])
```

This vector represents the current local context. It is not the Transformer
query itself; it is a compact routing signal derived from recent keys.

### 2. Score visible blocks

For each visible block summary `s_j`, compute cosine similarity:

```text
score(j) = dot(recent, s_j)
           ------------------------
           ||recent|| * ||s_j|| + eps
```

Only blocks whose tokens are visible in the active prefix are considered.

### 3. Add mandatory local blocks

The route always includes the latest two visible blocks. These blocks are
excluded from semantic selection, so they cannot be replaced by a low-scoring
distant block. Near the beginning of a sequence, fewer than two blocks may be
available.

This protects short-range syntax, recent entities, and the immediate causal
neighborhood of the query.

### 4. Select five semantic blocks

The five non-local blocks with the highest scores are placed into the route.
Ties are resolved deterministically by preferring the lower block index. The
selected block IDs are stored rather than copying their keys or values.

### 5. Add an exploration block

One additional block is selected using a deterministic pseudo-random sequence.
The implementation attempts to avoid duplicating a local or semantic block.
This keeps the route from becoming permanently locked to the same regions.

The exploration choice is deterministic for reproducibility; it is not a
cryptographically random or nondeterministic sample.

The resulting route has the form:

```text
[local_block_0,
 local_block_1,
 semantic_block_0,
 semantic_block_1,
 semantic_block_2,
 semantic_block_3,
 semantic_block_4,
 exploration_block]
```

## Sticky routes during decoding

Autoregressive decoding does not rebuild the route for every token. A route is
created once and reused for the next 50 decoded tokens. After the refresh
interval expires, the route is rebuilt from:

```text
the latest 100 keys
and the hierarchical summary index
```

This produces the following execution pattern:

```text
build route -> use route for 50 tokens -> build route -> ...
```

The route is kept independently for each layer and attention head. The current
key and value are written to the KV cache before routed attention is executed.

The summary for the current block is also updated as new tokens arrive. The
corresponding hierarchy leaf and all its ancestors are updated in place, so a
decode step does not rebuild or rescan the entire context. On the CUDA path
both summary and hierarchy updates use native kernels.

## Routed attention

After a route has been selected, Ocean performs ordinary scaled dot-product
attention over all tokens inside the selected blocks:

```text
Q                         [batch, heads, query_length, head_dim]
K_route, V_route          tokens from the eight selected blocks at most
scores = Q @ K_route^T
scores = scores / sqrt(head_dim)
scores = causal_softmax(scores)
output = scores @ V_route
```

The attention inside the route is exact. The approximation occurs only in the
block-selection stage: tokens outside the selected blocks are not available to
that query.

Causal masking is still applied. A selected block can be partially visible for
a query near the beginning of that block; future tokens are not included in the
attention result.

## Hybrid local attention plus Ocean routing

The next proposed variant combines a strict sliding-window component with the
existing Ocean route. This is a proposed extension, not the configuration used
for the Pythia measurements above.

For every query, the candidate keys are formed as the union of four sets:

```text
local window       = the latest W tokens
semantic blocks    = S blocks selected by Ocean routing
global blocks      = G mandatory blocks, for example the first block
exploration blocks = optional additional blocks
```

Duplicate tokens are removed before attention. The attention kernel then runs
exact causal softmax only over this union:

```text
K_candidate, V_candidate = unique(local_window
                                  ∪ semantic_blocks
                                  ∪ global_blocks
                                  ∪ exploration_blocks)
output = causal_attention(Q, K_candidate, V_candidate)
```

The local window is different from the current `local_blocks` parameter. The
current parameter forces the latest blocks into the route, but it still allows
the full KV cache to remain addressable. A strict window of `W` tokens gives a
hard bound on the recent-token part of the attention operation. Old tokens are
still retained in the KV cache unless an eviction or compression policy is
added.

A reasonable first configuration for an experiment is:

```text
window size W       = 256 tokens
block size B        = 256 tokens
semantic blocks S   = 8
global blocks G     = 1
refresh interval R  = 16 tokens
```

The maximum exact-attention candidate set is then approximately:

```text
K <= W + (S + G) * B
  = 256 + (8 + 1) * 256
  = 2560 tokens
```

The actual number can be smaller because the local window overlaps the latest
selected block and duplicate blocks are removed. A smaller route, such as
`S=4`, should be evaluated separately because increasing the route budget can
erase the intended speed benefit.

This hybrid design gives the model guaranteed access to recent syntax and
short-range dependencies while Ocean routing supplies a bounded number of
older, semantically relevant regions. It is not equivalent to dense attention:
any old token outside the window and selected blocks is invisible to the
current query. The quality claim therefore requires perplexity and long-range
retrieval ablations against both dense attention and pure sliding-window
attention.

### Asymptotic estimate for the hybrid variant

Let `N` be the active context length, `D` the head dimension, `B` the block
size, `W` the local-window size, `S` the number of semantic blocks, `G` the
number of global blocks, and `K` the number of unique candidate tokens:

```text
K <= W + (S + G) * B
```

For one decoded token, the exact attention computation is:

```text
O(K * D)
```

If `W`, `B`, `S`, and `G` are constants independent of `N`, this is
`O(D)` with respect to context length. In other words, the attention kernel is
constant-time in `N` for decoding, which is stronger than merely linear in
`N`. This statement applies only to the selected-attention kernel, not to the
whole Transformer block.

The route refresh adds a separate cost. With a full scan of block summaries,
one refresh costs approximately:

```text
O((W + N/B) * D)
```

Amortized over `R` decoded tokens, this becomes:

```text
O(((W + N/B) * D) / R)
```

Therefore a naive implementation is not truly `O(1)` in total decode cost as
`N` grows: the attention kernel is bounded, but route construction can grow
with the number of blocks. A persistent hierarchical index changes the refresh
term to approximately:

```text
O((W + beam * log(N/B)) * D)
```

and incremental summary updates add `O(D)` per appended token. The resulting
per-token decode estimate is:

```text
O(D^2)                                   projections and MLP
+ O(K * D)                                local + routed attention
+ O(((W + beam*log(N/B)) * D) / R)         amortized route refresh
```

For fixed model width, route budget, beam, and refresh interval, the attention
term is independent of `N`, while the indexed routing term grows only
logarithmically. The end-to-end model is therefore approximately
`O(log N)` in its context-dependent routing component, not automatically
`O(1)` overall.

For prefill, every prompt token or query chunk still has to be processed. With
fixed `K` and a hierarchical selector, the idealized attention and routing
work is approximately:

```text
O(N * K * D) + O((N/C) * beam * log(N/B) * D)
```

for chunk size `C`, in addition to the model's linear per-token projections
and MLP work. This is approximately linear in `N` for fixed parameters, but
Python route construction, gathers, kernel launches, and unfused sparse
kernels can dominate wall-clock time. A measured flat prefill throughput is
not by itself proof of sublinear asymptotic behavior.

Memory has a separate bound. If the complete KV cache is retained, memory is
still:

```text
O(L * N * D)
```

The sliding window limits the attention candidates, not the stored history. To
obtain `O(W + (S+G)B)` active memory, old KV entries must be evicted or replaced
by compressed summaries; that change can reduce quality and is a separate
algorithmic decision.

## Prefill behavior

During long-prompt prefill, the input is processed in query chunks of 50
tokens. For each chunk, Ocean builds a route and applies routed attention to
the chunk. The route is therefore reused by all queries in that chunk.

The prefill flow is:

```text
project full prompt to Q/K/V
        ↓
write K/V to the cache
        ↓
build persistent block summaries
        ↓
for every 50-token query chunk:
    build a visible-prefix route
    run causal routed attention
        ↓
merge attention output
```

This avoids materializing a dense `query_length × key_length` attention matrix
for the sparse path. It does not eliminate all prompt-length-dependent work:
QKV projections, block-summary construction, route selection, and output
projection still process the prompt.

For long prefill, the route selector uses a balanced hierarchy over the block
summaries. The leaves contain the original block means; every internal node is
a count-weighted mean of its two children. The tree is padded to the next
power-of-two number of leaves, but padded leaves are never eligible routes.
The runtime traverses the tree with a fixed beam (currently
`clamp(4 * semantic_blocks, 8, 32)`) and returns only the best leaf candidates.
This changes the selector from a full scan to a bounded tree traversal:

```text
build summaries + hierarchy = O(N * D)
one chunk route               = O(W * D + beam * log(N/S) * D)
all prefill chunks             = O(N * D + (N/C) * beam * log(N/S) * D)
```

With fixed `W`, `C`, `S`, and `beam`, the prefill route-selection component is
`O(N log N)` in the strict comparison model and usually behaves close to
linear for the practical context range. The model's QKV projections, MLP,
embedding, LayerNorm, and output projection remain linear in the number of
prompt tokens (with their usual per-token `D²` work). The index is an
approximation: hierarchical node cosine scores can differ from the exact
best leaf scores, so quality must be checked against the dense path.

## Complexity

Let:

```text
N = active context length
D = model hidden width
S = block size
M = number of routed blocks
K = M * S selected tokens
R = route refresh interval
```

### Dense decode attention

The attention part of one new token is:

```text
O(N * D)
```

Across `L` layers:

```text
O(L * N * D)
```

### Routed decode attention

The exact attention over selected blocks is:

```text
O(K * D)
```

With fixed `M = 8` and `S = 64`, `K` is bounded by 512, so the attention
kernel itself is effectively `O(D)` with respect to context length `N`.

### Route refresh cost

The legacy explicit route API scans approximately `N / S` block summaries and
compares each summary with the recent vector:

```text
one refresh = O((W + (N / S)) * D)
```

Amortized over `R` decoded tokens:

```text
per-token route cost = O(((W + (N / S)) * D) / R)
```

That compatibility path is described by:

```text
O(D^2)                         projections and MLP
+ O(K * D)                     routed attention
+ O(((W + N/S) * D) / R)       amortized route refresh
```

The hierarchical route API used by GPT-2 prefill and decode instead has bounded
traversal cost:

```text
one refresh = O((W + beam * log(N/S)) * D)
```

The GPT-2 decode cache now persists the hierarchy and updates one leaf-to-root
path per token. Its selector is bounded by the tree traversal; the legacy
full-scan route API remains available as a compatibility/reference path.

### Prefill cost

For a chunk size `C`, the legacy prefill route selector is called for roughly
`N / C` chunks. Its route-scanning component is approximately:

```text
O((N / C) * (N / S) * D)
```

The routed attention component is:

```text
O((N / C) * C * K * D) = O(N * K * D)
```

The hierarchical path removes this repeated global scan. Its indexed route
component is approximately:

```text
O((N / C) * beam * log(N/S) * D)
```

so all prompt-dependent work is no longer quadratic in `N`. The hierarchy is
constructed once after prefill and then updated along one leaf-to-root path per
decoded token in `O(log(N/S) * D)`. The raw `O(N * D)` hierarchy construction is
performed once per layer.

## Memory complexity

The persistent KV cache remains the dominant allocation:

```text
KV cache = O(2 * L * N * D) elements
```

With FP32 values, this is approximately:

```text
8 * L * N * D bytes
```

For GPT-2 Small (`L=12`, `D=768`, batch size 1):

```text
N = 9000   -> approximately 633 MiB for K/V
N = 10000  -> approximately 703 MiB for K/V
```

Block summaries require:

```text
O(L * N * D / S)
```

FP32 summaries for `N=9000`, `L=12`, `D=768`, and `S=64` require only about
5 MiB. Routes require:

```text
O(L * heads * M) int32 values
```

and are negligible compared with the KV cache.

The important consequence is:

> Sparse routing reduces attention computation, but it does not remove the
> linear KV-cache memory requirement.

## Expected reduction at a 9000-token context

With two local, five semantic, and one exploration block:

```text
maximum routed tokens = 512
dense candidate tokens = 9000
attention reduction     ≈ 9000 / 512 ≈ 17.6x
```

This is a reduction for the QK and value-aggregation loops, not an end-to-end
model speedup. Linear projections, MLP layers, embeddings, output logits,
kernel launches, synchronization, and memory movement remain.

The measured end-to-end throughput must therefore be interpreted separately
from the theoretical attention reduction.

## Correctness and quality considerations

The algorithm preserves:

```text
causal masking inside selected blocks
exact softmax over selected tokens
exact value aggregation over selected tokens
deterministic route construction
```

It does not preserve dense-attention equivalence, because relevant tokens can
be omitted during routing. Quality should be evaluated with:

```text
dense vs sparse logits
next-token agreement
perplexity
long-range retrieval tests
generation agreement over many random prompts
```

The route is selected from recent key statistics rather than directly from the
current query. This is intentionally cheap, but it can miss information that
is not represented by the latest 100-key mean.

For strict causal prefill semantics, route summaries must not contain
information from future tokens relative to the current query chunk. The current
implementation uses active-prefix routing and causal masking in the attention
kernel, but summary construction and route selection should continue to be
validated carefully for this property.

## Empirical validation on Pythia-1B

The routing variants were evaluated in a manually implemented PyTorch
Pythia-1B model loaded with the official weights. The evaluation used the same
Tiny Shakespeare token IDs for every model and an autoregressive KV-cache
protocol. The model's trained context is 2048 tokens, so the quality benchmark
uses 2048 input tokens and reports 2047 next-token targets. No 14K perplexity
claim is made: 14K exceeds Pythia's trained context and is only suitable as a
separate performance stress test.

### Baseline and first routing variants

| Variant | Mean NLL | Perplexity | Time | Tokens/s | Relative PPL |
|---|---:|---:|---:|---:|---:|
| Dense attention | 3.0701 | 21.54 | 26.318 s | 77.78 | 1.00x |
| Original Ocean, 8 blocks | 3.4758 | 32.33 | 30.171 s | 67.85 | 1.50x |
| Query-dependent Ocean, 8 blocks | 3.3669 | 28.99 | 56.893 s | 35.98 | 1.35x |

The first Ocean route used one mean summary per 64-token block, local blocks,
semantic blocks, and an exploration block. It increased PPL by about 50% and
was slower than dense attention. Replacing the recent-key heuristic with the
current query improved PPL, but the per-token query-dependent route construction
made the implementation roughly twice as slow as dense attention.

### Enhanced selector quality frontier

The enhanced selector uses four summary vectors per 64-token block, preserves
one global first block and two local final blocks, and selects the remaining
blocks using the current query. Results:

| Block size | Route blocks | Summary parts | Global blocks | Mean NLL | Perplexity | Tokens/s | Speedup vs dense |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 8 | 1 | 0 | 3.3669 | 28.99 | 32.53 | 0.45x |
| 64 | 8 | 4 | 1 | 3.0851 | 21.87 | 32.75 | 0.45x |
| 64 | 12 | 4 | 1 | 3.0754 | 21.66 | 32.82 | 0.45x |
| 64 | 16 | 4 | 1 | 3.0711 | 21.57 | 32.54 | 0.45x |
| 64 | 24 | 4 | 1 | 3.0700 | 21.54 | 33.03 | 0.46x |
| 64 | 32 | 4 | 1 | 3.0699 | 21.54 | 33.02 | 0.46x |
| 32 | 16 | 4 | 1 | 3.0816 | 21.79 | 32.69 | 0.45x |

The quality loss was therefore caused primarily by the coarse one-summary
representation, not simply by selecting fewer blocks. With four summaries per
block, 16 selected blocks recover dense quality within the noise of this
single 2048-token evaluation fragment. At 24 blocks the result is effectively
identical to dense; 32 blocks is the full-context control because 2048 tokens
contain 32 blocks of 64 tokens.

The 32-token block experiment did not improve quality at the same approximate
512-token route budget: `block_size=32, route_blocks=16` produced PPL 21.79,
versus 21.57 for `block_size=64, route_blocks=16`. This is not evidence that
smaller blocks are universally worse; it only means that this selector and
summary construction did not benefit from that change on this test.

### What the speed results actually establish

The enhanced variants all run at approximately 32--33 tokens/s regardless of
whether they select 8, 12, 16, 24, or 32 blocks. Reducing the route width
therefore does not yet reduce wall-clock time. The likely fixed costs are:

```text
full summary rebuild at every refresh
per-head Python route construction and sorting
irregular gather operations
dynamic attention masks
repeated torch.cat while extending the KV cache
unfused sparse attention versus dense SDPA kernels
MLP and projection work that routing does not reduce
```

At the 2048-token trained context, the current evidence supports a quality
result, not an end-to-end speed result. The incremental route optimization
changes the conclusion for long-cache decoding, as shown below.

```text
quality:       substantially improved; route=16 is nearly dense-equivalent
2048 speed:    still slower than dense because the context is short
14K decode:    faster in decode, but total speedup is limited by dense prefill
long context:  speed benchmark is valid; perplexity benchmark is not
```

Incremental summary updates and GPU `topk` route selection were then added to
the Pythia prototype. A separate 14K run measures performance only; it is not a
quality test because 14K exceeds the model's trained 2048-token context.

### 14K incremental decode performance

The prompt contains 14,000 tokens and generation measures 64 new tokens. The
prefill remains dense in every variant:

| Variant | Refresh | Prefill s | Decode s | Decode tok/s | Total s | Decode speedup | Total speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense | -- | 6.854 | 1.673 | 37.65 | 8.528 | 1.00x | 1.00x |
| Incremental Ocean | 4 | 6.856 | 1.152 | 54.68 | 8.008 | 1.45x | 1.06x |
| Incremental Ocean | 16 | 6.870 | 1.018 | 61.88 | 7.888 | 1.64x | 1.08x |

At `refresh_interval=16`, the route is rebuilt 16 times less frequently than
at `refresh_interval=1`, while the 2048-token quality benchmark remains nearly
unchanged:

```text
dense PPL                         = 21.54
incremental Ocean, refresh=16    = 21.62
```

The result demonstrates a real long-cache decode speedup, but not a comparable
end-to-end speedup: dense prefill accounts for approximately 6.9 of the 7.9
seconds. The remaining performance work is preallocated KV-cache storage,
fused block-sparse gather/attention, and reducing the cost of score evaluation
over all visible block summaries.

### Required routing ablation: full scan versus hierarchy

The hierarchical selector must be compared directly with a full-scan selector
before claiming that the asymptotic improvement is useful in practice. Both
variants must use the same model weights, tokenizer, numerical dtype, block
size, route width, local/global blocks, refresh interval, prompt fragments,
random seeds, and warm-up protocol. Only the route-construction algorithm may
change.

The comparison must report three separate dimensions:

```text
quality:
    mean NLL and perplexity on the same held-out token stream

performance:
    route-refresh time, prefill tok/s, decode tok/s, end-to-end latency,
    and peak GPU memory at several context lengths

long-range retrieval:
    exact retrieval accuracy for information placed outside the local window,
    selected-block recall against full-scan top-k blocks, and generated-answer
    accuracy on synthetic distance-controlled prompts
```

The minimum context sweep should include `N=2048`, `4096`, `8192`, and
`14000` tokens where the implementation supports them. The 14K Pythia result
must remain a speed/stress measurement rather than a trained-context
perplexity claim. For quality, use contexts within the model's validated
training limit or a separately trained long-context model.

The most important routing-specific metric is selected-block recall:

```text
recall = |hierarchical_route ∩ full_scan_route| / |full_scan_route|
```

This should be measured at the leaf-block level and separately for local,
global, and semantic blocks. Perplexity alone can hide a routing failure if
the model compensates using local or global blocks. Conversely, equal PPL at a
short context is weak evidence when the route covers most of the available
sequence.

The expected hypotheses are:

```text
full scan:
    better or equal route recall, but refresh cost grows as O(N/B)

hierarchical routing:
    lower refresh cost, approximately O(beam * log(N/B)), with possible
    quality loss from approximate pruning
```

No claim that hierarchical routing is superior should be made unless it keeps
perplexity and long-range retrieval within a predefined tolerance while
reducing measured routing or end-to-end latency. A speedup in the final model
is not sufficient if the hierarchy merely shifts the cost into summary
construction, gathers, or Python-side bookkeeping.

## Current limitations

The current implementation has these limitations:

1. The KV cache still grows linearly with context length.
2. Route refresh still scores all visible block summaries; only summary
   construction is incremental in the optimized Pythia path.
3. Prefill repeats route selection for every 50-token chunk.
4. The route budget is fixed rather than adaptive to query uncertainty.
5. One deterministic exploration block is not equivalent to true random
   sampling.
6. Sparse attention is an approximation and can lose long-range information.
7. End-to-end speed is also limited by projections and MLP computation.
8. In the Pythia prototype, query-dependent routing restores quality but is
   still slower than dense attention at the 2048-token context.
9. The 14K speedup applies to decode only; prefill remains dense.

## Future improvements

The next algorithmic improvements are:

### Hierarchical summaries

Build a hierarchy of summaries:

```text
token blocks -> local summaries -> region summaries -> global summaries
```

First select a few large regions, then search only their child blocks. This can
reduce route construction from a full `O(N/S)` scan toward `O(log N)` or a
bounded approximate search.

### Query-dependent routing

Use the actual query vector, or a fused query/key routing projection, instead
of only the mean of the recent keys.

### Adaptive route budgets

Use more blocks when similarity scores are flat or uncertain, and fewer blocks
when one region is clearly dominant.

### Incremental summary updates

Maintain block sums and counts so that appending a token updates a summary in
`O(D)` instead of recomputing the entire active block.

This optimization is implemented in the incremental Pythia prototype. The
remaining route cost is scoring visible summaries and selecting them at each
refresh interval.

### Paged or quantized KV cache

Use paged storage and FP16, BF16, or quantized K/V values to reduce the linear
memory cost of very long contexts.

## Summary

Ocean's attention algorithm replaces full-context attention with:

```text
recent-key summary
        ↓
cosine similarity against block summaries
        ↓
two mandatory local blocks
        ↓
top-5 semantic blocks
        ↓
one exploration block
        ↓
exact causal attention over at most 512 tokens
```

Its main computational benefit is that the attention kernel processes a fixed
token budget instead of all previous tokens. In the measured Pythia 14K stress
test, incremental routing reached a 1.64x decode speedup and a 1.08x total
speedup at `refresh_interval=16`; at the trained 2048-token context it remained
slower than dense end-to-end. The remaining bottlenecks are the global score
evaluation during route refresh, KV-cache management, and the lack of a fused
block-sparse kernel. The full KV cache remains linear in memory.

The design is best characterized as a sublinear-attention research prototype.
The Pythia experiment demonstrates both near-dense perplexity with a reduced
route and a real long-cache decode speedup. It does not provide a comparable
2048-token end-to-end speedup, nor does it establish valid Pythia quality at
14K. The full KV cache remains linear in memory, and the hierarchical route
described above remains an approximation whose quality and implementation speed
must be measured separately from the current Pythia baseline.
