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

## Current limitations

The current implementation has these limitations:

1. The KV cache still grows linearly with context length.
2. Route refresh scans every visible block summary.
3. Prefill repeats route selection for every 50-token chunk.
4. The route budget is fixed rather than adaptive to query uncertainty.
5. One deterministic exploration block is not equivalent to true random
   sampling.
6. Sparse attention is an approximation and can lose long-range information.
7. End-to-end speed is also limited by projections and MLP computation.

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
token budget instead of all previous tokens. Its main remaining bottleneck is
the global summary scan during route refresh, and its main memory cost is still
the full linear KV cache.

The current design is best characterized as a practical sublinear-attention
prototype: it substantially reduces the attention workload, approaches
constant attention cost per token for a fixed route budget, and now uses an
incrementally maintained hierarchy so route refresh no longer performs a
global context scan. The full KV cache remains linear in memory, and the
hierarchical route is still an approximation whose quality must be measured.
