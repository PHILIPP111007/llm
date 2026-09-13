#!/usr/bin/env python3
"""Ablation benchmark for Ocean routing.

Variants:
  dense                    native Pythia attention (only feasible at 2048 here)
  full_scan_cosine         exact scan of all block summaries, cosine Top-X
  hierarchical_cosine     existing Ocean cosine tree, cosine Top-X
  full_scan_reranker       full cosine candidate scan, then neural reranking
  hierarchical_reranker    cosine tree Top-X, then neural reranking

The reranker is deliberately a small model, not a second language model.  It
must be trained by distillation from dense attention before its quality numbers
are scientifically meaningful.  Without --reranker-checkpoint the script
initializes its residual to zero, so its scores are cosine-equivalent and the
reported neural variant is only a routing-overhead diagnostic.

Examples:
  python backend/routing_ablation_benchmark.py --model-dir /path/to/pythia-snapshot
  python backend/routing_ablation_benchmark.py --mode train-reranker --model-dir /path/to/pythia-snapshot
  python backend/routing_ablation_benchmark.py --reranker-checkpoint reranker.pt


./.venv/bin/python backend/routing_ablation_benchmark.py \
  --mode train-reranker \
  --model-dir /home/froschin/.cache/huggingface/hub/models--EleutherAI--pythia-1b/snapshots/f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2 \
  --reranker-output ./checkpoints/block-reranker.pt \
  --reranker-steps 1500 \
  --teacher-mode chunk_union \
  --teacher-chunk-size 256 \
  --teacher-local-window 256


./.venv/bin/python backend/routing_ablation_benchmark.py \
  --model-dir /home/froschin/.cache/huggingface/hub/models--EleutherAI--pythia-1b/snapshots/f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2 \
  --reranker-checkpoint ./checkpoints/block-reranker.pt \
  --output routing_ablation_results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

try:
    from .model import (
        DEVICE,
        DTYPE,
        INT4RoutedKVCache,
        OceanAttention,
        PythiaConfig,
        PythiaForCausalLM,
        ROUTING_CONFIG,
        apply_rotary,
        clear_gpu_cache,
        load_official_weights,
    )
except ImportError:
    from model import (
        DEVICE,
        DTYPE,
        INT4RoutedKVCache,
        OceanAttention,
        PythiaConfig,
        PythiaForCausalLM,
        ROUTING_CONFIG,
        apply_rotary,
        clear_gpu_cache,
        load_official_weights,
    )


torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def synchronize():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


class BlockReranker(nn.Module):
    """Small query/block scorer with a cosine-equivalent safe initialization."""

    def __init__(self, head_dim, rank_dim=128, hidden_dim=256):
        super().__init__()
        self.head_dim = int(head_dim)
        self.rank_dim = int(rank_dim)
        self.query_proj = nn.Linear(head_dim, rank_dim, bias=False)
        self.block_proj = nn.Linear(head_dim, rank_dim, bias=False)
        self.residual = nn.Sequential(
            nn.Linear(rank_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self._safe_initialize()

    def _safe_initialize(self):
        nn.init.normal_(self.query_proj.weight, std=0.02)
        nn.init.normal_(self.block_proj.weight, std=0.02)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, query, summaries):
        """Return scores for query [..., D] and summaries [..., X, D]."""
        query_f = F.normalize(query.float(), dim=-1)
        summaries_f = F.normalize(summaries.float(), dim=-1)
        cosine = (query_f.unsqueeze(-2) * summaries_f).sum(dim=-1)

        q = self.query_proj(query.float()).unsqueeze(-2)
        s = self.block_proj(summaries.float())
        features = torch.cat(
            [
                q.expand_as(s),
                s,
                q * s,
                (q - s).abs(),
            ],
            dim=-1,
        )
        return cosine + self.residual(features).squeeze(-1)


class MultiScaleKVPositionReranker(nn.Module):
    """Reranker using multi-part K/V summaries and relative block position."""

    architecture = "multiscale_kv_position_v1"
    supports_block_metadata = True

    def __init__(self, head_dim, summary_parts=4, rank_dim=128, hidden_dim=384):
        super().__init__()
        self.head_dim = int(head_dim)
        self.summary_parts = int(summary_parts)
        self.rank_dim = int(rank_dim)
        self.hidden_dim = int(hidden_dim)
        self.query_proj = nn.Linear(self.head_dim, self.rank_dim, bias=False)
        self.key_proj = nn.Linear(self.head_dim, self.rank_dim, bias=False)
        self.value_proj = nn.Linear(self.head_dim, self.rank_dim, bias=False)
        feature_dim = self.rank_dim * 9 + 2
        self.residual = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self._safe_initialize()

    def _safe_initialize(self):
        for layer in (self.query_proj, self.key_proj, self.value_proj):
            nn.init.normal_(layer.weight, std=0.02)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self,
        query,
        key_summaries,
        value_summaries=None,
        block_positions=None,
    ):
        """Score blocks from query, multi-part K/V summaries and position."""
        query = query.float()
        if key_summaries.ndim == 3:
            key_summaries = key_summaries.unsqueeze(-2)
        key_summaries = key_summaries.float()
        if value_summaries is None:
            value_summaries = key_summaries
        elif value_summaries.ndim == 3:
            value_summaries = value_summaries.unsqueeze(-2)
        value_summaries = value_summaries.float()

        if key_summaries.shape[-2] != self.summary_parts:
            if key_summaries.shape[-2] == 1:
                key_summaries = key_summaries.expand(
                    *key_summaries.shape[:-2],
                    self.summary_parts,
                    key_summaries.shape[-1],
                )
                value_summaries = value_summaries.expand_as(key_summaries)
            else:
                raise ValueError(
                    "reranker summary_parts не совпадает с checkpoint: "
                    f"{key_summaries.shape[-2]} != {self.summary_parts}"
                )

        q = self.query_proj(query)
        k_parts = self.key_proj(key_summaries)
        v_parts = self.value_proj(value_summaries)
        k_mean = k_parts.mean(dim=-2)
        k_max = k_parts.amax(dim=-2)
        k_first = k_parts[..., 0, :]
        k_last = k_parts[..., -1, :]
        v_mean = v_parts.mean(dim=-2)
        v_max = v_parts.amax(dim=-2)
        block_repr = torch.cat((k_mean, k_max, k_first, k_last), dim=-1)
        q_expanded = q.unsqueeze(-2).expand_as(k_mean)
        features = torch.cat(
            (
                q_expanded,
                block_repr,
                q_expanded * k_mean,
                (q_expanded - k_mean).abs(),
                q_expanded * v_mean,
                (q_expanded - v_mean).abs(),
            ),
            dim=-1,
        )
        if block_positions is None:
            block_positions = torch.zeros(
                features.shape[:-1] + (2,),
                device=features.device,
                dtype=features.dtype,
            )
        else:
            block_positions = block_positions.float()
            if block_positions.ndim == features.ndim - 1:
                block_positions = block_positions.unsqueeze(-1)
            distance = (1.0 - block_positions).clamp_min(0.0)
            log_distance = torch.log1p(distance)
            block_positions = torch.cat((block_positions, log_distance), dim=-1)
        features = torch.cat((features, block_positions), dim=-1)

        cosine = (
            F.normalize(query, dim=-1).unsqueeze(-2)
            * F.normalize(key_summaries.mean(dim=-2), dim=-1)
        ).sum(dim=-1)
        return cosine + self.residual(features).squeeze(-1)


class AblationKVCache(INT4RoutedKVCache):
    """INT4 full cache with selectable first-stage and second-stage routing."""

    def __init__(
        self,
        config,
        max_length,
        routing_mode="hierarchical_cosine",
        candidate_blocks=64,
        reranker=None,
        **kwargs,
    ):
        super().__init__(config, max_length=max_length, **kwargs)
        allowed = {
            "full_scan_cosine",
            "hierarchical_cosine",
            "full_scan_reranker",
            "hierarchical_reranker",
            "neural_full_scan",
        }
        if routing_mode not in allowed:
            raise ValueError(f"unknown routing mode: {routing_mode}")
        self.routing_mode = routing_mode
        self.candidate_blocks = int(candidate_blocks)
        self.reranker = reranker
        self.route_seconds = 0.0
        self.attention_kernel_seconds = 0.0
        self.route_calls = 0
        self.candidate_count_sum = 0

    def reset_benchmark_stats(self):
        self.route_seconds = 0.0
        self.attention_kernel_seconds = 0.0
        self.route_calls = 0
        self.candidate_count_sum = 0

    def _leaf_scores(self, query_vector, block_ids):
        leaf_ids = self.leaf_start + block_ids
        return self._node_scores(query_vector, leaf_ids)

    def _leaf_summaries(self, block_ids):
        leaf_ids = self.leaf_start + block_ids
        gather_ids = leaf_ids[:, :, None, None].expand(
            leaf_ids.shape[0],
            leaf_ids.shape[1],
            self.summary_parts,
            self.head_dim,
        )
        sums = self.tree_sums.gather(1, gather_ids).float()
        counts = self.tree_counts[leaf_ids].float()
        return sums.sum(dim=2) / counts.sum(dim=2).clamp_min(1.0).unsqueeze(-1)

    def _leaf_summary_parts(self, block_ids, tree):
        leaf_ids = self.leaf_start + block_ids
        gather_ids = leaf_ids[:, :, None, None].expand(
            leaf_ids.shape[0],
            leaf_ids.shape[1],
            self.summary_parts,
            self.head_dim,
        )
        sums = tree.gather(1, gather_ids).float()
        counts = self.tree_counts[leaf_ids].float()
        return sums / counts.clamp_min(1.0).unsqueeze(-1)

    def _hierarchical_candidates(self, query_vector, routeable_blocks):
        beam = min(
            max(self.beam_width, self.candidate_blocks),
            routeable_blocks,
        )
        candidates = torch.zeros(
            self.num_heads,
            1,
            device=self.device,
            dtype=torch.long,
        )
        depth = int(math.log2(self.tree_capacity))
        for _ in range(depth):
            left = candidates * 2 + 1
            right = left + 1
            children = torch.cat((left, right), dim=-1)
            scores = self._node_scores(query_vector, children)
            keep = min(beam, children.shape[-1])
            candidates = children.gather(
                -1,
                torch.topk(scores, k=keep, dim=-1).indices,
            )
        block_ids = (candidates - self.leaf_start).clamp(0, routeable_blocks - 1)
        scores = self._leaf_scores(query_vector, block_ids)
        return block_ids, scores

    def _candidate_blocks(self, query_vector, routeable_blocks):
        if self.routing_mode == "neural_full_scan":
            all_blocks = torch.arange(
                routeable_blocks,
                device=self.device,
                dtype=torch.long,
            ).view(1, -1).expand(self.num_heads, -1)
            return all_blocks, torch.zeros_like(all_blocks, dtype=torch.float32)
        candidate_count = min(self.candidate_blocks, routeable_blocks)
        all_blocks = torch.arange(
            routeable_blocks,
            device=self.device,
            dtype=torch.long,
        ).view(1, -1).expand(self.num_heads, -1)
        if self.routing_mode.startswith("full_scan"):
            scores = self._leaf_scores(query_vector, all_blocks)
            top = torch.topk(scores, k=candidate_count, dim=-1).indices
            return all_blocks.gather(1, top), scores.gather(1, top)
        block_ids, scores = self._hierarchical_candidates(
            query_vector,
            routeable_blocks,
        )
        if block_ids.shape[-1] <= candidate_count:
            return block_ids, scores
        top = torch.topk(scores, k=candidate_count, dim=-1).indices
        return block_ids.gather(1, top), scores.gather(1, top)

    @torch.no_grad()
    def _select_route(self, query, length):
        started = time.perf_counter()
        num_blocks = length // self.block_size
        local_start = self._aligned_local_start(length)
        routeable_blocks = min(num_blocks, local_start // self.block_size)
        if routeable_blocks <= 0:
            return torch.empty(
                1,
                self.num_heads,
                0,
                device=self.device,
                dtype=torch.long,
            )

        route_count = min(self.route_blocks, routeable_blocks)
        local_count = min(self.local_blocks, route_count)
        global_count = min(
            self.global_blocks,
            max(0, route_count - local_count),
        )
        global_ids = torch.arange(
            global_count,
            device=self.device,
            dtype=torch.long,
        )
        local_ids = torch.arange(
            routeable_blocks - local_count,
            routeable_blocks,
            device=self.device,
            dtype=torch.long,
        )
        mandatory = torch.cat((global_ids, local_ids), dim=0)
        semantic_count = route_count - mandatory.numel()
        if semantic_count <= 0:
            result = mandatory.view(1, 1, -1).expand(1, self.num_heads, -1)
            self.route_seconds += time.perf_counter() - started
            self.route_calls += 1
            return result

        query_vector = query[:, :, 0, :].reshape(self.num_heads, self.head_dim)
        candidate_ids, cosine_scores = self._candidate_blocks(
            query_vector,
            routeable_blocks,
        )
        self.candidate_count_sum += int(candidate_ids.shape[-1])

        if (
            self.routing_mode.endswith("reranker")
            or self.routing_mode == "neural_full_scan"
        ):
            if self.reranker is None:
                raise RuntimeError(
                    "neural reranker mode requires a trained or initialized reranker"
                )
            if getattr(self.reranker, "supports_block_metadata", False):
                key_parts = self._leaf_summary_parts(
                    candidate_ids,
                    self.tree_sums,
                )
                value_parts = self._leaf_summary_parts(
                    candidate_ids,
                    self.value_tree_sums,
                )
                positions = candidate_ids.float() / max(1, routeable_blocks - 1)
                scores = self.reranker(
                    query_vector,
                    key_parts,
                    value_parts,
                    positions,
                )
            else:
                candidate_summaries = self._leaf_summaries(candidate_ids)
                scores = self.reranker(query_vector, candidate_summaries)
        else:
            scores = cosine_scores

        blocked = (candidate_ids[:, :, None] == mandatory.view(1, 1, -1)).any(dim=-1)
        scores = scores.masked_fill(blocked, float("-inf"))
        keep = min(semantic_count, candidate_ids.shape[-1])
        selected = candidate_ids.gather(
            1,
            torch.topk(scores, k=keep, dim=-1).indices,
        )
        result = torch.cat(
            [
                mandatory.view(1, 1, -1).expand(1, self.num_heads, -1),
                selected.unsqueeze(0),
            ],
            dim=-1,
        )
        self.route_seconds += time.perf_counter() - started
        self.route_calls += 1
        return result


class AblationAttention(OceanAttention):
    """Ocean attention with timing around the selected attention kernel."""

    @torch.inference_mode()
    def forward_token(self, hidden_states, cache, position):
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(1, 1, self.num_attention_heads, 3 * self.head_dim)
        qkv = qkv.transpose(1, 2)
        query, key, value = qkv.chunk(3, dim=-1)
        position_ids = torch.tensor(
            [position],
            device=hidden_states.device,
            dtype=torch.long,
        )
        cos, sin = self.rotary_emb(position_ids, hidden_states.dtype)
        query, key = apply_rotary(
            query,
            key,
            cos,
            sin,
            self.rotary_ndims,
        )
        cache.append(key, value, position)
        route_key, route_value = cache.attention_kv(query, position)
        synchronize()
        started = time.perf_counter()
        output = F.scaled_dot_product_attention(
            query,
            route_key,
            route_value,
            dropout_p=0.0,
            is_causal=False,
        )
        synchronize()
        cache.attention_kernel_seconds += time.perf_counter() - started
        output = output.transpose(1, 2).contiguous().view(1, 1, -1)
        return self.dense(output)

    @torch.inference_mode()
    def forward_chunk(self, hidden_states, cache, start_position):
        q_len = hidden_states.shape[1]
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(1, q_len, self.num_attention_heads, 3 * self.head_dim)
        qkv = qkv.transpose(1, 2)
        query, key, value = qkv.chunk(3, dim=-1)
        positions = torch.arange(
            start_position,
            start_position + q_len,
            device=hidden_states.device,
            dtype=torch.long,
        )
        cos, sin = self.rotary_emb(positions, hidden_states.dtype)
        query, key = apply_rotary(
            query,
            key,
            cos,
            sin,
            self.rotary_ndims,
        )
        past_key, past_value = cache.attention_kv(
            query[:, :, -1:, :],
            start_position - 1,
        )
        past_len = past_key.shape[2]
        route_key = torch.cat((past_key, key), dim=2)
        route_value = torch.cat((past_value, value), dim=2)
        allowed = torch.ones(
            q_len,
            past_len + q_len,
            device=hidden_states.device,
            dtype=torch.bool,
        )
        allowed[:, past_len:] = torch.tril(
            torch.ones(
                q_len,
                q_len,
                device=hidden_states.device,
                dtype=torch.bool,
            )
        )
        synchronize()
        started = time.perf_counter()
        output = F.scaled_dot_product_attention(
            query,
            route_key,
            route_value,
            attn_mask=allowed[None, None, :, :],
            dropout_p=0.0,
            is_causal=False,
        )
        synchronize()
        cache.attention_kernel_seconds += time.perf_counter() - started
        cache.append_chunk(key, value, start_position)
        output = output.transpose(1, 2).contiguous().view(1, q_len, -1)
        return self.dense(output)


class AblationModel(PythiaForCausalLM):
    def __init__(
        self,
        config,
        routing_mode,
        candidate_blocks,
        reranker=None,
        routing=None,
    ):
        super().__init__(config)
        self.routing_mode = routing_mode
        self.candidate_blocks = int(candidate_blocks)
        self.routing = dict(routing or ROUTING_CONFIG)
        # Reranker is an external inference component with its own checkpoint.
        # Bypass nn.Module.__setattr__ so official Pythia weight loading does
        # not expect reranker.* keys in the HuggingFace state dict.
        object.__setattr__(self, "reranker", reranker)
        for layer in self.gpt_neox.layers:
            layer.attention = AblationAttention(config)

    def new_bounded_cache(self, max_length):
        return [
            AblationKVCache(
                self.config,
                max_length=max_length,
                routing_mode=self.routing_mode,
                candidate_blocks=self.candidate_blocks,
                reranker=self.reranker,
                device=DEVICE,
                **self.routing,
            )
            for _ in self.gpt_neox.layers
        ]

    @torch.inference_mode()
    def forward_bounded_chunk(self, input_ids, caches, start_position):
        hidden_states = self.gpt_neox.embed_in(input_ids.view(1, -1))
        for layer, cache in zip(self.gpt_neox.layers, caches):
            residual = hidden_states
            attention_input = layer.input_layernorm(hidden_states)
            attention_output = layer.attention.forward_chunk(
                attention_input,
                cache,
                start_position,
            )
            if layer.use_parallel_residual:
                hidden_states = residual + attention_output + layer.mlp(
                    layer.post_attention_layernorm(hidden_states)
                )
            else:
                hidden_states = residual + attention_output
                hidden_states = hidden_states + layer.mlp(
                    layer.post_attention_layernorm(hidden_states)
                )
        hidden_states = self.gpt_neox.final_layer_norm(hidden_states)
        return self.embed_out(hidden_states)

    @torch.inference_mode()
    def forward_bounded_token(self, input_ids, caches, position):
        hidden_states = self.gpt_neox.embed_in(input_ids.view(1, 1))
        for layer, cache in zip(self.gpt_neox.layers, caches):
            residual = hidden_states
            attention_input = layer.input_layernorm(hidden_states)
            attention_output = layer.attention.forward_token(
                attention_input,
                cache,
                position,
            )
            if layer.use_parallel_residual:
                hidden_states = residual + attention_output + layer.mlp(
                    layer.post_attention_layernorm(hidden_states)
                )
            else:
                hidden_states = residual + attention_output
                hidden_states = hidden_states + layer.mlp(
                    layer.post_attention_layernorm(hidden_states)
                )
        hidden_states = self.gpt_neox.final_layer_norm(hidden_states)
        return self.embed_out(hidden_states)


def load_reranker(path, head_dim):
    reranker = None
    if path is not None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        metadata = state if isinstance(state, dict) else {}
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        if metadata.get("architecture") == MultiScaleKVPositionReranker.architecture:
            reranker = MultiScaleKVPositionReranker(
                head_dim,
                summary_parts=int(metadata.get("summary_parts", 4)),
                rank_dim=int(metadata.get("rank_dim", 128)),
                hidden_dim=int(metadata.get("hidden_dim", 384)),
            )
        else:
            reranker = BlockReranker(head_dim)
        reranker.load_state_dict(state, strict=True)
        print("loaded reranker:", path)
    else:
        reranker = BlockReranker(head_dim)
        print(
            "WARNING: reranker checkpoint is absent; neural residual is zero. "
            "This is an overhead diagnostic, not a quality claim."
        )
    reranker = reranker.to(device=DEVICE, dtype=torch.float32)
    reranker.eval()
    return reranker


@torch.inference_mode()
def collect_reranker_teacher_samples(
    dense_model,
    sequences,
    block_size=256,
    query_stride=128,
    summary_parts=1,
    chunk_size=256,
    local_window=256,
    teacher_mode="chunk_union",
    return_multiscale=False,
):
    """Collect dense-attention block-mass targets for reranker distillation.

    ``chunk_union`` is the production-aligned protocol.  Inference computes one
    route for a whole prefill chunk, while local/global blocks are mandatory.
    Therefore the teacher uses the last query in the chunk as the reranker's
    query and averages dense attention mass over several queries in that same
    chunk.  Only complete blocks before the local window are labelled; the
    mandatory local/global blocks are not part of the target.

    ``single_query`` keeps the old protocol for controlled comparisons.  It is
    intentionally not the default because it does not match chunk prefill.
    """
    if block_size % summary_parts != 0:
        raise ValueError("block_size должен делиться на summary_parts")
    if chunk_size <= 0 or query_stride <= 0:
        raise ValueError("chunk_size и query_stride должны быть положительными")
    if local_window < 0:
        raise ValueError("local_window не может быть отрицательным")
    if teacher_mode not in {"chunk_union", "single_query"}:
        raise ValueError(
            "teacher_mode должен быть chunk_union или single_query"
        )
    query_rows = []
    summary_rows = []
    value_rows = []
    position_rows = []
    target_rows = []
    dense_model.eval()
    for sequence in sequences:
        ids = sequence[:2048].to(DEVICE)
        if ids.numel() < block_size * 3:
            continue
        hidden_states = dense_model.gpt_neox.embed_in(ids.view(1, -1))
        positions = torch.arange(
            ids.numel(),
            device=DEVICE,
            dtype=torch.long,
        )
        for layer in dense_model.gpt_neox.layers:
            attention_input = layer.input_layernorm(hidden_states)
            qkv = layer.attention.query_key_value(attention_input)
            qkv = qkv.view(
                1,
                ids.numel(),
                dense_model.config.num_attention_heads,
                3 * dense_model.config.head_dim,
            ).transpose(1, 2)
            query, key, value = qkv.chunk(3, dim=-1)
            cos, sin = layer.attention.rotary_emb(
                positions,
                hidden_states.dtype,
            )
            query, key = apply_rotary(
                query,
                key,
                cos,
                sin,
                dense_model.config.rotary_ndims,
            )
            if teacher_mode == "single_query":
                sample_starts = range(
                    block_size * 2,
                    ids.numel(),
                    max(1, query_stride),
                )
            else:
                sample_starts = range(
                    block_size * 2,
                    ids.numel(),
                    max(1, chunk_size),
                )
            for chunk_start in sample_starts:
                chunk_end = min(
                    ids.numel(),
                    chunk_start + (1 if teacher_mode == "single_query" else chunk_size),
                )
                query_positions = list(
                    range(
                        chunk_start,
                        chunk_end,
                        max(1, query_stride),
                    )
                )
                if not query_positions:
                    continue
                if query_positions[-1] != chunk_end - 1:
                    query_positions.append(chunk_end - 1)

                if teacher_mode == "single_query":
                    routeable_end = (chunk_end // block_size) * block_size
                else:
                    local_start = max(
                        0,
                        ((chunk_start - local_window) // block_size)
                        * block_size,
                    )
                    routeable_end = local_start
                block_count = routeable_end // block_size
                if block_count < 2:
                    continue
                usable = block_count * block_size
                block_keys = key[:, :, :usable, :].view(
                    1,
                    dense_model.config.num_attention_heads,
                    block_count,
                    block_size,
                    dense_model.config.head_dim,
                )
                block_values = value[:, :, :usable, :].view(
                    1,
                    dense_model.config.num_attention_heads,
                    block_count,
                    block_size,
                    dense_model.config.head_dim,
                )
                if summary_parts == 1:
                    summaries = block_keys.float().mean(dim=3)[0]
                    value_summaries = block_values.float().mean(dim=3)[0]
                else:
                    part_size = block_size // summary_parts
                    summaries = block_keys.float().view(
                        1,
                        dense_model.config.num_attention_heads,
                        block_count,
                        summary_parts,
                        part_size,
                        dense_model.config.head_dim,
                    ).mean(dim=4)[0]
                    value_summaries = block_values.float().view(
                        1,
                        dense_model.config.num_attention_heads,
                        block_count,
                        summary_parts,
                        part_size,
                        dense_model.config.head_dim,
                    ).mean(dim=4)[0]
                query_vectors = query[0, :, query_positions, :].float()
                token_scores = torch.einsum(
                    "hqd,hld->hql",
                    query_vectors,
                    key[0, :, :usable, :].float(),
                ) / math.sqrt(dense_model.config.head_dim)
                token_scores = token_scores - token_scores.amax(
                    dim=-1,
                    keepdim=True,
                )
                token_mass = token_scores.exp()
                block_mass = token_mass.reshape(
                    dense_model.config.num_attention_heads,
                    len(query_positions),
                    block_count,
                    block_size,
                ).sum(dim=-1)
                block_mass = block_mass / block_mass.sum(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-8)
                block_mass = block_mass.mean(dim=1)
                query_rows.append(
                    query[0, :, query_positions[-1], :].float().cpu()
                )
                summary_rows.append(summaries.cpu())
                value_rows.append(value_summaries.cpu())
                position_rows.append(
                    torch.arange(block_count, dtype=torch.float32)
                    / max(1, block_count - 1)
                )
                target_rows.append(block_mass.float().cpu())
            hidden_states, _ = layer(
                hidden_states,
                past_key_value=None,
                use_cache=False,
            )
    if not query_rows:
        raise RuntimeError("teacher не собрал ни одного reranker sample")
    max_blocks = max(item.shape[1] for item in summary_rows)
    head_dim = query_rows[0].shape[-1]
    heads = query_rows[0].shape[0]
    queries = torch.stack(query_rows)
    if summary_parts == 1:
        summaries = torch.zeros(
            len(summary_rows),
            heads,
            max_blocks,
            head_dim,
            dtype=torch.float32,
        )
        value_summaries = torch.zeros_like(summaries)
    else:
        summaries = torch.zeros(
            len(summary_rows),
            heads,
            max_blocks,
            summary_parts,
            head_dim,
            dtype=torch.float32,
        )
        value_summaries = torch.zeros_like(summaries)
    positions = torch.zeros(
        len(position_rows),
        max_blocks,
        dtype=torch.float32,
    )
    targets = torch.zeros(
        len(target_rows),
        heads,
        max_blocks,
        dtype=torch.float32,
    )
    valid = torch.zeros_like(targets, dtype=torch.bool)
    for index, (summary, value_summary, position, target) in enumerate(
        zip(summary_rows, value_rows, position_rows, target_rows)
    ):
        block_count = summary.shape[1]
        if summary_parts == 1:
            summaries[index, :, :block_count, :] = summary
            value_summaries[index, :, :block_count, :] = value_summary
        else:
            summaries[index, :, :block_count, :, :] = summary
            value_summaries[index, :, :block_count, :, :] = value_summary
        positions[index, :block_count] = position
        targets[index, :, :block_count] = target
        valid[index, :, :block_count] = True
    if return_multiscale:
        return queries, summaries, value_summaries, positions, targets, valid
    return queries, summaries, targets, valid


def train_reranker(
    model_dir,
    tokenizer,
    filler_ids,
    output_path,
    steps=500,
    sequence_count=8,
    query_stride=128,
    batch_size=8,
    chunk_size=256,
    local_window=256,
    teacher_mode="chunk_union",
    teacher_block_size=None,
    teacher_summary_parts=None,
):
    """Distill the multiscale K/V reranker from dense Pythia attention."""
    if teacher_block_size is None:
        teacher_block_size = int(ROUTING_CONFIG["block_size"])
    if teacher_summary_parts is None:
        teacher_summary_parts = int(ROUTING_CONFIG["summary_parts"])
    dense_model = load_base_model(model_dir)
    sequences = []
    for index in range(sequence_count):
        start = index * 2048
        sequence = repeat_ids(filler_ids, start + 2048)[start:]
        sequences.append(sequence)
    print("collecting dense teacher labels")
    summary_parts = int(teacher_summary_parts)
    (
        queries,
        key_summaries,
        value_summaries,
        positions,
        targets,
        valid,
    ) = collect_reranker_teacher_samples(
        dense_model,
        sequences,
        block_size=int(teacher_block_size),
        query_stride=query_stride,
        summary_parts=summary_parts,
        chunk_size=chunk_size,
        local_window=local_window,
        teacher_mode=teacher_mode,
        return_multiscale=True,
    )
    del dense_model
    clear_gpu_cache()

    reranker = MultiScaleKVPositionReranker(
        PythiaConfig().head_dim,
        summary_parts=summary_parts,
    ).to(
        device=DEVICE,
        dtype=torch.float32,
    )
    optimizer = torch.optim.AdamW(
        reranker.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )
    sample_count = queries.shape[0]
    reranker.train()
    started = time.perf_counter()
    for step in range(steps):
        indices = torch.randint(
            0,
            sample_count,
            (min(batch_size, sample_count),),
        )
        query_batch = queries[indices].to(DEVICE)
        key_batch = key_summaries[indices].to(DEVICE)
        value_batch = value_summaries[indices].to(DEVICE)
        position_batch = positions[indices].to(DEVICE)
        target_batch = targets[indices].to(DEVICE)
        valid_batch = valid[indices].to(DEVICE)
        query_flat = query_batch.reshape(-1, query_batch.shape[-1])
        key_flat = key_batch.reshape(
            -1,
            key_batch.shape[-3],
            key_batch.shape[-2],
            key_batch.shape[-1],
        )
        value_flat = value_batch.reshape(
            -1,
            value_batch.shape[-3],
            value_batch.shape[-2],
            value_batch.shape[-1],
        )
        position_flat = position_batch[:, None, :].expand(
            -1,
            query_batch.shape[1],
            -1,
        ).reshape(
            -1,
            position_batch.shape[-1],
        )
        target_flat = target_batch.reshape(-1, target_batch.shape[-1])
        valid_flat = valid_batch.reshape(-1, valid_batch.shape[-1])
        scores = reranker(
            query_flat,
            key_flat,
            value_flat,
            position_flat,
        )
        scores = scores.masked_fill(~valid_flat, float("-inf"))
        log_probs = F.log_softmax(scores, dim=-1)
        listwise_loss = -(target_flat * log_probs).masked_fill(
            ~valid_flat,
            0.0,
        ).sum(dim=-1).mean()
        positive_index = target_flat.argmax(dim=-1)
        positive_score = scores.gather(1, positive_index[:, None]).squeeze(1)
        cosine = (
            F.normalize(query_flat, dim=-1).unsqueeze(1)
            * F.normalize(key_flat.mean(dim=-2), dim=-1)
        ).sum(dim=-1)
        negative_mask = valid_flat.clone()
        negative_mask.scatter_(1, positive_index[:, None], False)
        hard_negative = cosine.masked_fill(~negative_mask, float("-inf")).argmax(dim=-1)
        negative_score = scores.gather(1, hard_negative[:, None]).squeeze(1)
        ranking_loss = F.relu(0.10 - positive_score + negative_score).mean()
        loss = listwise_loss + 0.25 * ranking_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reranker.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 50 == 0 or step + 1 == steps:
            print(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "listwise_loss": float(listwise_loss.detach()),
                    "ranking_loss": float(ranking_loss.detach()),
                    "teacher_samples": sample_count,
                    "seconds": time.perf_counter() - started,
                }
            )
    reranker.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": reranker.state_dict(),
            "architecture": MultiScaleKVPositionReranker.architecture,
            "head_dim": PythiaConfig().head_dim,
            "rank_dim": reranker.rank_dim,
            "hidden_dim": reranker.hidden_dim,
            "summary_parts": summary_parts,
            "teacher": "dense Pythia attention block mass with hard-negative ranking",
            "routing_config": {
                **ROUTING_CONFIG,
                "block_size": int(teacher_block_size),
                "summary_parts": int(summary_parts),
            },
            "steps": steps,
            "sequence_count": sequence_count,
            "query_stride": query_stride,
            "chunk_size": chunk_size,
            "local_window": local_window,
            "teacher_mode": teacher_mode,
        },
        output_path,
    )
    print("saved reranker:", output_path)
    return output_path


def repeat_ids(ids, length):
    ids = ids.flatten().tolist()
    if not ids:
        raise ValueError("filler token sequence is empty")
    repeats = (length + len(ids) - 1) // len(ids)
    return torch.tensor((ids * repeats)[:length], dtype=torch.long)


def make_needle_prompt(tokenizer, filler_ids, prompt_length):
    needle = tokenizer(
        " The secret code is ORBIT-314159.",
        add_special_tokens=False,
    ).input_ids
    query = tokenizer(
        "\nQuestion: What is the secret code? Answer:",
        add_special_tokens=False,
    ).input_ids
    position = max(1, prompt_length // 4)
    if position + len(needle) + len(query) > prompt_length:
        position = max(1, prompt_length - len(needle) - len(query))
    prefix = repeat_ids(filler_ids, position)
    suffix_length = prompt_length - len(prefix) - len(needle) - len(query)
    suffix = repeat_ids(filler_ids, max(0, suffix_length))
    prompt = torch.cat(
        [
            prefix,
            torch.tensor(needle, dtype=torch.long),
            suffix,
            torch.tensor(query, dtype=torch.long),
        ]
    )[:prompt_length]
    answer = torch.tensor(
        tokenizer(
            " ORBIT-314159",
            add_special_tokens=False,
        ).input_ids,
        dtype=torch.long,
    )
    return prompt, answer, position


@torch.inference_mode()
def routed_ppl(model, ids, context_length, chunk_size):
    ids = ids[:context_length].to(DEVICE)
    caches = model.new_bounded_cache(ids.numel() + 32)
    logits_parts = []
    for start in range(0, ids.numel(), chunk_size):
        chunk = ids[start : start + chunk_size]
        logits_parts.append(
            model.forward_bounded_chunk(
                chunk,
                caches,
                start,
            ).float().cpu()
        )
    logits = torch.cat(logits_parts, dim=1)
    nll = F.cross_entropy(
        logits[:, :-1].reshape(-1, model.config.vocab_size),
        ids[1:].cpu(),
    )
    stats = cache_stats(caches)
    return {
        "tokens": int(ids.numel() - 1),
        "mean_nll": float(nll),
        "perplexity": float(torch.exp(nll)),
        **stats,
    }


@torch.inference_mode()
def dense_ppl(model, ids, context_length):
    ids = ids[:context_length].to(DEVICE)
    started = time.perf_counter()
    logits, _ = model(ids.view(1, -1), use_cache=False)
    synchronize()
    nll = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, model.config.vocab_size),
        ids[1:].view(-1),
    )
    return {
        "tokens": int(ids.numel() - 1),
        "mean_nll": float(nll),
        "perplexity": float(torch.exp(nll)),
        "seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def dense_speed(model, ids, prompt_length, new_tokens):
    ids = ids[:prompt_length].to(DEVICE)
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    synchronize()
    prefill_start = time.perf_counter()
    logits, past = model(ids.view(1, -1), use_cache=True)
    next_token = logits[:, -1:].argmax(dim=-1)
    synchronize()
    prefill_seconds = time.perf_counter() - prefill_start

    synchronize()
    decode_start = time.perf_counter()
    for _ in range(new_tokens):
        logits, past = model(
            next_token,
            past_key_values=past,
            use_cache=True,
        )
        next_token = logits[:, -1:].argmax(dim=-1)
    synchronize()
    decode_seconds = time.perf_counter() - decode_start
    result = {
        "routing": "dense",
        "prompt_length": int(prompt_length),
        "new_tokens": int(new_tokens),
        "prefill_seconds": prefill_seconds,
        "prefill_tokens_per_second": prompt_length / prefill_seconds,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": new_tokens / decode_seconds,
        "total_seconds": prefill_seconds + decode_seconds,
    }
    if DEVICE.type == "cuda":
        result["peak_cuda_allocated_gib"] = (
            torch.cuda.max_memory_allocated() / 2**30
        )
        result["peak_cuda_reserved_gib"] = (
            torch.cuda.max_memory_reserved() / 2**30
        )
    return result


def cache_stats(caches):
    route_seconds = sum(cache.route_seconds for cache in caches)
    attention_seconds = sum(
        cache.attention_kernel_seconds for cache in caches
    )
    route_calls = sum(cache.route_calls for cache in caches)
    candidate_sum = sum(cache.candidate_count_sum for cache in caches)
    return {
        "routing_seconds": route_seconds,
        "attention_kernel_seconds": attention_seconds,
        "route_calls_all_layers": route_calls,
        "mean_candidates_per_route": (
            candidate_sum / route_calls if route_calls else 0.0
        ),
    }


@torch.inference_mode()
def routed_speed(model, ids, context_length, chunk_size, new_tokens):
    ids = ids[:context_length].to(DEVICE)
    caches = model.new_bounded_cache(ids.numel() + new_tokens + 8)
    for cache in caches:
        cache.reset_benchmark_stats()
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    synchronize()
    prefill_start = time.perf_counter()
    next_token = None
    for start in range(0, ids.numel(), chunk_size):
        chunk = ids[start : start + chunk_size]
        logits = model.forward_bounded_chunk(chunk, caches, start)
        next_token = logits[:, -1:].argmax(dim=-1)
    synchronize()
    prefill_seconds = time.perf_counter() - prefill_start

    synchronize()
    decode_start = time.perf_counter()
    position = ids.numel()
    for _ in range(new_tokens):
        logits = model.forward_bounded_token(next_token, caches, position)
        next_token = logits[:, -1:].argmax(dim=-1)
        position += 1
    synchronize()
    decode_seconds = time.perf_counter() - decode_start
    result = {
        "prompt_length": int(ids.numel()),
        "chunk_size": int(chunk_size),
        "new_tokens": int(new_tokens),
        "prefill_seconds": prefill_seconds,
        "prefill_tokens_per_second": ids.numel() / prefill_seconds,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": new_tokens / decode_seconds,
        "total_seconds": prefill_seconds + decode_seconds,
        "routing": model.routing_mode,
        **cache_stats(caches),
    }
    if DEVICE.type == "cuda":
        result["peak_cuda_allocated_gib"] = (
            torch.cuda.max_memory_allocated() / 2**30
        )
        result["peak_cuda_reserved_gib"] = (
            torch.cuda.max_memory_reserved() / 2**30
        )
    return result


@torch.inference_mode()
def routed_needle(model, tokenizer, filler_ids, prompt_length, chunk_size):
    prompt, answer, needle_position = make_needle_prompt(
        tokenizer,
        filler_ids,
        prompt_length,
    )
    prompt = prompt.to(DEVICE)
    answer = answer.to(DEVICE)
    caches = model.new_bounded_cache(prompt.numel() + answer.numel() + 8)
    for start in range(0, prompt.numel(), chunk_size):
        logits = model.forward_bounded_chunk(
            prompt[start : start + chunk_size],
            caches,
            start,
        )
    current_logits = logits[:, -1, :].float()
    losses = []
    generated = []
    position = prompt.numel()
    for target in answer:
        losses.append(F.cross_entropy(current_logits, target.view(1)))
        predicted = current_logits.argmax(dim=-1)
        generated.append(predicted.item())
        current_logits = model.forward_bounded_token(
            target.view(1, 1),
            caches,
            position,
        )[:, -1, :].float()
        position += 1
    mean_nll = torch.stack(losses).mean()
    decoded = tokenizer.decode(generated)
    target_text = tokenizer.decode(answer.tolist())
    return {
        "prompt_length": int(prompt_length),
        "needle_position": int(needle_position),
        "answer_mean_nll": float(mean_nll),
        "answer_perplexity": float(torch.exp(mean_nll)),
        "text_exact_match": decoded == target_text,
        "predicted_answer": decoded,
        "target_answer": target_text,
        "routing": model.routing_mode,
    }


@torch.inference_mode()
def dense_needle(model, tokenizer, filler_ids, prompt_length):
    prompt, answer, needle_position = make_needle_prompt(
        tokenizer,
        filler_ids,
        prompt_length,
    )
    prompt = prompt.to(DEVICE)
    answer = answer.to(DEVICE)
    logits, past = model(prompt.view(1, -1), use_cache=True)
    current_logits = logits[:, -1, :].float()
    losses = []
    generated = []
    for target in answer:
        losses.append(F.cross_entropy(current_logits, target.view(1)))
        generated.append(current_logits.argmax(dim=-1).item())
        current_logits, past = model(
            target.view(1, 1),
            past_key_values=past,
            use_cache=True,
        )
        current_logits = current_logits[:, -1, :].float()
    mean_nll = torch.stack(losses).mean()
    decoded = tokenizer.decode(generated)
    target_text = tokenizer.decode(answer.tolist())
    return {
        "routing": "dense",
        "prompt_length": int(prompt_length),
        "needle_position": int(needle_position),
        "answer_mean_nll": float(mean_nll),
        "answer_perplexity": float(torch.exp(mean_nll)),
        "text_exact_match": decoded == target_text,
        "predicted_answer": decoded,
        "target_answer": target_text,
    }


def read_filler(path):
    text = Path(path).read_text(encoding="utf-8")
    return text


def build_model(
    model_dir,
    routing_mode,
    candidate_blocks,
    reranker,
    routing=None,
):
    model = AblationModel(
        PythiaConfig(),
        routing_mode=routing_mode,
        candidate_blocks=candidate_blocks,
        reranker=reranker,
        routing=routing,
    )
    load_official_weights(model, model_dir)
    return model.to(device=DEVICE, dtype=DTYPE).eval()


def load_base_model(model_dir):
    model = PythiaForCausalLM(PythiaConfig())
    load_official_weights(model, model_dir)
    return model.to(device=DEVICE, dtype=DTYPE).eval()


def report_quality(
    model_dir,
    ids,
    reranker,
    contexts,
    chunk_size,
    modes,
    routing=None,
):
    rows = []
    for mode in modes:
        if mode == "dense":
            model = load_base_model(model_dir)
            for length in contexts:
                if length > 2048:
                    rows.append(
                        {
                            "routing": "dense",
                            "context_length": length,
                            "status": "skipped_dense_native_context_limit",
                        }
                    )
                    continue
                rows.append(
                    {
                        "routing": "dense",
                        "context_length": length,
                        **dense_ppl(model, ids, length),
                    }
                )
            del model
            clear_gpu_cache()
            continue
        model = build_model(
            model_dir,
            mode,
            candidate_blocks=64,
            reranker=(
                reranker
                if mode.endswith("reranker") or mode == "neural_full_scan"
                else None
            ),
            routing=routing,
        )
        for length in contexts:
            rows.append(
                {
                    "routing": mode,
                    "context_length": length,
                    **routed_ppl(model, ids, length, chunk_size),
                }
            )
        del model
        clear_gpu_cache()
    return rows


def report_speed(
    model_dir,
    ids,
    reranker,
    contexts,
    chunk_size,
    new_tokens,
    modes,
    routing=None,
):
    rows = []
    for mode in modes:
        if mode == "dense":
            model = load_base_model(model_dir)
            rows.append(dense_speed(model, ids, 2048, new_tokens))
            del model
            clear_gpu_cache()
            continue
        model = build_model(
            model_dir,
            mode,
            candidate_blocks=64,
            reranker=(
                reranker
                if mode.endswith("reranker") or mode == "neural_full_scan"
                else None
            ),
            routing=routing,
        )
        for length in contexts:
            rows.append(
                routed_speed(
                    model,
                    ids,
                    length,
                    chunk_size,
                    new_tokens,
                )
            )
        del model
        clear_gpu_cache()
    return rows


def report_needle(
    model_dir,
    tokenizer,
    filler_ids,
    reranker,
    contexts,
    chunk_size,
    modes,
    routing=None,
):
    rows = []
    for mode in modes:
        if mode == "dense":
            model = load_base_model(model_dir)
            rows.append(dense_needle(model, tokenizer, filler_ids, 2048))
            del model
            clear_gpu_cache()
            continue
        model = build_model(
            model_dir,
            mode,
            candidate_blocks=64,
            reranker=(
                reranker
                if mode.endswith("reranker") or mode == "neural_full_scan"
                else None
            ),
            routing=routing,
        )
        for length in contexts:
            rows.append(
                routed_needle(
                    model,
                    tokenizer,
                    filler_ids,
                    length,
                    chunk_size,
                )
            )
        del model
        clear_gpu_cache()
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("benchmark", "train-reranker"), default="benchmark")
    parser.add_argument("--model-id", default="EleutherAI/pythia-1b")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--reranker-checkpoint", default=None)
    parser.add_argument("--text-file", default="/home/froschin/work/llm/tinyshakespeare.txt")
    parser.add_argument("--contexts", default="2048,8192,32768,100000")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--output", default="routing_ablation_results.json")
    parser.add_argument(
        "--reranker-output",
        default="reranker_distilled.pt",
    )
    parser.add_argument("--reranker-steps", type=int, default=500)
    parser.add_argument("--reranker-sequences", type=int, default=8)
    parser.add_argument("--reranker-query-stride", type=int, default=128)
    parser.add_argument("--teacher-chunk-size", type=int, default=256)
    parser.add_argument("--teacher-local-window", type=int, default=256)
    parser.add_argument("--teacher-block-size", type=int, default=None)
    parser.add_argument("--teacher-summary-parts", type=int, default=None)
    parser.add_argument(
        "--teacher-mode",
        choices=("chunk_union", "single_query"),
        default="chunk_union",
        help=(
            "dense teacher protocol; chunk_union matches production prefill, "
            "single_query is the legacy control"
        ),
    )
    parser.add_argument("--routing-block-size", type=int, default=None)
    parser.add_argument("--routing-summary-parts", type=int, default=None)
    parser.add_argument("--skip-ppl", action="store_true")
    parser.add_argument("--skip-speed", action="store_true")
    parser.add_argument("--skip-needle", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.model_dir is None:
        model_dir = Path(
            snapshot_download(
                repo_id=args.model_id,
                allow_patterns=[
                    "config.json",
                    "tokenizer*",
                    "*.json",
                    "*.safetensors",
                    "*.bin",
                ],
            )
        )
    else:
        model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    filler_ids = torch.tensor(
        tokenizer(
            read_filler(args.text_file),
            add_special_tokens=False,
        ).input_ids,
        dtype=torch.long,
    )
    if args.mode == "train-reranker":
        train_reranker(
            model_dir=model_dir,
            tokenizer=tokenizer,
            filler_ids=filler_ids,
            output_path=args.reranker_output,
            steps=args.reranker_steps,
            sequence_count=args.reranker_sequences,
            query_stride=args.reranker_query_stride,
            chunk_size=args.teacher_chunk_size,
            local_window=args.teacher_local_window,
            teacher_mode=args.teacher_mode,
            teacher_block_size=args.teacher_block_size,
            teacher_summary_parts=args.teacher_summary_parts,
        )
        return

    contexts = tuple(int(item) for item in args.contexts.split(","))
    modes = (
        "dense",
        "full_scan_cosine",
        "hierarchical_cosine",
        "full_scan_reranker",
        "hierarchical_reranker",
        "neural_full_scan",
    )
    reranker = load_reranker(
        args.reranker_checkpoint,
        PythiaConfig().head_dim,
    )
    routing = dict(ROUTING_CONFIG)
    if args.routing_block_size is not None:
        routing["block_size"] = int(args.routing_block_size)
    if args.routing_summary_parts is not None:
        routing["summary_parts"] = int(args.routing_summary_parts)
    elif getattr(reranker, "supports_block_metadata", False):
        routing["summary_parts"] = int(reranker.summary_parts)
    if routing["block_size"] % routing["summary_parts"] != 0:
        raise ValueError(
            "routing block_size должен делиться на summary_parts"
        )
    result = {
        "protocol": {
            "contexts": contexts,
            "candidate_blocks": 64,
            "route_blocks": routing["route_blocks"],
            "block_size": routing["block_size"],
            "summary_parts": routing["summary_parts"],
            "route_refresh_interval": routing["route_refresh_interval"],
            "modes": modes,
            "reranker_checkpoint": args.reranker_checkpoint,
            "warning": (
                "neural variants are cosine-equivalent overhead diagnostics when "
                "reranker_checkpoint is absent"
            ),
        }
    }
    if not args.skip_ppl:
        print("--- routing ablation PPL ---")
        result["ppl"] = report_quality(
            model_dir,
            filler_ids,
            reranker,
            contexts,
            args.chunk_size,
            modes,
            routing=routing,
        )
        print(json.dumps(result["ppl"], indent=2))
    if not args.skip_speed:
        print("--- routing ablation speed ---")
        result["speed"] = report_speed(
            model_dir,
            filler_ids,
            reranker,
            contexts,
            args.chunk_size,
            args.new_tokens,
            modes,
            routing=routing,
        )
        print(json.dumps(result["speed"], indent=2))
    if not args.skip_needle:
        print("--- routing ablation needle ---")
        result["needle"] = report_needle(
            model_dir,
            tokenizer,
            filler_ids,
            reranker,
            contexts,
            args.chunk_size,
            modes,
            routing=routing,
        )
        print(json.dumps(result["needle"], indent=2))
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("saved:", args.output)


if __name__ == "__main__":
    main()
