"""Pythia-1B with Ocean routing and full INT4 KV-cache.

This module contains inference architecture only:

* manual Pythia-1B compatible decoder;
* RoPE and causal attention;
* Ocean local/global/semantic block routing;
* exact full INT4 KV-cache storage;
* official Hugging Face/safetensors weight loading.

Training, datasets, optimizers and training checkpoints intentionally do not
belong to this file.  Use ``Pythia_1B_INT4_routed_train_benchmark.ipynb`` or a
separate training script for those tasks.
"""

from __future__ import annotations

import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors


torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32


def clear_gpu_cache():
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def synchronize():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def unwrap_compiled_model(model):
    """Return the real module hidden inside torch.compile's _orig_mod wrapper."""
    while hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


@dataclass
class PythiaConfig:
    vocab_size: int = 50304
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 16
    num_attention_heads: int = 8
    max_position_embeddings: int = 2048
    rotary_pct: float = 0.25
    rotary_emb_base: float = 10000.0
    rope_scaling_factor: float = 1.0
    rope_native_context: int = 2048
    layer_norm_eps: float = 1e-5
    hidden_act: str = "gelu"
    use_parallel_residual: bool = True
    use_cache: bool = True
    attention_bias: bool = True

    @property
    def head_dim(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size должен делиться на num_attention_heads")
        return self.hidden_size // self.num_attention_heads

    @property
    def rotary_ndims(self):
        return int(self.head_dim * self.rotary_pct)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0, scaling_factor=1.0):
        super().__init__()
        if scaling_factor <= 0:
            raise ValueError("scaling_factor должен быть положительным")
        self.dim = dim
        self.base = base
        self.scaling_factor = float(scaling_factor)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids, dtype):
        scaled_positions = position_ids.float() / self.scaling_factor
        freqs = torch.outer(scaled_positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype)[None, None, :, :]
        sin = emb.sin().to(dtype=dtype)[None, None, :, :]
        return cos, sin


def apply_rotary(q, k, cos, sin, rotary_ndims):
    q_rot, q_pass = q[..., :rotary_ndims], q[..., rotary_ndims:]
    k_rot, k_pass = k[..., :rotary_ndims], k[..., rotary_ndims:]
    q_rot = q_rot * cos + rotate_half(q_rot) * sin
    k_rot = k_rot * cos + rotate_half(k_rot) * sin
    return torch.cat((q_rot, q_pass), dim=-1), torch.cat((k_rot, k_pass), dim=-1)


class PythiaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rotary_ndims = config.rotary_ndims
        self.query_key_value = nn.Linear(
            config.hidden_size,
            3 * config.hidden_size,
            bias=config.attention_bias,
        )
        self.dense = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.rotary_emb = RotaryEmbedding(
            config.rotary_ndims,
            config.rotary_emb_base,
            scaling_factor=getattr(config, "rope_scaling_factor", 1.0),
        )

    def forward(self, hidden_states, past_key_value=None, use_cache=False):
        batch_size, query_length, _ = hidden_states.shape
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(
            batch_size,
            query_length,
            self.num_attention_heads,
            3 * self.head_dim,
        ).transpose(1, 2)
        query, key, value = qkv.chunk(3, dim=-1)
        past_length = 0 if past_key_value is None else past_key_value[0].shape[2]
        positions = torch.arange(
            past_length,
            past_length + query_length,
            device=hidden_states.device,
            dtype=torch.long,
        )
        cos, sin = self.rotary_emb(positions, hidden_states.dtype)
        query, key = apply_rotary(query, key, cos, sin, self.rotary_ndims)
        if past_key_value is not None:
            key = torch.cat((past_key_value[0], key), dim=2)
            value = torch.cat((past_key_value[1], value), dim=2)
        key_length = key.shape[2]
        allowed = torch.ones(
            query_length,
            key_length,
            device=hidden_states.device,
            dtype=torch.bool,
        ).tril(diagonal=past_length)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed[None, None, :, :],
            dropout_p=0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).contiguous().view(batch_size, query_length, -1)
        present = (key, value) if use_cache else None
        return self.dense(output), present


class PythiaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        return self.dense_4h_to_h(F.gelu(self.dense_h_to_4h(hidden_states)))


class PythiaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.attention = PythiaAttention(config)
        self.mlp = PythiaMLP(config)
        self.use_parallel_residual = config.use_parallel_residual

    def forward(self, hidden_states, past_key_value=None, use_cache=False):
        residual = hidden_states
        attention_input = self.input_layernorm(hidden_states)
        attention_output, present = self.attention(
            attention_input,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        if self.use_parallel_residual:
            mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
            hidden_states = residual + attention_output + mlp_output
        else:
            hidden_states = residual + attention_output
            hidden_states = hidden_states + self.mlp(
                self.post_attention_layernorm(hidden_states)
            )
        return hidden_states, present


class PythiaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gpt_neox = nn.Module()
        self.gpt_neox.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        self.gpt_neox.layers = nn.ModuleList(
            [PythiaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gpt_neox.final_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        hidden_states = self.gpt_neox.embed_in(input_ids)
        if past_key_values is None:
            past_key_values = [None] * len(self.gpt_neox.layers)
        presents = []
        for layer, past in zip(self.gpt_neox.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                past_key_value=past,
                use_cache=use_cache,
            )
            if use_cache:
                presents.append(present)
        hidden_states = self.gpt_neox.final_layer_norm(hidden_states)
        return self.embed_out(hidden_states), tuple(presents) if use_cache else None


class INT4FullKVCache:
    """Exact full KV-cache: every token is stored, packed as two signed INT4 values."""

    def __init__(self, config, max_length, device=None):
        self.max_length = int(max_length)
        self.device = device or DEVICE
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.packed_dim = (self.head_dim + 1) // 2
        shape = (1, self.num_heads, self.max_length, self.packed_dim)
        self.key_packed = torch.empty(shape, device=self.device, dtype=torch.uint8)
        self.value_packed = torch.empty_like(self.key_packed)
        scale_shape = (1, self.num_heads, self.max_length)
        self.key_scale = torch.empty(scale_shape, device=self.device, dtype=torch.float16)
        self.value_scale = torch.empty_like(self.key_scale)

    @staticmethod
    def _quantize(x):
        scale = x.float().abs().amax(dim=-1).clamp_min(1e-6) / 7.0
        quant = torch.round(x.float() / scale.unsqueeze(-1)).clamp(-8, 7).to(torch.int16) + 8
        if quant.shape[-1] % 2:
            quant = F.pad(quant, (0, 1), value=8)
        packed = quant[..., 0::2].to(torch.uint8) | (quant[..., 1::2].to(torch.uint8) << 4)
        return packed, scale.to(torch.float16)

    @staticmethod
    def _dequantize(packed, scale, head_dim, dtype):
        low = packed & 15
        high = packed >> 4
        quant = torch.stack((low, high), dim=-1).reshape(
            *packed.shape[:-1],
            packed.shape[-1] * 2,
        )[..., :head_dim]
        return (quant.to(dtype) - 8.0) * scale.to(dtype).unsqueeze(-1)

    @torch.no_grad()
    def append_chunk(self, key, value, start_position):
        end = start_position + key.shape[2]
        key_packed, key_scale = self._quantize(key)
        value_packed, value_scale = self._quantize(value)
        self.key_packed[:, :, start_position:end, :].copy_(key_packed)
        self.value_packed[:, :, start_position:end, :].copy_(value_packed)
        self.key_scale[:, :, start_position:end].copy_(key_scale)
        self.value_scale[:, :, start_position:end].copy_(value_scale)

    def append(self, key, value, position):
        self.append_chunk(key, value, position)

    def get(self, length, dtype):
        key = self._dequantize(
            self.key_packed[:, :, :length, :],
            self.key_scale[:, :, :length],
            self.head_dim,
            dtype,
        )
        value = self._dequantize(
            self.value_packed[:, :, :length, :],
            self.value_scale[:, :, :length],
            self.head_dim,
            dtype,
        )
        return key, value


class INT4RoutedKVCache(INT4FullKVCache):
    """Full INT4 storage with selectable indexed block routing.

    ``hierarchical_cosine`` uses the persistent binary summary tree.  At every
    level it scores only the children of a fixed-width beam, so the number of
    scored tree nodes is proportional to ``beam_width * log2(num_blocks)``.
    ``full_scan_cosine`` is retained as the exact ablation baseline and scans
    every complete block summary.

    The cache still stores every K/V token.  The index changes lookup cost, not
    storage complexity.
    """

    def __init__(
        self,
        config,
        max_length,
        block_size=256,
        route_blocks=16,
        beam_width=32,
        summary_parts=4,
        global_blocks=1,
        local_blocks=2,
        local_window=256,
        route_refresh_interval=64,
        route_mode="hierarchical_cosine",
        device=None,
    ):
        super().__init__(config, max_length=max_length, device=device)
        if block_size % summary_parts != 0:
            raise ValueError("block_size должен делиться на summary_parts")
        self.block_size = int(block_size)
        self.route_blocks = int(route_blocks)
        self.beam_width = int(beam_width)
        self.summary_parts = int(summary_parts)
        self.global_blocks = int(global_blocks)
        self.local_blocks = int(local_blocks)
        self.local_window = int(local_window)
        self.route_refresh_interval = int(route_refresh_interval)
        self.route_mode = str(route_mode)
        if self.route_mode not in {"full_scan_cosine", "hierarchical_cosine"}:
            raise ValueError(
                "route_mode должен быть full_scan_cosine или hierarchical_cosine"
            )
        self.part_size = self.block_size // self.summary_parts
        max_blocks = max(1, math.ceil(self.max_length / self.block_size))
        self.tree_capacity = 1
        while self.tree_capacity < max_blocks:
            self.tree_capacity *= 2
        node_count = 2 * self.tree_capacity - 1
        self.leaf_start = self.tree_capacity - 1
        self.tree_sums = torch.zeros(
            self.num_heads,
            node_count,
            self.summary_parts,
            self.head_dim,
            device=self.device,
            dtype=torch.float16,
        )
        self.value_tree_sums = torch.zeros_like(self.tree_sums)
        self.tree_counts = torch.zeros(
            node_count,
            self.summary_parts,
            device=self.device,
            dtype=torch.int32,
        )
        self.current_sums = torch.zeros(
            self.num_heads,
            self.summary_parts,
            self.head_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self.current_value_sums = torch.zeros_like(self.current_sums)
        self.current_counts = torch.zeros(
            self.summary_parts,
            device=self.device,
            dtype=torch.int32,
        )
        self.current_count = 0
        self.current_block_id = 0
        self.offsets = torch.arange(self.block_size, device=self.device, dtype=torch.long)
        self._route_cache = None
        self._route_age = 0
        self._route_num_blocks = -1
        self.route_calls = 0
        self.route_nodes_scored = 0
        self.route_leaf_candidates = 0
        self.route_depth = int(math.log2(self.tree_capacity))

    @torch.no_grad()
    def _commit_current_block(self):
        if self.current_count == 0:
            return
        block_sums = self.current_sums.to(self.tree_sums.dtype)
        value_block_sums = self.current_value_sums.to(self.value_tree_sums.dtype)
        node = self.leaf_start + self.current_block_id
        while True:
            self.tree_sums[:, node, :, :].add_(block_sums)
            self.value_tree_sums[:, node, :, :].add_(value_block_sums)
            self.tree_counts[node, :].add_(self.current_counts)
            if node == 0:
                break
            node = (node - 1) // 2
        self.current_sums.zero_()
        self.current_value_sums.zero_()
        self.current_counts.zero_()
        self.current_count = 0

    @torch.no_grad()
    def append_chunk(self, key, value, start_position):
        super().append_chunk(key, value, start_position)
        offset = 0
        while offset < key.shape[2]:
            position = start_position + offset
            if position > 0 and position % self.block_size == 0:
                self._commit_current_block()
            if self.current_count == 0:
                self.current_block_id = position // self.block_size
            block_offset = position % self.block_size
            part = block_offset // self.part_size
            take = min(
                key.shape[2] - offset,
                self.block_size - block_offset,
                self.part_size - block_offset % self.part_size,
            )
            self.current_sums[:, part, :].add_(
                key[0, :, offset : offset + take, :].float().sum(dim=1)
            )
            self.current_value_sums[:, part, :].add_(
                value[0, :, offset : offset + take, :].float().sum(dim=1)
            )
            self.current_counts[part] += take
            self.current_count += take
            offset += take

    def reset_route_stats(self):
        """Reset routing counters without clearing the KV-cache."""
        self.route_calls = 0
        self.route_nodes_scored = 0
        self.route_leaf_candidates = 0

    def _cache_index(self, index):
        """Return a Long index on the same device as the cache buffers.

        Routing indices can be produced by notebook helpers or a CPU-side
        selector.  CUDA gather/index_select requires the index tensor itself
        to be on the CUDA device, so normalize it at this single boundary.
        """
        return index.to(device=self.tree_sums.device, dtype=torch.long)

    def _node_scores(self, query_vector, node_ids):
        node_ids = self._cache_index(node_ids)
        gather_ids = node_ids[:, :, None, None].expand(
            node_ids.shape[0],
            node_ids.shape[1],
            self.summary_parts,
            self.head_dim,
        )
        sums = self.tree_sums.gather(1, gather_ids)
        counts = self.tree_counts[node_ids]
        summaries = sums / counts.clamp_min(1).to(sums.dtype).unsqueeze(-1)
        query_norm = F.normalize(query_vector.float(), dim=-1)[:, None, None, :]
        summary_norm = F.normalize(summaries.float(), dim=-1)
        return (
            query_norm * summary_norm
        ).sum(dim=-1).masked_fill(counts <= 0, float("-inf")).max(dim=-1).values

    def _leaf_scores(self, query_vector, block_ids):
        """Cosine score complete block summaries for each attention head."""
        leaf_ids = self.leaf_start + block_ids
        return self._node_scores(query_vector, leaf_ids)

    def _aligned_local_start(self, length):
        return max(
            0,
            ((length - self.local_window) // self.block_size) * self.block_size,
        )

    @torch.no_grad()
    def _hierarchical_candidates(self, query_vector, routeable_blocks, count):
        """Search the summary tree with fixed beam width.

        For fixed ``beam_width`` and head dimension, each level evaluates a
        constant number of children.  The number of levels is the tree depth,
        ``ceil(log2(max_blocks))``.
        """
        beam = min(max(self.beam_width, count), routeable_blocks)
        candidates = torch.zeros(
            self.num_heads,
            1,
            device=self.device,
            dtype=torch.long,
        )
        for _ in range(self.route_depth):
            left = candidates * 2 + 1
            right = left + 1
            children = torch.cat((left, right), dim=-1)
            self.route_nodes_scored += int(children.numel())
            keep = min(beam, children.shape[-1])
            candidates = children.gather(
                -1,
                torch.topk(
                    self._node_scores(query_vector, children),
                    k=keep,
                    dim=-1,
                ).indices,
            )
        block_ids = (candidates - self.leaf_start).clamp(0, routeable_blocks - 1)
        self.route_leaf_candidates += int(block_ids.numel())
        scores = self._leaf_scores(query_vector, block_ids)
        self.route_nodes_scored += int(block_ids.numel())
        return block_ids, scores

    @torch.no_grad()
    def _full_scan_candidates(self, query_vector, routeable_blocks):
        all_blocks = torch.arange(
            routeable_blocks,
            device=self.device,
            dtype=torch.long,
        ).view(1, -1).expand(self.num_heads, -1)
        self.route_nodes_scored += int(all_blocks.numel())
        self.route_leaf_candidates += int(all_blocks.numel())
        return all_blocks, self._leaf_scores(query_vector, all_blocks)

    @torch.no_grad()
    def _select_route(self, query, length):
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
            self.route_calls += 1
            return mandatory.view(1, 1, -1).expand(1, self.num_heads, -1)

        query_vector = query[:, :, 0, :].reshape(self.num_heads, self.head_dim)
        if self.route_mode == "full_scan_cosine":
            candidate_ids, scores = self._full_scan_candidates(
                query_vector,
                routeable_blocks,
            )
        else:
            candidate_ids, scores = self._hierarchical_candidates(
                query_vector,
                routeable_blocks,
                semantic_count,
            )

        blocked = (candidate_ids[:, :, None] == mandatory.view(1, 1, -1)).any(dim=-1)
        scores = scores.masked_fill(blocked, float("-inf"))
        keep = min(semantic_count, candidate_ids.shape[-1])
        selected = candidate_ids.gather(
            1,
            torch.topk(scores, k=keep, dim=-1).indices,
        )
        self.route_calls += 1
        return torch.cat(
            [
                mandatory.view(1, 1, -1).expand(1, self.num_heads, -1),
                selected.unsqueeze(0),
            ],
            dim=-1,
        )

    def _gather_int4(self, indices, dtype):
        indices = self._cache_index(indices)
        packed_ids = indices.unsqueeze(-1).expand(-1, -1, -1, self.packed_dim)
        packed_key = self.key_packed.gather(2, packed_ids)
        packed_value = self.value_packed.gather(2, packed_ids)
        key_scale = self.key_scale.gather(2, indices)
        value_scale = self.value_scale.gather(2, indices)
        return (
            self._dequantize(packed_key, key_scale, self.head_dim, dtype),
            self._dequantize(packed_value, value_scale, self.head_dim, dtype),
        )

    def attention_kv(self, query, position):
        length = position + 1
        if length <= 0:
            empty = torch.empty(
                1,
                self.num_heads,
                0,
                self.head_dim,
                device=self.device,
                dtype=query.dtype,
            )
            return empty, empty
        local_start = self._aligned_local_start(length)
        local_ids = torch.arange(
            local_start,
            length,
            device=self.device,
            dtype=torch.long,
        ).view(1, 1, -1).expand(1, self.num_heads, -1)
        local_key, local_value = self._gather_int4(local_ids, query.dtype)
        num_blocks = length // self.block_size
        if (
            self._route_cache is None
            or self._route_age >= self.route_refresh_interval
            or self._route_num_blocks != num_blocks
        ):
            self._route_cache = self._select_route(query, length)
            self._route_age = 0
            self._route_num_blocks = num_blocks
        else:
            self._route_age += 1
        if self._route_cache.shape[-1] == 0:
            return local_key, local_value
        route_ids = (
            self._cache_index(self._route_cache)[:, :, :, None]
            * self.block_size
            + self.offsets.view(1, 1, 1, -1)
        ).reshape(1, self.num_heads, -1).clamp_max(length - 1)
        route_key, route_value = self._gather_int4(route_ids, query.dtype)
        return (
            torch.cat((local_key, route_key), dim=2),
            torch.cat((local_value, route_value), dim=2),
        )


class OceanAttention(PythiaAttention):
    def __init__(self, config, **routing):
        super().__init__(config)
        self.routing = dict(routing)

    @torch.inference_mode()
    def forward_token(self, hidden_states, cache, position):
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(1, 1, self.num_attention_heads, 3 * self.head_dim).transpose(1, 2)
        query, key, value = qkv.chunk(3, dim=-1)
        position_ids = torch.tensor([position], device=hidden_states.device, dtype=torch.long)
        cos, sin = self.rotary_emb(position_ids, hidden_states.dtype)
        query, key = apply_rotary(query, key, cos, sin, self.rotary_ndims)
        cache.append(key, value, position)
        route_key, route_value = cache.attention_kv(query, position)
        output = F.scaled_dot_product_attention(
            query,
            route_key,
            route_value,
            dropout_p=0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).contiguous().view(1, 1, -1)
        return self.dense(output)

    @torch.inference_mode()
    def forward_chunk(self, hidden_states, cache, start_position):
        q_len = hidden_states.shape[1]
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(1, q_len, self.num_attention_heads, 3 * self.head_dim).transpose(1, 2)
        query, key, value = qkv.chunk(3, dim=-1)
        positions = torch.arange(
            start_position,
            start_position + q_len,
            device=hidden_states.device,
            dtype=torch.long,
        )
        cos, sin = self.rotary_emb(positions, hidden_states.dtype)
        query, key = apply_rotary(query, key, cos, sin, self.rotary_ndims)
        past_key, past_value = cache.attention_kv(query[:, :, -1:, :], start_position - 1)
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
            torch.ones(q_len, q_len, device=hidden_states.device, dtype=torch.bool)
        )
        output = F.scaled_dot_product_attention(
            query,
            route_key,
            route_value,
            attn_mask=allowed[None, None, :, :],
            dropout_p=0.0,
            is_causal=False,
        )
        cache.append_chunk(key, value, start_position)
        output = output.transpose(1, 2).contiguous().view(1, q_len, -1)
        return self.dense(output)


ROUTING_CONFIG = {
    # Production baseline.  Use HIERARCHICAL_ROUTING_CONFIG below to activate
    # the indexed O(log N) route search.
    "route_mode": "full_scan_cosine",
    "block_size": 256,
    "route_blocks": 16,
    "beam_width": 32,
    "summary_parts": 4,
    "global_blocks": 1,
    "local_blocks": 2,
    "local_window": 256,
    "route_refresh_interval": 64,
}

HIERARCHICAL_ROUTING_CONFIG = {
    **ROUTING_CONFIG,
    "route_mode": "hierarchical_cosine",
}


class OceanINT4PythiaForCausalLM(PythiaForCausalLM):
    def __init__(self, config, routing=None):
        super().__init__(config)
        self.routing = dict(routing or ROUTING_CONFIG)
        for layer in self.gpt_neox.layers:
            layer.attention = OceanAttention(config, **self.routing)

    def new_bounded_cache(self, max_length):
        return [
            INT4RoutedKVCache(
                self.config,
                max_length=max_length,
                device=DEVICE,
                **self.routing,
            )
            for _ in self.gpt_neox.layers
        ]

    @torch.inference_mode()
    def forward_bounded_chunk(self, input_ids, caches, start_position):
        input_ids = input_ids.to(
            device=self.gpt_neox.embed_in.weight.device,
            dtype=torch.long,
            non_blocking=True,
        )
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
        input_ids = input_ids.to(
            device=self.gpt_neox.embed_in.weight.device,
            dtype=torch.long,
            non_blocking=True,
        )
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


def load_weight_file(path):
    if path.suffix == ".safetensors":
        return load_safetensors(str(path), device="cpu")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_official_weights(model, model_dir):
    model = unwrap_compiled_model(model)
    root = Path(model_dir)
    index = next(
        (
            path
            for path in (
                root / "model.safetensors.index.json",
                root / "pytorch_model.bin.index.json",
            )
            if path.exists()
        ),
        None,
    )
    if index is not None:
        info = json.loads(index.read_text())
        state = {}
        for name in sorted(set(info["weight_map"].values())):
            state.update(load_weight_file(root / name))
    else:
        path = next(
            (
                candidate
                for candidate in (
                    root / "model.safetensors",
                    root / "pytorch_model.bin",
                )
                if candidate.exists()
            ),
            None,
        )
        if path is None:
            raise FileNotFoundError(f"Не найден файл весов в {root}")
        state = load_weight_file(path)
    expected = set(model.state_dict())
    filtered = {key: value for key, value in state.items() if key in expected}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing or unexpected:
        raise RuntimeError({"missing": missing[:20], "unexpected": unexpected[:20]})
    print("ignored auxiliary checkpoint keys:", len(set(state) - expected))
    return model


def load_checkpoint(model, checkpoint_path):
    model = unwrap_compiled_model(model)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state, strict=True)
    del state
    return model


def build_ocean_model(
    config=None,
    routing=None,
    device=DEVICE,
    dtype=DTYPE,
):
    """Construct the inference-only Ocean Pythia model."""
    model = OceanINT4PythiaForCausalLM(
        config or PythiaConfig(),
        routing=routing or ROUTING_CONFIG,
    )
    return model.to(device=device, dtype=dtype).eval()


def load_ocean_model(
    model_dir=None,
    checkpoint_path=None,
    config=None,
    routing=None,
    device=DEVICE,
    dtype=DTYPE,
):
    """Build Ocean and load official or previously saved model weights.

    ``checkpoint_path`` takes precedence over ``model_dir``.  The checkpoint
    must contain only model parameters; optimizer/training state is not used.
    """
    model = build_ocean_model(
        config=config,
        routing=routing,
        device=device,
        dtype=dtype,
    )
    if checkpoint_path is not None:
        return load_checkpoint(model, checkpoint_path)
    if model_dir is None:
        raise ValueError("Укажите model_dir или checkpoint_path")
    return load_official_weights(model, model_dir)
