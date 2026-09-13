#!/usr/bin/env python3
"""Benchmark indexed hierarchical routing against the exact full scan.

The hierarchical path uses the persistent binary summary tree in
``backend.model.INT4RoutedKVCache``.  Its route search scores a fixed-width
beam at each tree level, so with fixed ``beam_width`` and head dimension the
route lookup is O(log(number_of_blocks)).  The benchmark reports the actual
number of scored tree nodes; it does not infer asymptotics from wall time.

This is a speed/index benchmark.  It does not claim long-context quality.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

try:
    from .model import (
        DEVICE,
        DTYPE,
        HIERARCHICAL_ROUTING_CONFIG,
        PythiaConfig,
        ROUTING_CONFIG,
        clear_gpu_cache,
        load_ocean_model,
        synchronize,
    )
except ImportError:
    from model import (
        DEVICE,
        DTYPE,
        HIERARCHICAL_ROUTING_CONFIG,
        PythiaConfig,
        ROUTING_CONFIG,
        clear_gpu_cache,
        load_ocean_model,
        synchronize,
    )


DEFAULT_MODEL_ID = "EleutherAI/pythia-1b"
DEFAULT_MODEL_DIR = (
    "/home/froschin/.cache/huggingface/hub/models--EleutherAI--pythia-1b/"
    "snapshots/f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2"
)
DEFAULT_CONTEXTS = (2_048, 14_000, 32_000, 100_000)


def estimated_full_int4_kv_gib(context_length, config=None):
    """Estimate full exact INT4 K/V storage, excluding weights and temporaries."""
    config = config or PythiaConfig()
    packed_dim = (config.head_dim + 1) // 2
    packed_bytes = (
        2
        * config.num_attention_heads
        * context_length
        * packed_dim
    )
    scale_bytes = (
        2
        * config.num_attention_heads
        * context_length
        * torch.tensor([], dtype=torch.float16).element_size()
    )
    return config.num_hidden_layers * (packed_bytes + scale_bytes) / 2**30


def repeat_ids(source_ids, length):
    if source_ids.numel() == 0:
        raise ValueError("Источник токенов пуст")
    repeats = math.ceil(length / source_ids.numel())
    return source_ids.repeat(repeats)[:length].contiguous()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--text-file", default="./tinyshakespeare.txt")
    parser.add_argument(
        "--contexts",
        default=",".join(str(item) for item in DEFAULT_CONTEXTS),
    )
    parser.add_argument(
        "--routing",
        choices=("full_scan_cosine", "hierarchical_cosine", "both"),
        default="both",
    )
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--max-cache-gib", type=float, default=28.0)
    parser.add_argument(
        "--output",
        default="./hierarchical_routing_results.json",
    )
    return parser.parse_args()


def resolve_model_dir(args):
    model_dir = Path(args.model_dir)
    if model_dir.exists():
        return model_dir
    return Path(
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


def _reset_route_stats(caches):
    for cache in caches:
        cache.reset_route_stats()


def _route_stats(caches):
    route_calls = sum(cache.route_calls for cache in caches)
    nodes_scored = sum(cache.route_nodes_scored for cache in caches)
    leaf_candidates = sum(cache.route_leaf_candidates for cache in caches)
    depths = [cache.route_depth for cache in caches]
    return {
        "route_calls_all_layers": route_calls,
        "route_nodes_scored_all_layers": nodes_scored,
        "route_leaf_candidates_all_layers": leaf_candidates,
        "mean_nodes_scored_per_route": (
            nodes_scored / route_calls if route_calls else 0.0
        ),
        "tree_depth": max(depths) if depths else 0,
    }


@torch.inference_mode()
def benchmark_model(model, input_ids, chunk_size, new_tokens):
    model_device = model.gpt_neox.embed_in.weight.device
    input_ids = input_ids.to(
        device=model_device,
        dtype=torch.long,
        non_blocking=True,
    )
    context_length = int(input_ids.numel())
    caches = model.new_bounded_cache(context_length + new_tokens)
    _reset_route_stats(caches)
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    synchronize()
    prefill_start = time.perf_counter()
    logits = None
    for start in range(0, context_length, chunk_size):
        end = min(start + chunk_size, context_length)
        logits = model.forward_bounded_chunk(
            input_ids[start:end],
            caches,
            start,
        )
    synchronize()
    prefill_seconds = time.perf_counter() - prefill_start

    next_token = logits[:, -1].argmax(dim=-1)
    synchronize()
    decode_start = time.perf_counter()
    for position in range(context_length, context_length + new_tokens):
        logits = model.forward_bounded_token(next_token, caches, position)
        next_token = logits[:, -1].argmax(dim=-1)
    synchronize()
    decode_seconds = time.perf_counter() - decode_start

    row = {
        "prompt_length": context_length,
        "chunk_size": chunk_size,
        "new_tokens": new_tokens,
        "prefill_seconds": prefill_seconds,
        "prefill_tokens_per_second": context_length / max(prefill_seconds, 1e-9),
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": new_tokens / max(decode_seconds, 1e-9),
        "total_seconds": prefill_seconds + decode_seconds,
        "peak_cuda_allocated_gib": (
            torch.cuda.max_memory_allocated() / 2**30
            if DEVICE.type == "cuda"
            else None
        ),
        "routing_stats": _route_stats(caches),
    }
    del caches, logits, next_token
    return row


def run_mode(model_dir, filler_ids, contexts, args, routing_mode):
    routing = dict(
        ROUTING_CONFIG
        if routing_mode == "full_scan_cosine"
        else HIERARCHICAL_ROUTING_CONFIG
    )
    model = load_ocean_model(
        model_dir=model_dir,
        routing=routing,
        device=DEVICE,
        dtype=DTYPE,
    )
    rows = []
    try:
        for context_length in contexts:
            estimate = estimated_full_int4_kv_gib(context_length, model.config)
            print(
                f"[{routing_mode}] context={context_length:,}; "
                f"estimated_full_int4_kv={estimate:.3f} GiB"
            )
            if estimate > args.max_cache_gib:
                row = {
                    "routing": routing_mode,
                    "prompt_length": context_length,
                    "status": "skipped_memory_guard",
                    "estimated_full_int4_kv_gib": estimate,
                    "max_cache_gib": args.max_cache_gib,
                }
                rows.append(row)
                print(row)
                continue
            ids = repeat_ids(filler_ids, context_length)
            try:
                row = benchmark_model(
                    model,
                    ids,
                    chunk_size=args.chunk_size,
                    new_tokens=args.new_tokens,
                )
                row["routing"] = routing_mode
                row["status"] = "ok"
                row["estimated_full_int4_kv_gib"] = estimate
                rows.append(row)
                print(row)
            finally:
                del ids
                clear_gpu_cache()
    finally:
        del model
        gc.collect()
        clear_gpu_cache()
    return rows


def main():
    args = parse_args()
    model_dir = resolve_model_dir(args)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    text = Path(args.text_file).read_text(encoding="utf-8")
    filler_ids = torch.tensor(
        tokenizer(text, add_special_tokens=False).input_ids,
        dtype=torch.long,
    )
    contexts = tuple(int(item) for item in args.contexts.split(","))
    modes = (
        ("full_scan_cosine", "hierarchical_cosine")
        if args.routing == "both"
        else (args.routing,)
    )
    results = {
        "benchmark": "indexed_hierarchical_routing",
        "cache": "full_exact_int4_kv",
        "contexts": contexts,
        "chunk_size": args.chunk_size,
        "new_tokens": args.new_tokens,
        "routing_configs": {
            "full_scan_cosine": dict(ROUTING_CONFIG),
            "hierarchical_cosine": dict(HIERARCHICAL_ROUTING_CONFIG),
        },
        "rows": [],
    }
    for mode in modes:
        results["rows"].extend(
            run_mode(model_dir, filler_ids, contexts, args, mode)
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("saved:", output)


if __name__ == "__main__":
    main()
