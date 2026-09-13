#!/usr/bin/env python3
"""Final speed benchmark for the production Ocean INT4 cosine baseline.

This benchmark intentionally excludes neural reranking and hierarchical routing.
It measures the current production candidate:

    full INT4 KV-cache + local window + full-scan cosine routing

The 1M case is guarded because the exact full INT4 cache alone is about 31 GiB
for Pythia-1B, before model weights, temporary tensors and allocator overhead.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

try:
    from .model import PythiaConfig, ROUTING_CONFIG, clear_gpu_cache
    from .routing_ablation_benchmark import (
        build_model,
        repeat_ids,
        routed_speed,
    )
except ImportError:
    from model import PythiaConfig, ROUTING_CONFIG, clear_gpu_cache
    from routing_ablation_benchmark import (
        build_model,
        repeat_ids,
        routed_speed,
    )


DEFAULT_MODEL_ID = "EleutherAI/pythia-1b"
DEFAULT_MODEL_DIR = (
    "/home/froschin/.cache/huggingface/hub/models--EleutherAI--pythia-1b/"
    "snapshots/f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2"
)
DEFAULT_CONTEXTS = (2_048, 14_000, 32_000, 100_000, 1_000_000)


def estimated_full_int4_kv_gib(context_length, config=None):
    """Estimate exact full INT4 K/V storage, excluding all other tensors."""
    config = config or PythiaConfig()
    packed_dim = (config.head_dim + 1) // 2
    packed_bytes = 2 * config.num_attention_heads * context_length * packed_dim
    scale_bytes = (
        2
        * config.num_attention_heads
        * context_length
        * torch.tensor([], dtype=torch.float16).element_size()
    )
    per_layer = packed_bytes + scale_bytes
    total = config.num_hidden_layers * per_layer
    return total / 2**30


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--text-file", default="./tinyshakespeare.txt")
    parser.add_argument(
        "--contexts",
        default=",".join(str(item) for item in DEFAULT_CONTEXTS),
    )
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--max-cache-gib", type=float, default=28.0)
    parser.add_argument(
        "--output",
        default="./final_long_context_cosine_int4.json",
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
    config = PythiaConfig()
    routing = dict(ROUTING_CONFIG)
    model = build_model(
        model_dir,
        routing_mode="full_scan_cosine",
        candidate_blocks=64,
        reranker=None,
        routing=routing,
    )
    results = {
        "benchmark": "final_long_context_cosine_int4",
        "model": "Pythia-1B Ocean production baseline",
        "cache": "full_exact_int4_kv_routed",
        "routing": "full_scan_cosine",
        "config": routing,
        "chunk_size": args.chunk_size,
        "new_tokens": args.new_tokens,
        "rows": [],
    }
    for context_length in contexts:
        estimated_cache = estimated_full_int4_kv_gib(context_length, config)
        print(
            f"context={context_length:,}; "
            f"estimated_full_int4_kv={estimated_cache:.3f} GiB"
        )
        if estimated_cache > args.max_cache_gib:
            row = {
                "prompt_length": context_length,
                "status": "skipped_memory_guard",
                "estimated_full_int4_kv_gib": estimated_cache,
                "max_cache_gib": args.max_cache_gib,
                "cache": "full_exact_int4_kv_routed",
                "routing": "full_scan_cosine",
            }
            results["rows"].append(row)
            print(row)
            continue
        ids = repeat_ids(filler_ids, context_length)
        try:
            row = routed_speed(
                model,
                ids,
                context_length=context_length,
                chunk_size=args.chunk_size,
                new_tokens=args.new_tokens,
            )
            row["status"] = "ok"
            row["estimated_full_int4_kv_gib"] = estimated_cache
            row["cache"] = "full_exact_int4_kv_routed"
            results["rows"].append(row)
            print(row)
        finally:
            del ids
            gc.collect()
            clear_gpu_cache()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("saved:", output)


if __name__ == "__main__":
    main()
