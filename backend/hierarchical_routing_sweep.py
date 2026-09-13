#!/usr/bin/env python3
"""Sweep the practical constants of indexed hierarchical routing.

The asymptotic route is fixed; this script measures the constants that decide
whether it is useful in practice:

* ``beam_width`` controls how many branches survive at every tree level;
* ``route_refresh_interval`` controls how often a route is recomputed;
* 64 generated tokens make refresh cost visible during decode.

The model weights are loaded only once.  Each row creates a fresh full INT4
cache, so rows are independent and directly comparable.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

try:
    from .hierarchical_routing_benchmark import (
        benchmark_model,
        estimated_full_int4_kv_gib,
        repeat_ids,
    )
    from .model import (
        DEVICE,
        DTYPE,
        HIERARCHICAL_ROUTING_CONFIG,
        clear_gpu_cache,
        load_ocean_model,
    )
except ImportError:
    from hierarchical_routing_benchmark import (
        benchmark_model,
        estimated_full_int4_kv_gib,
        repeat_ids,
    )
    from model import (
        DEVICE,
        DTYPE,
        HIERARCHICAL_ROUTING_CONFIG,
        clear_gpu_cache,
        load_ocean_model,
    )


DEFAULT_MODEL_ID = "EleutherAI/pythia-1b"
DEFAULT_MODEL_DIR = (
    "/home/froschin/.cache/huggingface/hub/models--EleutherAI--pythia-1b/"
    "snapshots/f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2"
)


def parse_int_list(value):
    values = tuple(int(item.strip()) for item in value.split(","))
    if not values or any(item <= 0 for item in values):
        raise ValueError("Список должен содержать положительные целые числа")
    return values


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--text-file", default="./tinyshakespeare.txt")
    parser.add_argument("--contexts", default="32768,100000")
    parser.add_argument("--beam-widths", default="8,16,32")
    parser.add_argument("--refresh-intervals", default="1,4,64")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--max-cache-gib", type=float, default=28.0)
    parser.add_argument(
        "--output",
        default="./hierarchical_routing_sweep.json",
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
    contexts = parse_int_list(args.contexts)
    beam_widths = parse_int_list(args.beam_widths)
    refresh_intervals = parse_int_list(args.refresh_intervals)
    model_dir = resolve_model_dir(args)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    text = Path(args.text_file).read_text(encoding="utf-8")
    filler_ids = torch.tensor(
        tokenizer(text, add_special_tokens=False).input_ids,
        dtype=torch.long,
    )

    model = load_ocean_model(
        model_dir=model_dir,
        routing=dict(HIERARCHICAL_ROUTING_CONFIG),
        device=DEVICE,
        dtype=DTYPE,
    )
    rows = []
    try:
        for context_length in contexts:
            estimate = estimated_full_int4_kv_gib(context_length, model.config)
            if estimate > args.max_cache_gib:
                row = {
                    "routing": "hierarchical_cosine",
                    "prompt_length": context_length,
                    "status": "skipped_memory_guard",
                    "estimated_full_int4_kv_gib": estimate,
                    "max_cache_gib": args.max_cache_gib,
                }
                rows.append(row)
                print(row)
                continue

            ids = repeat_ids(filler_ids, context_length)
            for beam_width in beam_widths:
                for refresh_interval in refresh_intervals:
                    model.routing["beam_width"] = beam_width
                    model.routing["route_refresh_interval"] = refresh_interval
                    row = benchmark_model(
                        model,
                        ids,
                        chunk_size=args.chunk_size,
                        new_tokens=args.new_tokens,
                    )
                    row.update(
                        {
                            "routing": "hierarchical_cosine",
                            "beam_width": beam_width,
                            "route_refresh_interval": refresh_interval,
                            "status": "ok",
                            "estimated_full_int4_kv_gib": estimate,
                        }
                    )
                    rows.append(row)
                    print(row)
                    clear_gpu_cache()
            del ids
            gc.collect()
            clear_gpu_cache()
    finally:
        del model
        gc.collect()
        clear_gpu_cache()

    result = {
        "benchmark": "hierarchical_routing_constant_sweep",
        "cache": "full_exact_int4_kv",
        "routing": "hierarchical_cosine",
        "contexts": contexts,
        "beam_widths": beam_widths,
        "refresh_intervals": refresh_intervals,
        "chunk_size": args.chunk_size,
        "new_tokens": args.new_tokens,
        "base_config": dict(HIERARCHICAL_ROUTING_CONFIG),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("saved:", output)


if __name__ == "__main__":
    main()
