#!/usr/bin/env python3
"""Native-context sweep over smaller hierarchical-routing block sizes.

The benchmark keeps the model weights and prompt fixed, measures dense PPL
once, then evaluates hierarchical cosine routing for several block sizes.
The default chunked protocol keeps routing active while avoiding the very slow
token-by-token quality path.

./.venv/bin/python backend/block_size_sweep_2048.py \
  --model-dir /home/froschin/.cache/huggingface/hub/models--EleutherAI--pythia-1b/snapshots/f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2 \
  --context-length 2048 \
  --protocol chunked \
  --chunk-size 256 \
  --block-sizes 16,32,64,128,256 \
  --beam-width 16 \
  --route-refresh-interval 64 \
  --output block_size_sweep_2048.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer

try:
    from .hierarchical_routing_ppl import (
        evaluate_dense,
        evaluate_routed_chunked,
        evaluate_routed_tokenwise,
        repeat_ids,
        resolve_model_dir,
    )
    from .model import (
        DEVICE,
        DTYPE,
        HIERARCHICAL_ROUTING_CONFIG,
        PythiaConfig,
        PythiaForCausalLM,
        clear_gpu_cache,
        load_official_weights,
        load_ocean_model,
    )
except ImportError:
    from hierarchical_routing_ppl import (
        evaluate_dense,
        evaluate_routed_chunked,
        evaluate_routed_tokenwise,
        repeat_ids,
        resolve_model_dir,
    )
    from model import (
        DEVICE,
        DTYPE,
        HIERARCHICAL_ROUTING_CONFIG,
        PythiaConfig,
        PythiaForCausalLM,
        clear_gpu_cache,
        load_official_weights,
        load_ocean_model,
    )


DEFAULT_MODEL_ID = "EleutherAI/pythia-1b"
DEFAULT_MODEL_DIR = (
    "/home/froschin/.cache/huggingface/hub/models--EleutherAI--pythia-1b/"
    "snapshots/f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--text-file", default="./tinyshakespeare.txt")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--protocol", choices=("chunked", "tokenwise"), default="chunked")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--block-sizes", default="16,32,64,128,256")
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--route-blocks", type=int, default=16)
    parser.add_argument("--summary-parts", type=int, default=4)
    parser.add_argument("--local-window", type=int, default=256)
    parser.add_argument("--route-refresh-interval", type=int, default=64)
    parser.add_argument(
        "--output",
        default="./block_size_sweep_2048.json",
    )
    return parser.parse_args()


def parse_positive_ints(value, name):
    values = tuple(int(item.strip()) for item in value.split(","))
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"{name} должны содержать положительные числа")
    return values


def make_config(args, block_size):
    config = dict(HIERARCHICAL_ROUTING_CONFIG)
    config.update(
        {
            "route_mode": "hierarchical_cosine",
            "block_size": block_size,
            "route_blocks": args.route_blocks,
            "beam_width": args.beam_width,
            "summary_parts": args.summary_parts,
            "local_window": args.local_window,
            "route_refresh_interval": args.route_refresh_interval,
        }
    )
    return config


def main():
    args = parse_args()
    if args.context_length != 2048:
        raise ValueError(
            "Этот benchmark предназначен для native context 2048; "
            "изменение context-length сделает имя/интерпретацию sweep неверными"
        )
    if args.chunk_size >= args.context_length and args.protocol == "chunked":
        raise ValueError(
            "chunk-size должен быть меньше context-length, иначе routing не активируется"
        )
    block_sizes = parse_positive_ints(args.block_sizes, "block-sizes")
    if any(item % args.summary_parts for item in block_sizes):
        raise ValueError("каждый block-size должен делиться на summary-parts")

    model_dir = resolve_model_dir(args)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    text = Path(args.text_file).read_text(encoding="utf-8")
    source_ids = torch.tensor(
        tokenizer(text, add_special_tokens=False).input_ids,
        dtype=torch.long,
    )
    ids = repeat_ids(source_ids, args.context_length)

    results = {
        "benchmark": "native_context_hierarchical_block_size_sweep",
        "context_length": args.context_length,
        "protocol": args.protocol,
        "chunk_size": args.chunk_size,
        "block_sizes": block_sizes,
        "routing": {
            "route_mode": "hierarchical_cosine",
            "route_blocks": args.route_blocks,
            "beam_width": args.beam_width,
            "summary_parts": args.summary_parts,
            "local_window": args.local_window,
            "route_refresh_interval": args.route_refresh_interval,
        },
        "rows": [],
    }

    dense_model = PythiaForCausalLM(PythiaConfig()).to(
        device=DEVICE,
        dtype=DTYPE,
    ).eval()
    try:
        load_official_weights(dense_model, model_dir)
        dense_row = evaluate_dense(dense_model, ids)
    finally:
        del dense_model
        gc.collect()
        clear_gpu_cache()
    results["dense"] = dense_row
    results["rows"].append(dense_row)
    print("--- dense baseline ---")
    print(dense_row)

    for block_size in block_sizes:
        config = make_config(args, block_size)
        print(f"--- hierarchical cosine, block_size={block_size} ---")
        model = load_ocean_model(
            model_dir=model_dir,
            routing=config,
            device=DEVICE,
            dtype=DTYPE,
        )
        try:
            if args.protocol == "chunked":
                row = evaluate_routed_chunked(
                    model,
                    ids,
                    "hierarchical_cosine",
                    config,
                    args.chunk_size,
                )
            else:
                row = evaluate_routed_tokenwise(
                    model,
                    ids,
                    "hierarchical_cosine",
                    config,
                )
            row["block_size"] = block_size
            row["ppl_delta_vs_dense"] = (
                row["perplexity"] - dense_row["perplexity"]
            )
            row["ppl_relative_percent_vs_dense"] = 100.0 * (
                row["perplexity"] / dense_row["perplexity"] - 1.0
            )
            results["rows"].append(row)
            print(row)
        finally:
            del model
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
