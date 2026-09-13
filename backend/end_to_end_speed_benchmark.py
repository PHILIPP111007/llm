#!/usr/bin/env python3
"""End-to-end speed comparison: dense Pythia versus hierarchical INT4.

Both models receive the same prompt and generate the same number of tokens by
greedy argmax.  Dense uses the ordinary FP16/FP32 tuple KV-cache; hierarchical
uses the full packed INT4 cache and indexed cosine routing.

Dense long-context work is guarded because its prefill and KV concatenation
become prohibitively expensive.  A skipped dense row is an explicit result,
not evidence that the dense model was measured at that length.
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
    from .hierarchical_routing_benchmark import (
        benchmark_model as benchmark_routed,
        estimated_full_int4_kv_gib,
        repeat_ids,
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
        synchronize,
    )
except ImportError:
    from hierarchical_routing_benchmark import (
        benchmark_model as benchmark_routed,
        estimated_full_int4_kv_gib,
        repeat_ids,
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
        synchronize,
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
    parser.add_argument("--contexts", default="2048,14000,32000")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--dense-max-context", type=int, default=14000)
    parser.add_argument("--max-cache-gib", type=float, default=28.0)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument(
        "--output",
        default="./end_to_end_speed_results.json",
    )
    return parser.parse_args()


def parse_int_list(value):
    values = tuple(int(item.strip()) for item in value.split(","))
    if not values or any(item <= 0 for item in values):
        raise ValueError("contexts должны содержать положительные числа")
    return values


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


@torch.inference_mode()
def benchmark_dense(model, input_ids, chunk_size, new_tokens):
    device = model.gpt_neox.embed_in.weight.device
    ids = input_ids.to(device=device, dtype=torch.long, non_blocking=True)
    past = None
    logits = None
    synchronize()
    prefill_start = time.perf_counter()
    for start in range(0, ids.numel(), chunk_size):
        end = min(start + chunk_size, ids.numel())
        logits, past = model(
            ids[start:end].view(1, -1),
            past_key_values=past,
            use_cache=True,
        )
    synchronize()
    prefill_seconds = time.perf_counter() - prefill_start

    next_token = logits[:, -1].argmax(dim=-1)
    synchronize()
    decode_start = time.perf_counter()
    for _ in range(new_tokens):
        logits, past = model(
            next_token.view(1, 1),
            past_key_values=past,
            use_cache=True,
        )
        next_token = logits[:, -1].argmax(dim=-1)
    synchronize()
    decode_seconds = time.perf_counter() - decode_start
    return {
        "model": "dense",
        "prompt_length": int(ids.numel()),
        "chunk_size": chunk_size,
        "new_tokens": new_tokens,
        "prefill_seconds": prefill_seconds,
        "prefill_tokens_per_second": ids.numel() / max(prefill_seconds, 1e-9),
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": new_tokens / max(decode_seconds, 1e-9),
        "total_seconds": prefill_seconds + decode_seconds,
        "status": "ok",
    }


def main():
    args = parse_args()
    contexts = parse_int_list(args.contexts)
    model_dir = resolve_model_dir(args)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    text = Path(args.text_file).read_text(encoding="utf-8")
    source_ids = torch.tensor(
        tokenizer(text, add_special_tokens=False).input_ids,
        dtype=torch.long,
    )

    results = {
        "benchmark": "end_to_end_dense_vs_hierarchical_int4",
        "chunk_size": args.chunk_size,
        "new_tokens": args.new_tokens,
        "dense_max_context": args.dense_max_context,
        "rows": [],
    }

    # Run the two implementations sequentially.  Keeping both 1B models on
    # the GPU makes the benchmark needlessly fragile, especially on a 32 GiB
    # card once long-context temporary buffers are included.
    dense_rows = {}
    dense_model = PythiaForCausalLM(PythiaConfig()).to(
        device=DEVICE,
        dtype=DTYPE,
    ).eval()
    try:
        load_official_weights(dense_model, model_dir)
        for context_length in contexts:
            print(f"dense context={context_length:,}")
            if context_length <= args.dense_max_context:
                ids = repeat_ids(source_ids, context_length)
                try:
                    dense_row = benchmark_dense(
                        dense_model,
                        ids,
                        chunk_size=args.chunk_size,
                        new_tokens=args.new_tokens,
                    )
                finally:
                    del ids
            else:
                dense_row = {
                    "model": "dense",
                    "prompt_length": context_length,
                    "status": "skipped_dense_guard",
                    "reason": "dense context exceeds dense_max_context",
                }
            dense_rows[context_length] = dense_row
            print(dense_row)
            clear_gpu_cache()
    finally:
        del dense_model
        gc.collect()
        clear_gpu_cache()

    routed_config = dict(HIERARCHICAL_ROUTING_CONFIG)
    routed_config["beam_width"] = args.beam_width
    results["routing_config"] = routed_config
    routed_rows = {}
    routed_model = load_ocean_model(
        model_dir=model_dir,
        routing=routed_config,
        device=DEVICE,
        dtype=DTYPE,
    )
    try:
        for context_length in contexts:
            print(f"hierarchical INT4 context={context_length:,}")
            estimate = estimated_full_int4_kv_gib(
                context_length,
                routed_model.config,
            )
            if estimate > args.max_cache_gib:
                routed_row = {
                    "model": "hierarchical_int4",
                    "prompt_length": context_length,
                    "status": "skipped_memory_guard",
                    "estimated_full_int4_kv_gib": estimate,
                }
            else:
                ids = repeat_ids(source_ids, context_length)
                try:
                    routed_row = benchmark_routed(
                        routed_model,
                        ids,
                        chunk_size=args.chunk_size,
                        new_tokens=args.new_tokens,
                    )
                finally:
                    del ids
                routed_row.update(
                    {
                        "model": "hierarchical_int4",
                        "status": "ok",
                        "estimated_full_int4_kv_gib": estimate,
                        "routing": "hierarchical_cosine",
                    }
                )
            routed_rows[context_length] = routed_row
            print(routed_row)
            clear_gpu_cache()
    finally:
        del routed_model
        gc.collect()
        clear_gpu_cache()

    for context_length in contexts:
        dense_row = dense_rows[context_length]
        routed_row = routed_rows[context_length]
        row = {
            "prompt_length": context_length,
            "dense": dense_row,
            "hierarchical_int4": routed_row,
        }
        if (
            dense_row.get("status") == "ok"
            and routed_row.get("status") == "ok"
        ):
            row["prefill_speedup_dense_over_hierarchical"] = (
                dense_row["prefill_seconds"]
                / routed_row["prefill_seconds"]
            )
            row["decode_speedup_dense_over_hierarchical"] = (
                dense_row["decode_seconds"]
                / routed_row["decode_seconds"]
            )
            row["total_speedup_dense_over_hierarchical"] = (
                dense_row["total_seconds"]
                / routed_row["total_seconds"]
            )
        elif dense_row.get("status") != "ok":
            row["comparison_status"] = "dense_skipped"
        else:
            row["comparison_status"] = "hierarchical_skipped"
        results["rows"].append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("saved:", output)


if __name__ == "__main__":
    main()
