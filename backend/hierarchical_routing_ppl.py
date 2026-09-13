#!/usr/bin/env python3
"""PPL comparison for dense, full-scan and indexed cosine routing.

The routed variants are evaluated token by token.  Each next-token query gets
its own route decision and attends to the exact dequantized K/V values selected
from the full INT4 cache.  This is slower than chunked prefill, but it avoids
assigning one representative route to several different queries.

This is a native-context quality benchmark.  It does not establish quality
outside the context on which the original Pythia checkpoint was trained.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

try:
    from .model import (
        DEVICE,
        DTYPE,
        HIERARCHICAL_ROUTING_CONFIG,
        PythiaForCausalLM,
        PythiaConfig,
        ROUTING_CONFIG,
        clear_gpu_cache,
        load_official_weights,
        load_ocean_model,
        synchronize,
    )
except ImportError:
    from model import (
        DEVICE,
        DTYPE,
        HIERARCHICAL_ROUTING_CONFIG,
        PythiaForCausalLM,
        PythiaConfig,
        ROUTING_CONFIG,
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
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument(
        "--protocol",
        choices=("tokenwise", "chunked"),
        default="tokenwise",
        help="tokenwise — строгий quality path; chunked — быстрый speed path",
    )
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--modes",
        default="dense,full_scan_cosine,hierarchical_cosine",
        help="варианты через запятую; для быстрого tree-теста укажите hierarchical_cosine",
    )
    parser.add_argument(
        "--allow-slow-tokenwise",
        action="store_true",
        help="разрешить token-by-token benchmark на контексте выше 2048",
    )
    parser.add_argument(
        "--skip-dense",
        action="store_true",
        help="не запускать dense baseline",
    )
    parser.add_argument(
        "--allow-dense-long-context",
        action="store_true",
        help="разрешить квадратичный dense baseline выше native context",
    )
    parser.add_argument("--beam-widths", default="32")
    parser.add_argument("--route-refresh-interval", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--route-blocks", type=int, default=16)
    parser.add_argument("--summary-parts", type=int, default=4)
    parser.add_argument("--local-window", type=int, default=256)
    parser.add_argument(
        "--output",
        default="./hierarchical_routing_ppl.json",
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


def repeat_ids(source_ids, length):
    if source_ids.numel() == 0:
        raise ValueError("Источник токенов пуст")
    repeats = math.ceil(length / source_ids.numel())
    return source_ids.repeat(repeats)[:length].contiguous()


def config_for(args, mode, beam_width):
    base = dict(
        ROUTING_CONFIG
        if mode == "full_scan_cosine"
        else HIERARCHICAL_ROUTING_CONFIG
    )
    base.update(
        {
            "route_mode": mode,
            "block_size": args.block_size,
            "route_blocks": args.route_blocks,
            "beam_width": beam_width,
            "summary_parts": args.summary_parts,
            "local_window": args.local_window,
            "route_refresh_interval": args.route_refresh_interval,
        }
    )
    return base


@torch.inference_mode()
def evaluate_dense(model, ids):
    ids = ids.to(
        device=model.gpt_neox.embed_in.weight.device,
        dtype=torch.long,
        non_blocking=True,
    )
    synchronize()
    started = time.perf_counter()
    logits, _ = model(ids.view(1, -1), use_cache=False)
    loss = F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, model.config.vocab_size),
        ids[1:],
        reduction="mean",
    )
    synchronize()
    seconds = time.perf_counter() - started
    mean_nll = float(loss)
    return {
        "routing": "dense",
        "tokens": int(ids.numel() - 1),
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "seconds": seconds,
        "tokens_per_second": (ids.numel() - 1) / max(seconds, 1e-9),
    }


def _route_stats(caches):
    route_calls = sum(cache.route_calls for cache in caches)
    nodes_scored = sum(cache.route_nodes_scored for cache in caches)
    return {
        "route_calls_all_layers": route_calls,
        "route_nodes_scored_all_layers": nodes_scored,
        "mean_nodes_scored_per_route": (
            nodes_scored / route_calls if route_calls else 0.0
        ),
        "routing_active": route_calls > 0,
    }


@torch.inference_mode()
def evaluate_routed_tokenwise(model, ids, mode, config):
    model.routing = dict(config)
    device = model.gpt_neox.embed_in.weight.device
    ids = ids.to(device=device, dtype=torch.long, non_blocking=True)
    caches = model.new_bounded_cache(int(ids.numel()))
    total_nll = 0.0
    total_tokens = 0
    synchronize()
    started = time.perf_counter()
    try:
        for position in range(ids.numel() - 1):
            logits = model.forward_bounded_token(
                ids[position],
                caches,
                position,
            )
            target = ids[position + 1].view(1)
            total_nll += float(
                F.cross_entropy(
                    logits[:, -1, :].float(),
                    target,
                    reduction="sum",
                )
            )
            total_tokens += 1
        synchronize()
        seconds = time.perf_counter() - started
        mean_nll = total_nll / max(total_tokens, 1)
        route_stats = _route_stats(caches)
        return {
            "routing": mode,
            "tokens": total_tokens,
            "mean_nll": mean_nll,
            "perplexity": math.exp(mean_nll),
            "seconds": seconds,
            "tokens_per_second": total_tokens / max(seconds, 1e-9),
            **route_stats,
            "config": dict(config),
        }
    finally:
        del caches


@torch.inference_mode()
def evaluate_routed_chunked(model, ids, mode, config, chunk_size):
    device = model.gpt_neox.embed_in.weight.device
    ids = ids.to(device=device, dtype=torch.long, non_blocking=True)
    caches = model.new_bounded_cache(int(ids.numel()))
    total_nll = 0.0
    total_tokens = 0
    synchronize()
    started = time.perf_counter()
    try:
        for start in range(0, ids.numel(), chunk_size):
            end = min(start + chunk_size, ids.numel())
            logits = model.forward_bounded_chunk(
                ids[start:end],
                caches,
                start,
            )
            target = ids[start + 1 : min(end + 1, ids.numel())]
            if target.numel():
                total_nll += float(
                    F.cross_entropy(
                        logits[:, : target.numel(), :].float().reshape(
                            -1,
                            model.config.vocab_size,
                        ),
                        target,
                        reduction="sum",
                    )
                )
                total_tokens += int(target.numel())
        synchronize()
        seconds = time.perf_counter() - started
        mean_nll = total_nll / max(total_tokens, 1)
        return {
            "routing": mode,
            "tokens": total_tokens,
            "mean_nll": mean_nll,
            "perplexity": math.exp(mean_nll),
            "seconds": seconds,
            "tokens_per_second": total_tokens / max(seconds, 1e-9),
            **_route_stats(caches),
            "config": dict(config),
        }
    finally:
        del caches


def main():
    args = parse_args()
    beam_widths = tuple(int(item) for item in args.beam_widths.split(","))
    if any(item <= 0 for item in beam_widths):
        raise ValueError("beam-widths должны быть положительными")
    modes = tuple(item.strip() for item in args.modes.split(",") if item.strip())
    allowed_modes = {
        "dense",
        "full_scan_cosine",
        "hierarchical_cosine",
    }
    unknown_modes = set(modes) - allowed_modes
    if not modes or unknown_modes:
        raise ValueError(
            f"Неизвестные modes: {sorted(unknown_modes)}; "
            f"доступны {sorted(allowed_modes)}"
        )
    if (
        args.protocol == "chunked"
        and args.context_length <= args.chunk_size
        and any(mode != "dense" for mode in modes)
    ):
        print(
            "WARNING: chunk_size >= context_length; "
            "routed past-context не будет активирован. "
            "Это фактически dense prefill для одного chunk."
        )
    if (
        args.context_length > PythiaConfig().max_position_embeddings
        and args.protocol == "tokenwise"
        and not args.allow_slow_tokenwise
    ):
        raise ValueError(
            "Tokenwise benchmark на контексте выше native слишком медленный. "
            "Используйте --protocol chunked --skip-dense или явно добавьте "
            "--allow-slow-tokenwise."
        )
    model_dir = resolve_model_dir(args)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    text = Path(args.text_file).read_text(encoding="utf-8")
    source_ids = torch.tensor(
        tokenizer(text, add_special_tokens=False).input_ids,
        dtype=torch.long,
    )
    ids = repeat_ids(source_ids, args.context_length)
    results = {
        "benchmark": "native_context_cosine_tree_ppl",
        "context_length": args.context_length,
        "protocol": args.protocol,
        "chunk_size": args.chunk_size,
        "modes": modes,
        "text_file": str(args.text_file),
        "rows": [],
    }

    dense_row = None
    should_run_dense = (
        "dense" in modes
        and not args.skip_dense
        and (
            args.allow_dense_long_context
            or args.context_length <= PythiaConfig().max_position_embeddings
        )
    )
    if should_run_dense:
        dense_model = PythiaForCausalLM(PythiaConfig()).to(
            device=DEVICE,
            dtype=DTYPE,
        ).eval()
        load_official_weights(dense_model, model_dir)
        dense_row = evaluate_dense(dense_model, ids)
        results["dense"] = dense_row
        results["rows"].append(dense_row)
        del dense_model
        clear_gpu_cache()
        print("--- dense ---")
        print(dense_row)
    else:
        results["dense"] = {
            "status": "skipped",
            "reason": (
                "context exceeds native context; use "
                "--allow-dense-long-context to override"
            ),
        }
        print("--- dense skipped ---")

    for mode in ("full_scan_cosine", "hierarchical_cosine"):
        if mode not in modes:
            continue
        widths = (32,) if mode == "full_scan_cosine" else beam_widths
        for beam_width in widths:
            config = config_for(args, mode, beam_width)
            model = load_ocean_model(
                model_dir=model_dir,
                routing=config,
                device=DEVICE,
                dtype=DTYPE,
            )
            try:
                if args.protocol == "tokenwise":
                    row = evaluate_routed_tokenwise(
                        model,
                        ids,
                        mode,
                        config,
                    )
                else:
                    row = evaluate_routed_chunked(
                        model,
                        ids,
                        mode,
                        config,
                        args.chunk_size,
                    )
                if dense_row is not None:
                    row["ppl_delta_vs_dense"] = (
                        row["perplexity"] - dense_row["perplexity"]
                    )
                    row["ppl_relative_percent_vs_dense"] = 100.0 * (
                        row["perplexity"] / dense_row["perplexity"] - 1.0
                    )
                results["rows"].append(row)
                print(f"--- {mode}, beam={beam_width} ---")
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
