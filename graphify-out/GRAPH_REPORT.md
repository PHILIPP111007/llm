# Graph Report - llm  (2026-09-17)

## Corpus Check
- 10 files · ~25,494 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 4 file(s) not represented in the graph (top: .ipynb 3, (none) 1)

## Summary
- 318 nodes · 603 edges · 10 communities
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS · INFERRED: 1 edges (avg confidence: 0.95)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `facbc937`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- model.py
- Исследование каскадной маршрутизации блоков KV-cache для эффективной обработки длинного контекста в GPT-подобной модели
- routing_ablation_benchmark.py
- INT4RoutedKVCache
- Handoff: Pythia-1B Ocean / sublinear attention
- .__init__
- Измеренные результаты
- MultiScaleKVPositionReranker
- 6. История ключевых экспериментов
- AblationKVCache

## God Nodes (most connected - your core abstractions)
1. `INT4RoutedKVCache` - 20 edges
2. `Исследование каскадной маршрутизации блоков KV-cache для эффективной обработки длинного контекста в GPT-подобной модели` - 19 edges
3. `PythiaConfig` - 18 edges
4. `clear_gpu_cache()` - 16 edges
5. `Handoff: Pythia-1B Ocean / sublinear attention` - 15 edges
6. `main()` - 13 edges
7. `main()` - 13 edges
8. `load_ocean_model()` - 13 edges
9. `AblationKVCache` - 12 edges
10. `PythiaForCausalLM` - 11 edges

## Surprising Connections (you probably didn't know these)
- `На каких признаках обучался reranker` --references--> `MultiScaleKVPositionReranker`  [INFERRED]
  docs/Handoff.md → backend/routing_ablation_benchmark.py
- `main()` --calls--> `clear_gpu_cache()`  [EXTRACTED]
  backend/end_to_end_speed_benchmark.py → backend/model.py
- `main()` --calls--> `PythiaConfig`  [EXTRACTED]
  backend/end_to_end_speed_benchmark.py → backend/model.py
- `estimated_full_int4_kv_gib()` --calls--> `PythiaConfig`  [EXTRACTED]
  backend/hierarchical_routing_benchmark.py → backend/model.py
- `run_mode()` --calls--> `clear_gpu_cache()`  [EXTRACTED]
  backend/hierarchical_routing_benchmark.py → backend/model.py

## Import Cycles
- None detected.

## Communities (10 total, 0 thin omitted)

### Community 0 - "model.py"
Cohesion: 0.08
Nodes (60): argparse, benchmark_dense(), main(), parse_args(), parse_int_list(), inference_mode, End-to-end speed comparison: dense Pythia versus hierarchical INT4. Both models…, resolve_model_dir() (+52 more)

### Community 1 - "Исследование каскадной маршрутизации блоков KV-cache для эффективной обработки длинного контекста в GPT-подобной модели"
Cohesion: 0.04
Nodes (49): 14K incremental decode performance, 1. Build a recent-query summary, 2. Score visible blocks, 3. Add mandatory local blocks, 4. Select five semantic blocks, 5. Add an exploration block, Adaptive route budgets, Asymptotic estimate for the hybrid variant (+41 more)

### Community 2 - "routing_ablation_benchmark.py"
Cohesion: 0.11
Nodes (36): estimated_full_int4_kv_gib(), main(), parse_args(), Final speed benchmark for the production Ocean INT4 cosine baseline. This…, Estimate exact full INT4 K/V storage, excluding all other tensors., resolve_model_dir(), clear_gpu_cache(), PythiaConfig (+28 more)

### Community 3 - "INT4RoutedKVCache"
Cohesion: 0.13
Nodes (10): INT4FullKVCache, INT4RoutedKVCache, no_grad, Exact full KV-cache: every token is stored, packed as two signed INT4 values., Full INT4 storage with selectable indexed block routing.…, Reset routing counters without clearing the KV-cache., Return a Long index on the same device as the cache buffers. Routing indices…, Cosine score complete block summaries for each attention head. (+2 more)

### Community 4 - "Handoff: Pythia-1B Ocean / sublinear attention"
Cohesion: 0.07
Nodes (29): 10. Что пока не доказано, 11. Основные файлы, 12. Воспроизводимость, 13. План дальнейшей работы, 14. Итоговый статус, 1. Цель проекта, 2. Базовая модель, 3. Архитектура Ocean (+21 more)

### Community 5 - ".__init__"
Cohesion: 0.11
Nodes (9): apply_rotary(), OceanAttention, OceanINT4PythiaForCausalLM, inference_mode, PythiaAttention, PythiaDecoderLayer, PythiaMLP, RotaryEmbedding (+1 more)

### Community 6 - "Измеренные результаты"
Cohesion: 0.08
Nodes (23): Controlled production baseline: cosine routing + full INT4, End-to-end dense vs hierarchical INT4, Hierarchical routing, INT4 KV-cache, INT4 routed speed, Native-context quality, PPL на контексте до 2048 токенов, Speed: от 14K до 1M токенов (+15 more)

### Community 7 - "MultiScaleKVPositionReranker"
Cohesion: 0.09
Nodes (17): BlockReranker, MultiScaleKVPositionReranker, Return scores for query [..., D] and summaries [..., X, D]., Reranker using multi-part K/V summaries and relative block position., Score blocks from query, multi-part K/V summaries and position., Small query/block scorer with a cosine-equivalent safe initialization., 8. Neural reranker, Итог эксперимента (+9 more)

### Community 8 - "6. История ключевых экспериментов"
Cohesion: 0.15
Nodes (13): 6.10. End-to-end speed: dense против hierarchical INT4, 6.1. Базовая Pythia и первая Ocean-версия, 6.2. Multi-summary и query-dependent routing, 6.3. Refresh interval, 6.4. Полный INT4 routed speed, 6.5. Практический sweep hierarchical routing, 6.6. PPL cosine tree при `block_size=64`, 6.7. Chunked speed/PPL после оптимизации cache (+5 more)

### Community 9 - "AblationKVCache"
Cohesion: 0.33
Nodes (3): AblationKVCache, no_grad, INT4 full cache with selectable first-stage and second-stage routing.

## Knowledge Gaps
- **103 isolated node(s):** `Архитектура`, `Текущая конфигурация`, `INT4 KV-cache`, `Hierarchical routing`, `Память полного cache` (+98 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 144 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Handoff: Pythia-1B Ocean / sublinear attention` connect `Handoff: Pythia-1B Ocean / sublinear attention` to `6. История ключевых экспериментов`, `Измеренные результаты`, `MultiScaleKVPositionReranker`?**
  _High betweenness centrality (0.540) - this node is a cross-community bridge._
- **Why does `MultiScaleKVPositionReranker` connect `MultiScaleKVPositionReranker` to `routing_ablation_benchmark.py`?**
  _High betweenness centrality (0.501) - this node is a cross-community bridge._
- **What connects `Архитектура`, `Текущая конфигурация`, `INT4 KV-cache` to the rest of the system?**
  _103 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `model.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07785547785547786 - nodes in this community are weakly interconnected._
- **Should `Исследование каскадной маршрутизации блоков KV-cache для эффективной обработки длинного контекста в GPT-подобной модели` be split into smaller, more focused modules?**
  _Cohesion score 0.04081632653061224 - nodes in this community are weakly interconnected._
- **Should `routing_ablation_benchmark.py` be split into smaller, more focused modules?**
  _Cohesion score 0.1101010101010101 - nodes in this community are weakly interconnected._
- **Should `INT4RoutedKVCache` be split into smaller, more focused modules?**
  _Cohesion score 0.1330049261083744 - nodes in this community are weakly interconnected._