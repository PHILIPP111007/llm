# Handoff: Pythia-1B Ocean / sublinear attention

Дата фиксации: 2026-09-13

Этот документ описывает цель проекта, реализованную архитектуру, проведённые
эксперименты, достигнутые результаты, ограничения и порядок дальнейшей работы.
Главный принцип: speed и quality нужно рассматривать раздельно. Прототип умеет
обрабатывать очень длинные входы в инженерном смысле, но качество исходной
Pythia-1B за пределами обученного контекста 2048 токенов пока не доказано.

## 1. Цель проекта

Цель — исследовать прозрачную GPT-подобную модель, способную обрабатывать
сверхдлинный контекст за счёт:

1. полного KV-cache с INT4-квантизацией;
2. local/sliding-window attention;
3. content-dependent routing по блокам KV-cache;
4. cosine-поиска релевантных блоков;
5. hierarchical routing по дереву summaries;
6. neural reranking после дешёвого cosine-фильтра.

Базовая модель — Pythia-1B с официальными весами EleutherAI. В основном
эксперименте оптимизировался inference path; полноценное переобучение модели
под миллион токенов не выполнялось.

## 2. Базовая модель

Pythia-1B была реализована вручную на PyTorch. Канонический model-only модуль:
[model.py](../backend/model.py).

Конфигурация:

| Параметр | Значение |
|---|---:|
| Vocabulary | 50,304 |
| Hidden size | 2,048 |
| Intermediate size | 8,192 |
| Layers | 16 |
| Attention heads | 8 |
| Head dimension | 256 |
| Native context | 2,048 |
| RoPE percentage | 25% |
| Parameters | примерно 1.012B |

Старое имя pythia_ocean_int4_train.py больше не является каноническим.
Актуальный model-only модуль — model.py; код обучения в него не входит.

## 3. Архитектура Ocean

История K/V для каждого слоя и головы делится на блоки. Для каждого блока
хранятся summaries. При запросе выполняется:

    текущий query Q
            ↓
    cosine similarity с summaries
            ↓
    Top-X кандидатов или hierarchical beam search
            ↓
    обязательные global/local blocks
            ↓
    Top-Y semantic blocks
            ↓
    точный causal attention по выбранным токенам

Attention внутри выбранных блоков остаётся точным. Approximation возникает
только потому, что токены вне route не участвуют в текущем attention.

Текущий production config:

| Параметр | Значение |
|---|---:|
| block_size | 256 |
| route_blocks | 16 |
| beam_width | 32 |
| summary_parts | 4 |
| global_blocks | 1 |
| local_blocks | 2 |
| local_window | 256 |
| route_refresh_interval | 64 |

Local window ограничивает recent candidates, но старые токены из полного cache
не удаляет.

## 4. INT4 KV-cache

Реализован packed INT4 KV-cache с отдельными scale для K и V. В текущей версии
для routing также поддерживаются persistent multi-part summaries ключей и
значений, обновляемые при добавлении токенов.

Full INT4-cache означает, что K/V сохраняются для каждого токена. Routing не
уменьшает число сохранённых токенов; он уменьшает число токенов, извлекаемых
для конкретного attention-запроса.

Оценки памяти полного cache:

| Контекст | FP16 full KV-cache | INT4 full KV-cache |
|---:|---:|---:|
| 32,768 | 4.00 GiB | 1.02 GiB |
| 131,072 | 16.00 GiB | 4.06 GiB |
| 1,000,000 | 122.07 GiB | 30.99 GiB |

Оценки не включают веса модели, временные attention buffers, allocator
fragmentation и прочие расходы. Поэтому 1M-token full INT4-cache близок к
пределу 32 GiB GPU и не гарантирует запуск на одной V100S.

Отдельно bounded local-plus-logarithmic-segments cache выполнил speed-only
тест на 1M:

| Метрика | Результат |
|---|---:|
| Prompt | 1,000,000 токенов |
| Prefill | 274.35 s |
| Prefill throughput | 3,645 tok/s |
| Decode throughput | 54.02 tok/s |
| Peak allocated | 7.71 GiB |

Это другой cache path и не является доказательством успешного million-token
full-cache retrieval.

## 5. Асимптотика

Пусть N — длина контекста, D — размер головы, B — размер блока, K — число
выбранных токенов, R — interval обновления route.

Dense decode attention работает за O(N · D) на один новый токен. Полная prefill
attention matrix обычно имеет квадратичную стоимость O(N²).

При фиксированных route_blocks и block_size selected attention работает как:

    O(K · D)

То есть относительно N attention kernel имеет bounded, практически O(1)
стоимость. Это не означает, что весь Transformer работает за O(1):
projections, MLP, prefill и memory movement остаются.

Full-scan route refresh:

    O((N / B) · D)
    O((N / B) · D / R) в среднем на decode-токен

Hierarchical refresh при balanced tree:

    O(beam · log(N / B) · D)

Корректное описание текущего прототипа:

    selected attention  — O(1) относительно N при фиксированном route;
    hierarchical route  — примерно O(log N) на refresh;
    полный KV-cache      — O(N) по памяти.

Для prefill каждый input token всё равно должен быть обработан. Идеализированно
attention/routing path может быть близким к линейному по N, но wall-clock
зависит от chunking, gather, Python loops, kernel launches и synchronization.

## 6. История ключевых экспериментов

### 6.1. Базовая Pythia и первая Ocean-версия

Официальные веса были загружены в ручную реализацию. На Tiny Shakespeare
получен baseline:

    dense mean NLL   = 3.0701
    dense PPL        = 21.54

Первая грубая Ocean-конфигурация:

    Ocean mean NLL   = 3.4758
    Ocean PPL        = 32.33

Она ухудшала PPL примерно на 50% и была медленнее dense. Основные причины:
один грубый summary, дорогая Python-side маршрутизация и irregular memory access.

### 6.2. Multi-summary и query-dependent routing

После перехода к четырём summaries на блок, query-dependent routing, global
block и увеличению route budget:

| Route blocks | Summary parts | PPL |
|---:|---:|---:|
| 8 | 1 | 28.99 |
| 8 | 4 | 21.87 |
| 12 | 4 | 21.66 |
| 16 | 4 | 21.57 |
| 24 | 4 | 21.54 |
| 32 | 4 | 21.54 |

Главный вывод: качество улучшилось не только за счёт количества блоков, но и за
счёт более информативного block representation.

### 6.3. Refresh interval

На 14K prompt, только speed benchmark:

| Вариант | Refresh | Prefill s | Decode tok/s | Total s |
|---|---:|---:|---:|---:|
| Dense | — | 6.854 | 37.65 | 8.528 |
| Incremental Ocean | 4 | 6.856 | 54.68 | 8.008 |
| Incremental Ocean | 16 | 6.870 | 61.88 | 7.888 |

Refresh 16 давал более быстрый decode при почти неизменном native-context
PPL. Total speedup оставался ограничен dense prefill.

### 6.4. Полный INT4 routed speed

Для full INT4 routed модели, chunk_size 256 и new_tokens 16:

| Prompt | Prefill s | Prefill tok/s | Decode tok/s | Total s | Estimated cache |
|---:|---:|---:|---:|---:|---:|
| 2,048 | 0.704 | 2,911 | 30.14 | 1.234 | 0.064 GiB |
| 14,000 | 5.984 | 2,340 | 27.53 | 6.565 | 0.434 GiB |
| 32,000 | 14.411 | 2,221 | 25.72 | 15.033 | 0.992 GiB |
| 100,000 | 47.633 | 2,099 | 27.53 | 48.214 | 3.100 GiB |

Decode tok/s — скорость генерации новых токенов. Prefill tok/s — скорость
обработки входного prompt.

### 6.5. Практический sweep hierarchical routing

Для оценки постоянных факторов hierarchical routing создан
`backend/hierarchical_routing_sweep.py`. Веса модели загружались один раз,
для каждой конфигурации создавался независимый полный INT4 KV-cache.
Проверялись `beam_width=8,16,32`, `route_refresh_interval=1,4,64`,
`chunk_size=256` и генерация 64 новых токенов.

#### Контекст 32,768

| `beam_width` | Refresh 1 total s | Refresh 4 total s | Refresh 64 total s | Decode tok/s при refresh 64 |
|---:|---:|---:|---:|---:|
| 8 | 15.875 | 14.606 | 13.962 | 24.78 |
| 16 | 15.539 | 14.499 | 13.875 | 24.87 |
| 32 | 15.525 | 14.535 | 13.902 | 24.86 |

#### Контекст 100,000

| `beam_width` | Refresh 1 total s | Refresh 4 total s | Refresh 64 total s | Decode tok/s при refresh 64 |
|---:|---:|---:|---:|---:|
| 8 | 41.527 | 40.385 | 39.820 | 25.22 |
| 16 | 41.570 | 40.472 | 39.633 | 25.25 |
| 32 | 41.588 | 40.432 | 39.964 | 25.15 |

Средняя стоимость маршрутизации на один route при `refresh=64`:

| Контекст | Beam 8 | Beam 16 | Beam 32 |
|---:|---:|---:|---:|
| 32,768 | 1,113 узлов | 1,314 узлов | 2,044 узла |
| 100,000 | 1,359 узлов | 1,617 узлов | 2,700 узлов |

Основные наблюдения:

1. Уменьшение beam с 32 до 8 сокращает число посещаемых узлов примерно в два
   раза, но почти не меняет prefill wall-clock в этом прототипе.
2. `refresh_interval=64` оказался лучшим по decode: около 25 tok/s против
   примерно 20 tok/s при refresh 4 и 15 tok/s при refresh 1.
3. На 100K лучший измеренный total был у `beam_width=16,
   refresh_interval=64`: 39.633 s. Разница с beam 8 небольшая.
4. Hierarchical routing сохраняет логарифмический рост числа индексных
   операций, но на 32K–100K всё ещё не превосходит full-scan по end-to-end
   prefill: дерево, `top-k`, `gather` и Python/PyTorch overhead дают большой
   постоянный множитель.

Это speed/index benchmark. В данном sweep не измерялись PPL, Recall и needle
retrieval, поэтому конфигурация `beam=8` или `beam=16` не может быть выбрана
как production-конфигурация только по скорости.

### 6.6. PPL cosine tree при `block_size=64`

Для проверки влияния beam search был выполнен отдельный token-by-token PPL
benchmark на контексте 2048 токенов. Использовались один и тот же фрагмент
Tiny Shakespeare, полный INT4 KV-cache, `route_blocks=16`,
`summary_parts=4`, `local_window=256` и `route_refresh_interval=64`.

| Routing | Beam width | Mean NLL | PPL | Δ PPL к dense | Δ к dense, % | Tok/s |
|---|---:|---:|---:|---:|---:|---:|
| Dense | — | 3.0693 | 21.5260 | — | — | 10,698 |
| Full-scan cosine | 32 | 3.1519 | 23.3812 | +1.8553 | +8.62% | 35.68 |
| Hierarchical cosine | 4 | 3.1750 | 23.9269 | +2.4009 | +11.15% | 34.94 |
| Hierarchical cosine | 8 | 3.1777 | 23.9919 | +2.4659 | +11.46% | 35.06 |
| Hierarchical cosine | 16 | 3.1635 | 23.6525 | +2.1265 | +9.88% | 35.11 |
| Hierarchical cosine | 32 | 3.1499 | 23.3348 | +1.8088 | +8.40% | 35.04 |

Среднее число проверенных index-узлов на один route:

| Routing | Beam width | Узлов на route |
|---|---:|---:|
| Full-scan cosine | 32 | 110 |
| Hierarchical cosine | 4 | 414 |
| Hierarchical cosine | 8 | 431 |
| Hierarchical cosine | 16 | 479 |
| Hierarchical cosine | 32 | 499 |

Результат показывает, что при меньшем beam качество cosine tree ухудшается.
Только `beam_width=32` приблизился к full-scan: PPL `23.3348` против `23.3812`.
Однако tree в этом протоколе проверяет примерно в 4–4.5 раза больше index-узлов
и не даёт практического speedup.

Сравнение с dense нельзя интерпретировать как чистый эффект routing: разница
включает INT4-квантизацию KV, sparse block selection и ошибку hierarchical
поиска. Для изоляции этих факторов нужен отдельный full-INT4 контроль с
полным набором блоков без sparse-ограничения.

### 6.7. Качество на длинных контекстах

Исходная Pythia обучена на context 2048. Поэтому 14K, 32K и 1M тесты нельзя
считать valid quality benchmarks.

Full INT4 native-context control:

    context             = 2,048
    needle text match   = True
    needle PPL          ≈ 3.96

На 32K synthetic needle full INT4 control не смог надёжно извлечь needle,
PPL был примерно 18,757. Это не доказывает, что именно INT4 виноват:
модель не обучалась на 32K, а RoPE использовался за пределами native range.

На PG-19 были получены следующие значения в текущем непереобученном checkpoint:

| Context | PPL |
|---:|---:|
| 2,048 | примерно 1.40 в smoke-test протоколе |
| 8,192 | примерно 188.55 |
| 16,384 | примерно 337.75 |

Значения зависят от документа и protocol, но вывод устойчив: inference
optimization не заменяет long-context training.

## 7. Controlled routing ablation

Созданы:
[Pythia_1B_routing.ipynb](../notebooks/Pythia_1B_routing.ipynb) и
[routing_ablation_benchmark.py](../backend/routing_ablation_benchmark.py).

Сравнивались:

    dense
    full_scan_cosine
    hierarchical_cosine
    full_scan_reranker
    hierarchical_reranker
    neural_full_scan

neural_full_scan — learned selector без cosine prefilter.

### PPL на 2048

Актуальный v3 PPL benchmark был сохранён во время последнего GPU-запуска.
Ключевые значения:

| Variant | PPL | Delta vs dense |
|---|---:|---:|
| Dense | 21.52597 | 0.00000 |
| Full-scan cosine | 21.71604 | +0.19007 |
| Hierarchical cosine | 21.71604 | +0.19007 |
| Full-scan reranker v3 | 21.71773 | +0.19176 |
| Hierarchical reranker v3 | 21.71773 | +0.19176 |
| Neural full scan v3 | 21.71773 | +0.19176 |

Cosine routing сохраняет качество близкое к dense на native context, но v3
reranker PPL не улучшил.

### Recall diagnostic

Oracle — Top-16 blocks по dense attention mass. В исходном diagnostic был
использован block size 16, чтобы Recall@64 не стал тривиальным. Production
reranker обучался на block size 256, поэтому этот тест является
out-of-distribution и не должен трактоваться как окончательная production
оценка.

Последний v3 diagnostic:

| Metric | Result |
|---|---:|
| Cosine Recall@64 | 99.54% |
| Cosine-only Recall@16 | 86.82% |
| Reranker after cosine Recall@16 | 68.51% |
| Neural full scan Recall@16 | 67.94% |
| Hierarchical reranker Recall@16 | 68.51% |
| Reranker latency | 0.84 ms/head |

Старый v2 давал около 68.53% Recall@16. Поэтому v3 пока не показал улучшения.
Основная проблема — mismatch block size и mismatch между token-level teacher
labels и chunk-level route application.

### Speed ablation

Один из запусков на context 2048:

| Variant | Decode tok/s | Total s |
|---|---:|---:|
| Dense | 75.31 | 0.293 |
| Full-scan cosine | 30.02 | 0.901 |
| Hierarchical cosine | 27.72 | 1.060 |
| Full-scan reranker | 29.53 | 0.924 |
| Hierarchical reranker | 28.12 | 1.022 |
| Neural full scan | 29.67 | 0.935 |

На коротком context routing медленнее dense из-за Python-side selection,
irregular gathers, small GPU kernels и отсутствия fused block-sparse kernel.

На длинных prompt full-scan в текущей реализации оказался быстрее hierarchy:

| Context | Full-scan prefill tok/s | Hierarchical prefill tok/s |
|---:|---:|---:|
| 14,000 | 4,382 | 2,902 |
| 32,768 | 4,191 | 2,460 |
| 100,000 | 3,981 | 2,272 |

Это не опровержение асимптотики дерева; это демонстрация слишком большого
constant factor текущей Python/GPU реализации hierarchy.

### v4 chunk-union: long-context speed and Recall

После обучения `block-reranker-v4-chunk-union.pt` выполнен benchmark на
контекстах 16K и 32K. При production `block_size=256` это соответственно 64 и
128 доступных блоков, поэтому speed-тест уже не вырождается в случай 8 блоков.

| Context | Routing | Prefill tok/s | Decode tok/s | Total s | Routing s |
|---:|---|---:|---:|---:|---:|
| 16,384 | Full-scan cosine | 3,810 | 22.26 | 5.019 | 0.664 |
| 16,384 | Full-scan v4 reranker | 3,409 | 22.30 | 5.524 | 1.319 |
| 16,384 | Hierarchical cosine | 2,616 | 21.30 | 7.014 | 2.874 |
| 16,384 | Hierarchical v4 reranker | 2,348 | 21.46 | 7.722 | 3.597 |
| 32,768 | Full-scan cosine | 4,030 | 22.96 | 8.828 | 1.059 |
| 32,768 | Full-scan v4 reranker | 3,384 | 22.51 | 10.393 | 2.609 |
| 32,768 | Hierarchical cosine | 2,388 | 21.67 | 14.458 | 6.647 |
| 32,768 | Hierarchical v4 reranker | 2,151 | 21.32 | 15.985 | 8.167 |

В сравнении с full-scan cosine v4 увеличил total latency на 10.1% при 16K и
на 17.7% при 32K. Hierarchical routing в текущей Python-реализации оказался
медленнее full-scan на 39.7–63.8%, а hierarchical v4 — на 53.9–81.1%.

Recall diagnostic дал следующий результат:

| Metric | v4 result |
|---|---:|
| Cosine Recall@64 | 93.75% |
| Hierarchical Recall@64 | 93.75% |
| Cosine-only Recall@16 | 51.74% |
| Cosine Top-64 → v4 Top-16 | 35.62% |
| Neural full-scan v4 Top-16 | 33.88% |
| Hierarchical Top-64 → v4 Top-16 | 35.62% |
| Mean reranker latency | 0.769 ms/head |

Итог отрицательный: v4 не улучшил cosine-only routing и снизил Recall@16 с
51.74% до 35.62%. Однако Recall diagnostic использовал `block_size=16`, тогда
как v4 обучался на `block_size=256`; это out-of-distribution проверка и не
является окончательным production-выводом. Тем не менее reranker также не дал
улучшения на PPL 2048 и добавил latency, поэтому текущий production baseline —
full-scan cosine без neural reranking.

### Следующий эксперимент: v5 с block_size=16

Для проверки гипотезы о более мелких semantic blocks benchmark теперь позволяет
переопределять `teacher_block_size`, `teacher_summary_parts`,
`routing_block_size` и `routing_summary_parts`. v5 должен обучаться и
тестироваться с одной и той же конфигурацией; v4 и v5 нельзя сравнивать как
один и тот же reranker.

Обучение v5:

    ./.venv/bin/python backend/routing_ablation_benchmark.py \
      --mode train-reranker \
      --model-dir /home/froschin/.cache/huggingface/hub/models--EleutherAI--pythia-1b/snapshots/f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2 \
      --text-file ./tinyshakespeare.txt \
      --reranker-output ./checkpoints/block-reranker-v5-b16.pt \
      --reranker-steps 1500 \
      --reranker-sequences 32 \
      --reranker-query-stride 64 \
      --teacher-mode chunk_union \
      --teacher-block-size 16 \
      --teacher-summary-parts 2 \
      --teacher-chunk-size 256 \
      --teacher-local-window 256

PPL на native context:

    ./.venv/bin/python backend/routing_ablation_benchmark.py \
      --model-dir /home/froschin/.cache/huggingface/hub/models--EleutherAI--pythia-1b/snapshots/f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2 \
      --text-file ./tinyshakespeare.txt \
      --reranker-checkpoint ./checkpoints/block-reranker-v5-b16.pt \
      --routing-block-size 16 \
      --routing-summary-parts 2 \
      --contexts 2048 \
      --output ./routing_ablation_v5_b16_2048.json

Long-context speed:

    ./.venv/bin/python backend/routing_ablation_benchmark.py \
      --model-dir /home/froschin/.cache/huggingface/hub/models--EleutherAI--pythia-1b/snapshots/f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2 \
      --text-file ./tinyshakespeare.txt \
      --reranker-checkpoint ./checkpoints/block-reranker-v5-b16.pt \
      --routing-block-size 16 \
      --routing-summary-parts 2 \
      --contexts 2048,16384,32768 \
      --skip-ppl \
      --skip-needle \
      --output ./routing_ablation_v5_b16_speed.json

Успех v5 означает: Recall@16 выше cosine-only, PPL не хуже dense на 2048 и
приемлемая route latency. Даже успешный v5 не уменьшит O(N) память полного
INT4 KV-cache; он только уменьшит гранулярность и потенциальный selected
attention workload.

v5 обучен:

    steps: 1500
    teacher samples: 3584
    final total loss: 2.2496
    final listwise loss: 2.2441
    final ranking loss: 0.0220

Loss v5 нельзя напрямую сравнивать с v4: при `block_size=16` в контексте 2048
доступно 128 блоков против примерно 8 блоков при `block_size=256`, поэтому
listwise classification существенно сложнее.

### Needle retrieval

На 2048 synthetic needle benchmark ни один вариант не дал exact text match.
Модель предсказывала частичный текст \nBIT-314159 вместо полного
 ORBIT-314159. Тест нужно повторить с нормализацией ведущего пробела, разными
позициями needle и несколькими random needle strings.

## 8. Neural reranker

### Старый reranker

Первая BlockReranker получала только query Q и mean(K) блока. Teacher target
был нормированной dense attention mass.

Старый checkpoint:

    checkpoints/block-reranker.pt
    steps          = 500
    sequences      = 8
    query_stride   = 128

v2:

    checkpoints/block-reranker-v2.pt
    steps          = 1500
    sequences      = 32
    query_stride   = 64

v2 дал небольшой положительный сдвиг PPL относительно старой версии, но не
решил проблему Recall.

### Текущий reranker v3

MultiScaleKVPositionReranker получает:

    текущий Q
    четыре K summaries внутри блока
    четыре V summaries внутри блока
    relative block position
    logarithmic distance

Обучение использует:

    listwise dense-attention distillation loss
    hard-negative ranking loss

Checkpoint:

    checkpoints/block-reranker-v3.pt
    architecture   = multiscale_kv_position_v1
    steps          = 1500
    sequences      = 32
    query_stride   = 64

Loader понимает metadata и сохраняет backward compatibility со старыми
checkpoint’ами.

v3 технически работает внутри routed attention, но пока не дал научно
положительного результата: PPL немного хуже v2, а Recall практически такой же.

### Полный эксперимент с neural reranking

Neural reranking был добавлен как второй этап маршрутизации:

    1. cosine или hierarchical selector выбирает candidate blocks;
    2. небольшой neural reranker пересчитывает score только этих кандидатов;
    3. выбираются финальные semantic blocks;
    4. mandatory global/local blocks добавляются отдельно.

Reranker не является второй языковой моделью и не изменяет веса Pythia. Он
используется только для выбора блоков KV-cache.

#### Какие данные использовались для обучения

Teacher — dense Pythia attention. Для каждого sampled query вычислялась dense
attention mass по блокам, после чего mass нормировалась и использовалась как
target распределение важности.

Источником текста был `tinyshakespeare.txt`. Из него формировались циклически
повторяемые последовательности длиной 2048 токенов:

| Версия | Steps | Sequences | Query stride | Teacher samples | Block size |
|---|---:|---:|---:|---:|---:|
| v1 | 500 | 8 | 128 | не зафиксировано | 256 |
| v2 | 1500 | 32 | 64 | не зафиксировано | 256 |
| v3 | 1500 | 32 | 64 | не зафиксировано | 256 |
| v4 | 1500 | 32 | 64 | 2,560 | 256 |
| v5 | 1500 | 32 | 64 | 3,584 | 16 |

Это не полноценный разнообразный train/validation/test corpus: данные v1–v5
получены из одного Shakespeare-текста и повторяются циклически. Поэтому
результаты reranker нельзя считать обобщением на книги, код или технические
документы.

#### На каких признаках обучался reranker

v1 использовал:

- текущий query-вектор `Q`;
- одно среднее `K`-представление блока;
- cosine similarity;
- признаки `Q`, `K`, `Q*K`, `abs(Q-K)` после low-rank projections.

v3/v4/v5 использовали `MultiScaleKVPositionReranker`:

- текущий query `Q` последнего токена chunk;
- несколько K-summaries внутри каждого блока;
- несколько V-summaries внутри каждого блока;
- mean/max/first/last агрегаты K-представлений;
- взаимодействия query с K и V;
- нормированную позицию блока;
- `log1p` относительной дистанции блока;
- исходную cosine score как baseline плюс residual neural score.

Для v3/v4 использовалось 4 summary parts на блок. Для v5 использовалось 2
summary parts при `block_size=16`.

#### Как формировалась objective function

Использовалась комбинация:

    listwise_loss = cross_entropy между dense attention mass и scores reranker
    ranking_loss = hard-negative margin loss
    total_loss = listwise_loss + 0.25 * ranking_loss

Hard negative выбирался из блоков с высокой cosine similarity, но с меньшей
dense attention mass. Это должно было научить reranker исправлять ошибки cosine
selector, а не просто копировать его.

В v4 был введён `chunk_union` protocol: один route используется для всего
prefill chunk, поэтому teacher усреднял dense attention mass нескольких query
внутри chunk. В target попадали только semantic blocks; local/global mandatory
blocks исключались.

#### Результаты PPL на native context 2048

Dense baseline во всех сравнениях: `PPL=21.52597`.

v4, `block_size=256`:

| Variant | PPL | Δ к dense |
|---|---:|---:|
| Full-scan cosine | 21.71604 | +0.19007 |
| Full-scan v4 reranker | 21.71688 | +0.19091 |
| Hierarchical v4 reranker | 21.71688 | +0.19091 |
| Neural full scan v4 | 21.71688 | +0.19091 |

v5, `block_size=16`:

| Variant | PPL | Δ к dense |
|---|---:|---:|
| Full-scan cosine | 21.97970 | +0.45373 |
| Hierarchical cosine | 22.10494 | +0.57897 |
| Full-scan v5 reranker | 21.92365 | +0.39768 |
| Hierarchical v5 reranker | 22.09641 | +0.57044 |
| Neural full scan v5 | 21.93103 | +0.40506 |

Таким образом, v5 улучшил PPL относительно собственного cosine baseline всего
на `0.05605`, но всё ещё был хуже dense на `1.85%`.

#### Результаты Recall

Recall diagnostic использовал `block_size=16`, 216 samples и oracle Top-16 по
dense attention mass.

v4:

| Metric | Result |
|---|---:|
| Cosine Recall@64 | 93.75% |
| Hierarchical Recall@64 | 93.75% |
| Cosine-only Recall@16 | 51.74% |
| Cosine Top-64 → v4 Top-16 | 35.62% |
| Neural full-scan v4 Top-16 | 33.88% |
| Hierarchical Top-64 → v4 Top-16 | 35.62% |
| Mean reranker latency | 0.769 ms/head |

Для v5 в файле `routing_ablation_v5_b16_2048.json` Recall diagnostic не был
сохранён; там есть PPL, speed и needle. Поэтому утверждать, что v5 улучшил
Recall@16, пока нельзя.

#### Результаты speed и needle для v5

На контексте 2048:

| Variant | Prefill tok/s | Decode tok/s | Total s |
|---|---:|---:|---:|
| Dense | 24,894 | 84.15 | 0.272 |
| Full-scan cosine | 1,664 | 32.08 | 1.730 |
| Full-scan v5 reranker | 1,082 | 23.17 | 2.583 |
| Hierarchical v5 reranker | 906 | 19.73 | 3.071 |

Относительно v5 cosine reranker увеличил total time на `49.4%`, снизил decode
throughput с `32.08` до `23.17 tok/s` и увеличил route overhead с `0.067` до
`0.223 s`.

Needle answer perplexity:

| Variant | Answer PPL | Exact match |
|---|---:|---:|
| Dense | 1.63 | нет |
| Full-scan cosine | 2.63 | нет |
| Full-scan v5 reranker | 34.81 | нет |
| Hierarchical v5 reranker | 13.72 | нет |

На этом тесте v5 существенно ухудшил retrieval конкретного факта.

#### Итог эксперимента

Neural reranking технически реализован, обучен на dense teacher и интегрирован
в full-scan/hierarchical routing. Однако текущие данные не подтверждают его
практическую пользу:

1. v4 ухудшил diagnostic Recall@16 с `51.74%` до `35.62%`;
2. v4 не улучшил PPL и добавил latency;
3. v5 немного улучшил PPL относительно block16 cosine baseline, но потерял
   почти половину total speed;
4. v5 ухудшил needle answer PPL с `2.63` до `34.81`;
5. обучение на одном Tiny Shakespeare не позволяет судить об обобщении.

Текущий production baseline — cosine routing без neural reranking. Neural
reranker имеет смысл переобучать только после перехода на разнородные данные,
hold-out validation и in-distribution Recall при том же `block_size`, что и в
inference. Иначе снижение teacher loss не является доказательством улучшения
маршрутизации.

## 9. Что уже работает

- ручная Pythia-1B архитектура на PyTorch;
- загрузка официальных весов;
- dense reference path;
- packed INT4 K/V cache;
- per-token K/V scales;
- local/sliding-window candidate path;
- cosine block routing;
- persistent multi-part K summaries;
- persistent V summaries;
- hierarchical summary tree;
- route refresh interval;
- full-scan и hierarchical routing modes;
- старые и новые reranker checkpoints;
- PPL, speed, Recall и needle benchmarks;
- speed-only bounded-cache запуск на 1M токенов.

## 10. Что пока не доказано

Нельзя утверждать, что:

1. модель является надёжной million-token language model;
2. Pythia сохраняет качество на 14K, 32K или 1M;
3. neural reranker лучше cosine-only routing;
4. hierarchical routing быстрее full scan на реальном runtime;
5. вся модель работает за O(1) или O(log N);
6. полный INT4-cache на 1M помещается вместе с весами и buffers на одной V100S;
7. комбинация методов является новой научной архитектурой без систематического
   сравнения с существующей литературой.

Научно корректная формулировка: selected attention kernel имеет фиксированный
budget при фиксированном route, hierarchical router имеет примерно
логарифмическую стоимость refresh, а полный INT4-cache требует O(N) памяти.

## 11. Основные файлы

| Файл | Назначение |
|---|---|
| [model.py](../backend/model.py) | Каноническая model-only Pythia/Ocean/INT4 реализация |
| [routing_ablation_benchmark.py](../backend/routing_ablation_benchmark.py) | Обучение reranker и controlled ablation |
| [final_long_context_benchmark.py](../backend/final_long_context_benchmark.py) | Финальный speed benchmark production cosine + full INT4 |
| [Pythia_1B_routing.ipynb](../notebooks/Pythia_1B_routing.ipynb) | PPL, speed, Recall и needle |
| [Pythia_1B_INT4_routed_train_benchmark.ipynb](../notebooks/Pythia_1B_INT4_routed_train_benchmark.ipynb) | Старые INT4 и training эксперименты |
| [README.md](../README.md) | Краткое описание и таблицы |
| [sublinear_attention_computation.md](./sublinear_attention_computation.md) | Подробный журнал алгоритма и экспериментов |
| [tinyshakespeare.txt](../tinyshakespeare.txt) | Локальный benchmark text |
| checkpoints/block-reranker.pt | Старый reranker, 500 steps |
| checkpoints/block-reranker-v2.pt | Улучшенный старый reranker |
| checkpoints/block-reranker-v3.pt | Multi-scale K/V/position reranker |
| checkpoints/block-reranker-v4-chunk-union.pt | Multi-scale reranker с chunk-level dense teacher |
| checkpoints/block-reranker-v5-b16.pt | Обученный reranker для block_size=16 |

Финальный production speed benchmark:

    ./.venv/bin/python backend/final_long_context_benchmark.py \
      --model-dir /home/froschin/.cache/huggingface/hub/models--EleutherAI--pythia-1b/snapshots/f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2 \
      --text-file ./tinyshakespeare.txt \
      --contexts 2048,14000,32000,100000,1000000 \
      --chunk-size 256 \
      --new-tokens 16 \
      --max-cache-gib 28 \
      --output ./final_long_context_cosine_int4.json

Этот тест фиксирует именно production baseline: `block_size=256`,
`route_blocks=16`, `local_window=256`, `route_refresh_interval=64`, полный
точный INT4 KV-cache и full-scan cosine routing. Контекст 1M пропускается при
memory guard, поскольку оценочный cache составляет около 31 GiB без весов и
временных буферов; это не является успешным запуском 1M.

### Финальный full INT4 benchmark с генерацией 1000 токенов

Обновлённый запуск использовал `new_tokens=1000` и успешно обработал контекст
до 500,000 токенов:

| Context | Prefill tok/s | Decode tok/s | Total s | Peak allocated |
|---:|---:|---:|---:|---:|
| 2,048 | 3,733 | 28.47 | 35.67 | 2.07 GiB |
| 14,000 | 4,268 | 23.33 | 46.15 | 2.53 GiB |
| 32,000 | 4,079 | 23.33 | 50.70 | 3.28 GiB |
| 100,000 | 3,992 | 23.45 | 67.70 | 5.65 GiB |
| 500,000 | 3,790 | 23.25 | 174.94 | 19.66 GiB |

Для 500K оценочный полный INT4 KV-cache составил `15.50 GiB`; запуск был
выполнен с полным хранением K/V для каждого входного токена. Decode throughput
на контекстах 14K–500K оставался в диапазоне `23.25–23.45 tok/s`.

Контекст 1M был пропущен memory guard: оценочный full INT4 KV-cache составляет
`30.99 GiB` без весов модели и временных буферов. Поэтому 1M-token запуск не
считался успешным результатом.

Это speed-only benchmark. PPL и retrieval quality на 500K/1M не измерялись.

## 12. Воспроизводимость

Benchmark:

    ./.venv/bin/python backend/routing_ablation_benchmark.py \
      --mode benchmark \
      --model-dir /home/froschin/.cache/huggingface/hub/models--EleutherAI--pythia-1b/snapshots/f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2 \
      --reranker-checkpoint ./checkpoints/block-reranker-v3.pt \
      --contexts 2048,14000,32768,100000 \
      --output ./routing_ablation_v3.json

Training:

    ./.venv/bin/python backend/routing_ablation_benchmark.py \
      --mode train-reranker \
      --model-dir /home/froschin/.cache/huggingface/hub/models--EleutherAI--pythia-1b/snapshots/f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2 \
      --text-file /home/froschin/work/llm/tinyshakespeare.txt \
      --reranker-output ./checkpoints/block-reranker-v4.pt \
      --reranker-steps 1500 \
      --reranker-sequences 32 \
      --reranker-query-stride 64 \
      --teacher-mode chunk_union \
      --teacher-chunk-size 256 \
      --teacher-local-window 256

GPU testing выполнялось на двух Tesla V100S-PCIE-32GB. Для честного timing
нужны warm-up, torch.cuda.synchronize(), одинаковые dtype, prompt,
new_tokens и очистка cache между моделями.

## 13. План дальнейшей работы

### Шаг 1. Исправить evaluation protocol

Это критично.

1. Обучать и тестировать reranker на одинаковом block size.
2. Для production test использовать block size 256.
3. Для Recall либо использовать отдельный checkpoint для block size 16, либо
   перенести Recall на контекст, где block size 256 даёт минимум 64 blocks.
4. Разделить train/validation/test documents.
5. Использовать несколько needle positions и random seeds.

Пока это не сделано, Recall v2/v3 нельзя считать строгим production comparison.

### Шаг 2. Сделать teacher targets соответствующими inference

Ранее при chunk-prefill route строился по query последнего токена chunk, а
teacher samples описывали одиночный query и включали блоки, которые в runtime
являются local или global mandatory blocks. Это было несоответствием протокола.
В `backend/routing_ablation_benchmark.py` теперь добавлен production-aligned
режим `teacher_mode=chunk_union`:

- route query — последний query текущего chunk;
- dense targets — средняя attention mass нескольких query внутри chunk;
- targets строятся только по complete semantic blocks до local window;
- local/global mandatory blocks не попадают в выбор reranker;
- старый `single_query` оставлен как контрольный режим.

Эквивалентно, labels собираются для всего chunk:

    Q_chunk -> normalized dense-important mass over all Q in chunk

Targets следует нормировать по semantic candidate blocks и исключать mandatory
local/global blocks, которые reranker не выбирает.

### Шаг 3. Переобучить reranker по исправленному протоколу

Сначала нужен отдельный checkpoint, обученный с `chunk_union`; старые v1/v2/v3
нельзя считать эквивалентными этому эксперименту:

    ./.venv/bin/python backend/routing_ablation_benchmark.py \
      --mode train-reranker \
      --model-dir /home/froschin/.cache/huggingface/hub/models--EleutherAI--pythia-1b/snapshots/f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2 \
      --text-file ./tinyshakespeare.txt \
      --reranker-output ./checkpoints/block-reranker-v4-chunk-union.pt \
      --reranker-steps 1500 \
      --reranker-sequences 32 \
      --reranker-query-stride 64 \
      --teacher-mode chunk_union \
      --teacher-chunk-size 256 \
      --teacher-local-window 256

Сравнение v4 выполнено. На текущем diagnostic v4 не показал роста Recall или
PPL при измеримой цене reranking latency, поэтому его следует убрать из
production path до переобучения на production `block_size=256`.

Обучение v4 выполнено успешно:

    steps: 1500
    teacher samples: 2560
    final total loss: 0.6474
    final listwise loss: 0.6474
    final ranking loss: 0.0

Это доказывает только оптимизацию loss на teacher samples. Это не доказывает
улучшение PPL, Recall или скорости на отложенном наборе.

Следующее сравнение на GPU:

    ./.venv/bin/python backend/routing_ablation_benchmark.py \
      --model-dir /home/froschin/.cache/huggingface/hub/models--EleutherAI--pythia-1b/snapshots/f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2 \
      --text-file ./tinyshakespeare.txt \
      --reranker-checkpoint ./checkpoints/block-reranker-v4-chunk-union.pt \
      --contexts 2048 \
      --chunk-size 256 \
      --new-tokens 16 \
      --output ./routing_ablation_v4_2048.json

Затем повторить ту же команду с `block-reranker-v3.pt` и сравнить JSON-файлы.
Для speed-only отдельно использовать контексты 14000, 32768 и 100000, поскольку
PPL за пределами native context 2048 не является честной оценкой качества.

### Шаг 4. Обучение на разнообразных данных

Tiny Shakespeare недостаточен. Добавить:

- PG-19;
- длинные английские книги;
- LongBench/RULER-style retrieval prompts;
- технические документы и код, если они входят в target domain.

Необходимо хранить train/validation/test split и не полагаться на циклически
повторяющиеся token IDs.

### Шаг 5. Оптимизация по ranking metrics

Измерять и оптимизировать:

    Recall@16
    Recall@route_budget
    NDCG по dense attention mass
    pairwise ranking accuracy
    routed logit error
    PPL
    latency

Hard negatives должны приходить из cosine Top-64: это блоки, которые дешёвый
selector считает похожими, но dense teacher считает менее важными.

### Шаг 6. Проверка пользы reranker

Обязательная таблица:

    cosine Top-16
    cosine Top-64 -> reranker Top-16
    hierarchical Top-64 -> reranker Top-16
    neural full scan Top-16
    dense oracle Top-16

Для каждого варианта нужны Recall, PPL, logit error, latency и end-to-end
speed. Если reranker не улучшает Recall при приемлемой latency, его следует
убрать из production path.

### Шаг 7. Runtime optimization

Текущие bottleneck’и:

- Python loops в hierarchy traversal;
- частые маленькие GPU operations;
- irregular gather из INT4 cache;
- отсутствие fused route/attention kernel;
- allocations и torch.cat;
- route для chunk по одному representative query.

Порядок:

1. preallocate route/index tensors;
2. векторизовать scoring по heads и blocks;
3. заменить Python tree traversal на tensorized levels;
4. профилировать dequantization и gather;
5. написать CUDA/Triton fused kernel;
6. после стабилизации повторно проверить torch.compile.

При torch.compile модель нужно компилировать после загрузки весов либо unwrap
compiled module: ранее compile добавлял префикс _orig_mod. и ломал strict
state-dict loading.

### Шаг 8. Long-context adaptation

После стабилизации routing:

1. RoPE scaling или другой position extension;
2. continued pretraining на sequence length 8192;
3. curriculum 16K/32K;
4. synthetic retrieval loss;
5. validation на PG-19, RULER, LongBench и needle tasks.

Модель, обученная на 2048, не начнёт надёжно понимать 32K только благодаря
INT4 и routing.

### Шаг 9. Multi-GPU и 1M

Для full INT4 1M cache потребуется изучить:

- tensor/model parallel;
- layer-wise или sequence-wise KV sharding;
- paged KV-cache;
- overlap communication/compute;
- benchmark на одной и нескольких GPU.

Сначала измерять prefill/decode speed и peak memory. PPL на 1M имеет смысл
только после long-context training и retrieval validation.

## 14. Итоговый статус

Проект достиг проверяемого состояния:

    Pythia-1B вручную реализована и загружает официальные веса.
    Полный KV-cache сжат до packed INT4.
    Routing ограничивает selected-attention workload.
    Local window и hierarchical summaries реализованы.
    Проверен bounded-cache speed-only запуск на 1M токенов.
    Полный exact INT4 KV-cache проверен на 500K токенах с генерацией 1000 токенов.
    Полный exact INT4 запуск на 1M не выполнен: memory guard оценил cache в 30.99 GiB.
    На native context routing сохраняет PPL близкий к dense.
    Neural reranker v3 реализован и обучен.
    Neural reranker v4 обучен на chunk_union teacher protocol.

Научно честный вывод:

    INT4 уменьшил memory constant.
    Routing дал bounded selected-attention workload.
    Near-dense PPL подтверждён только около native context.
    Для v4 выполнен 16K/32K speed benchmark и diagnostic Recall.
    Production Recall при block_size=256 пока не доказан.
    В текущем diagnostic v4 не улучшил cosine routing.
    Hierarchical routing пока проигрывает full scan по wall-clock.
    Speed на 500K подтверждён, но long-context PPL и retrieval quality не измерены.
    Long-context quality не доказана.

Ближайшая цель — исправить protocol: одинаковый block size, chunk-level teacher
targets, hold-out long documents и корректный Recall benchmark. После этого
можно переходить к long-context training и fused CUDA/Triton implementation.
