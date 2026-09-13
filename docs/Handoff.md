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
[model.py](./model.py).

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

### 6.5. Качество на длинных контекстах

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
[Pythia_1B_routing.ipynb](./Pythia_1B_routing.ipynb) и
[routing_ablation_benchmark.py](./routing_ablation_benchmark.py).

Сравнивались:

    dense
    full_scan_cosine
    hierarchical_cosine
    full_scan_reranker
    hierarchical_reranker
    neural_full_scan

neural_full_scan — learned selector без cosine prefilter.

### PPL на 2048

Актуальный v3 PPL benchmark находится в
[routing_ablation_v3_ppl.json](./routing_ablation_v3_ppl.json):

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
| [model.py](./model.py) | Каноническая model-only Pythia/Ocean/INT4 реализация |
| [routing_ablation_benchmark.py](./routing_ablation_benchmark.py) | Обучение reranker и controlled ablation |
| [Pythia_1B_routing.ipynb](./Pythia_1B_routing.ipynb) | PPL, speed, Recall и needle |
| [Pythia_1B_INT4_routed_train_benchmark.ipynb](./Pythia_1B_INT4_routed_train_benchmark.ipynb) | Старые INT4 и training эксперименты |
| [README.md](./README.md) | Краткое описание и таблицы |
| [sublinear_attention_computation.md](./sublinear_attention_computation.md) | Подробный журнал алгоритма и экспериментов |
| [tinyshakespeare.txt](./tinyshakespeare.txt) | Локальный benchmark text |
| checkpoints/block-reranker.pt | Старый reranker, 500 steps |
| checkpoints/block-reranker-v2.pt | Улучшенный старый reranker |
| checkpoints/block-reranker-v3.pt | Multi-scale K/V/position reranker |

## 12. Воспроизводимость

Benchmark:

    ./.venv/bin/python routing_ablation_benchmark.py \
      --mode benchmark \
      --model-dir /home/froschin/.cache/huggingface/hub/models--EleutherAI--pythia-1b/snapshots/f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2 \
      --reranker-checkpoint ./checkpoints/block-reranker-v3.pt \
      --contexts 2048,14000,32768,100000 \
      --output ./routing_ablation_v3.json

Training:

    ./.venv/bin/python routing_ablation_benchmark.py \
      --mode train-reranker \
      --model-dir /home/froschin/.cache/huggingface/hub/models--EleutherAI--pythia-1b/snapshots/f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2 \
      --text-file /home/froschin/work/llm/tinyshakespeare.txt \
      --reranker-output ./checkpoints/block-reranker-v4.pt \
      --reranker-steps 1500 \
      --reranker-sequences 32 \
      --reranker-query-stride 64

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

Сейчас при chunk-prefill route строится по query последнего токена chunk, а
teacher samples в основном описывают одиночный query. Нужно собирать labels
для всего chunk:

    Q_chunk -> union/top-k dense-important blocks for all Q in chunk

Targets следует нормировать по semantic candidate blocks и исключать mandatory
local/global blocks, которые reranker не выбирает.

### Шаг 3. Обучение на разнообразных данных

Tiny Shakespeare недостаточен. Добавить:

- PG-19;
- длинные английские книги;
- LongBench/RULER-style retrieval prompts;
- технические документы и код, если они входят в target domain.

Необходимо хранить train/validation/test split и не полагаться на циклически
повторяющиеся token IDs.

### Шаг 4. Оптимизация по ranking metrics

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

### Шаг 5. Проверка пользы reranker

Обязательная таблица:

    cosine Top-16
    cosine Top-64 -> reranker Top-16
    hierarchical Top-64 -> reranker Top-16
    neural full scan Top-16
    dense oracle Top-16

Для каждого варианта нужны Recall, PPL, logit error, latency и end-to-end
speed. Если reranker не улучшает Recall при приемлемой latency, его следует
убрать из production path.

### Шаг 6. Runtime optimization

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

### Шаг 7. Long-context adaptation

После стабилизации routing:

1. RoPE scaling или другой position extension;
2. continued pretraining на sequence length 8192;
3. curriculum 16K/32K;
4. synthetic retrieval loss;
5. validation на PG-19, RULER, LongBench и needle tasks.

Модель, обученная на 2048, не начнёт надёжно понимать 32K только благодаря
INT4 и routing.

### Шаг 8. Multi-GPU и 1M

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
    На native context routing сохраняет PPL близкий к dense.
    Neural reranker v3 реализован и обучен.

Научно честный вывод:

    INT4 уменьшил memory constant.
    Routing дал bounded selected-attention workload.
    Near-dense PPL подтверждён только около native context.
    Neural reranker пока не улучшил cosine routing.
    Hierarchical routing пока проигрывает full scan по wall-clock.
    Long-context quality не доказана.

Ближайшая цель — исправить protocol: одинаковый block size, chunk-level teacher
targets, hold-out long documents и корректный Recall benchmark. После этого
можно переходить к long-context training и fused CUDA/Triton implementation.
