# LLM

## Базовая архитектура

```mermaid
graph TD
    A["input_ids (B, T)"] --> B[token_embedding]
    B --> C[pos_embedding]
    C --> D[embedding_dropout]

    D --> E["blocks: DecoderBlock x N"]
    E --> F[final_norm]
    F --> G[lm_head]
    G --> H["logits (B, T, vocab_size)"]

    subgraph DecoderBlock ["DecoderBlock"]
        I[ln_attn] --> J[CausalSelfAttention]
        J --> K[ln_ffn]
        K --> L[ffn]
    end

    subgraph CausalSelfAttention ["CausalSelfAttention"]
        M[q_proj] --> Q
        N[k_proj] --> Q
        O[v_proj] --> Q
        Q["scaled_dot_product_attention"] --> R[out_proj]
    end

    subgraph ffn ["FFN"]
        S[linear1] --> T[GELU]
        T --> U[linear2]
        U --> V[Dropout]
    end

    %% Связи внутри блока
    I -->|x| J
    J -->|attn_out| I_skip["+"]
    I_skip --> K
    K -->|ffn_out| K_skip["+"]
    K_skip --> output_block["output"]

    %% Опционально: указать, что линейные слои могут быть TernaryLinear
    classDef ternary fill:#f9f,stroke:#333,stroke-width:2px
    class M,N,O,R,S,U ternary
```

## Оптимизированная Pythia-1B с иерархическим Ocean attention

Ниже показана фактическая streaming-архитектура, реализованная в `Pythia_1B.ipynb`. Модель сохраняет 16 decoder-слоёв Pythia-1B и исходные веса, но заменяет полное causal attention на query-dependent маршрутизацию по блокам KV-кэша.

```mermaid
flowchart TD
    A["Входной токен x_t"] --> B["Token embedding<br/>vocab=50304, d_model=2048"]
    B --> C["Decoder layers × 16"]
    C --> D["Final LayerNorm"]
    D --> E["LM head<br/>2048 → 50304"]
    E --> F["Logits"]

    subgraph Layer["Один decoder layer — повторяется 16 раз"]
        L1["Input LayerNorm"] --> L2["SublinearOceanAttention"]
        L1 --> L3["Parallel residual"]
        L2 --> L3
        L3 --> L4["Post-attention LayerNorm"]
        L4 --> L5["MLP: 2048 → 8192 → 2048<br/>GELU"]
        L5 --> L6["Parallel residual output"]
        L3 --> L6
    end

    subgraph Attention["SublinearOceanAttention — один шаг q_len=1"]
        P["QKV projection<br/>2048 → 6144"] --> R["RoPE<br/>25% head dimensions"]
        R --> Q["Query q_t"]
        R --> K["Новые K/V"]
        K --> Cache["Preallocated KV cache<br/>O(N) memory"]
        K --> Tree["Incremental summary tree<br/>block_size=64<br/>summary_parts=4"]
        Q --> Route["Hierarchical route<br/>beam_width=32"]
        Tree --> Route
        Route --> Selected["Выбрать ≤ 16 блоков<br/>global_blocks=1<br/>local_blocks=2"]
        Selected --> Gather["Gather выбранных K/V<br/>≤ 16 × 64 = 1024 токена"]
        Cache --> Gather
        Q --> SDPA["Causal SDPA<br/>только по выбранным токенам"]
        Gather --> SDPA
        SDPA --> Out["Output projection<br/>2048 → 2048"]
    end

    C -. каждый layer содержит .-> Layer
    L2 -. реализовано как .-> Attention
```

### Конфигурация оптимизированной модели

| Параметр | Значение | Назначение |
|---|---:|---|
| Decoder layers | 16 | Число transformer-блоков Pythia-1B |
| Hidden size | 2048 | Размер скрытого представления |
| Attention heads | 8 | Число голов внимания |
| Head dimension | 256 | `2048 / 8` |
| `block_size` | 64 токена | Размер одного маршрутизируемого блока |
| `route_blocks` | 16 максимум | Сколько блоков может выбрать attention на шаге |
| `beam_width` | 32 | Ширина иерархического поиска маршрута |
| `summary_parts` | 4 | Число summary-представлений блока |
| `global_blocks` | 1 | Принудительно доступный глобальный блок |
| `local_blocks` | 2 | Принудительно доступные последние локальные блоки |
| Максимум выбранных токенов | 1024 | `16 × 64`, вместо всего префикса |
| Штатный контекст обучения | 2048 токенов | Контекст Pythia-1B, 14K — только стресс-тест |

Здесь «блоки, используемые для внимания» — это `route_blocks=16`: на каждом streaming-шаге attention работает максимум с 16 блоками, а не со всеми блоками префикса. Значение 5 было бы другим, более агрессивным режимом маршрутизации; в текущем экспериментальном конфиге используется 16.

### Асимптотика

При фиксированных `block_size=64`, `route_blocks=16`, `beam_width=32` и `summary_parts=4`:

- KV-кэш растёт как `O(N)` по длине контекста.
- Иерархические summary обновляются за `O(log N)` на новый токен.
- Маршрутизация занимает примерно `O(log N · beam_width · head_dim)` на шаг.
- Само выбранное attention занимает `O(route_blocks · block_size · head_dim)`, то есть `O(1)` по `N` при фиксированных параметрах.
- При фиксированных параметрах маршрутизации один streaming-шаг attention имеет порядок `O(log N)` для обновления дерева и поиска маршрута плюс константную работу по выбранным токенам; KV-кэш при этом занимает `O(N)` памяти.
- Для генерации последовательности длины `N` суммарная стоимость маршрутизации составляет примерно `O(N log N)`, тогда как у обычного incremental dense attention она составляет `O(N²)` из-за просмотра всего префикса на каждом новом токене.

Следовательно, это не безусловно «сублинейная модель» во всех смыслах: точнее говорить о sublinear routing и ограниченном attention-workload при линейной памяти KV-кэша. Качество маршрутизации подтверждено текущими экспериментами на Shakespeare, но не является доказательством эквивалентности dense attention на общих данных.
