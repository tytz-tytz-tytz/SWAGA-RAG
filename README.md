<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="swaga-rag.horizontal-white.png">
    <img src="swaga-rag.horizontal-black.png" alt="swaga_rag" width="420">
  </picture>
</p>

# SWAGA-RAG

Structured Walk with Adaptive Granularity Approach for RAG — метод структурно-ориентированного извлечения контекста, в котором retrieval выполняется с учётом иерархии разделов и связей между элементами документа.

Репозиторий содержит исходный код метода, реализации baseline-подходов и воспроизводимый экспериментальный пайплайн, на результаты которого ссылается текст статьи. Это **исследовательский репозиторий метода и его экспериментальной оценки** — он самодостаточен и не требует внешних компонентов для воспроизведения результатов.

## Результаты и воспроизводимость

- **Готовые результаты** (лёгкие, версионируемые) — в [`results/`](results/): retrieval-метрики
  (BM25 / dense / SWAGA-chunks / windows по корпусам и энкодерам, first-stage и
  threshold-оси) и парная LLM-оценка (150 запросов: win/tie/loss, согласованность
  судей, калибровка). Сводка и трактовка — [`results/README.md`](results/README.md).
- **Как воспроизвести** — [`REPRODUCE.md`](REPRODUCE.md) и [`docs/windows_benchmarks.md`](docs/windows_benchmarks.md).
- Оценка расширена относительно первоначальной: **сменные энкодеры** (mpnet / bge /
  e5 / S-PubMedBert с query/passage-префиксами), **честный dense-baseline на том же
  энкодере**, **dense/BM25 first-stage** и **percentile/rank пороги** drill-down,
  обновлённая панель LLM-судей (CometAPI). Тяжёлые артефакты (индексы, кэши моделей,
  датасеты, пофайловые прогоны) не версионируются — см. `.gitignore`.

---

## О методе

Классические retrieval-подходы (BM25, dense retrieval) рассматривают корпус как множество независимых текстовых фрагментов, а распространённые графовые подходы оперируют семантическими сущностями, извлечёнными из текста, а не иерархической авторской структурой самого документа. SWAGA-RAG использует структурно-графовую модель документа как основу управляемого процесса поиска.

На вход подаются пользовательский запрос и структурно-графовое представление документации; результатом является набор текстовых узлов, формирующих retrieval-контекст для последующей генерации или оценки. Метод реализует **структурный обход документа (structured walk) с адаптивным изменением гранулярности** рассматриваемых элементов: сначала выполняется локализация релевантной области на уровне секций, затем — структурно-семантическое ранжирование текстовых узлов внутри неё. Тем самым область поиска ограничивается структурно связной частью документа, что снижает долю нерелевантных фрагментов в итоговом контексте.

Процедурно метод состоит из этапов:

1. построение семантических представлений секций;
2. выбор релевантных корневых ветвей документа;
3. адаптивный drill-down по иерархии секций с выбором seed-секций;
4. построение локального структурного подграфа вокруг seed-секций;
5. структурно-семантический скоринг текстовых узлов (семантическая близость + структурное положение + графовая дистанция);
6. формирование связных текстовых окон вокруг наиболее релевантных узлов как итоговой формы выдачи.

Метод рассматривается в двух конфигурациях:

- **SWAGA-окна** (`swaga_windows`) — финальная конфигурация: отобранные узлы упаковываются в связные текстовые окна по структурным границам секций;
- **SWAGA-чанки** (`swaga_chunks`) — упрощённая конфигурация без оконного расширения: выдача формируется как ранжированный список якорных узлов.

Реализация ориентирована на управляемый и воспроизводимый retrieval-пайплайн, а не на онлайн-обслуживание или end-to-end QA-агента.

---

## Экспериментальная оценка

Внимание сосредоточено на изолированном анализе этапа retrieval. Все методы реализованы в едином пайплайне и сравниваются при одинаковых условиях: единый набор запросов на корпус, единое ограничение на объём выдачи, **без дополнительного reranking** (осознанное методологическое решение — чтобы не смешивать вклад retrieval с последующей переоценкой кандидатов).

### Корпуса и режимы retrieval

| Корпус | Режим retrieval | Конфигурация SWAGA | Разметка | Протокол оценки |
|--------|-----------------|--------------------|----------|-----------------|
| Пользовательская документация (150 запросов) | retrieval в иерархически структурированной документации | `swaga_windows` + `swaga_chunks` (единый конфиг) | — | парный LLM-as-judge |
| Qasper (888 запросов) | single-document, слабая иерархия | `swaga_chunks` + `swaga_windows`, гибрид с first-stage | paragraph-level | Recall@k / MRR / nDCG |
| BioASQ (280 вопросов) | multi-document (отбор документов + локализация) | `swaga_chunks` + `swaga_windows`, гибрид `{BM25,dense} + SWAGA-RAG` | snippet-level | Recall@k / MRR / nDCG / context noise |

### Сравниваемые методы

- **BM25** — лексический retrieval; **Dense** — векторный retrieval по семантическим эмбеддингам;
- **BM25 + heuristic**, **Dense + heuristic** — over-retrieval с эвристическим расширением окрестности, фильтрацией и дедупликацией (структуру документа явно не используют);
- **SWAGA-RAG** (`swaga_windows` / `swaga_chunks`) — основной метод;
- **Hybrid (BM25 + SWAGA-RAG)** — BM25 для глобального отбора документов, SWAGA-RAG для структурной локализации внутри них (сценарий BioASQ/Qasper).

### Протокол формирования контекста

Выдача каждого метода ограничивается единым token budget = **2000 токенов** по токенизации `cl100k_base` (tiktoken). Фрагменты добавляются последовательно в порядке ранжирования; не помещающийся целиком фрагмент не включается (усечение посередине не применяется). Внутренняя структура выдачи (число и размер фрагментов) определяется самим методом и не унифицируется — сопоставимость обеспечивается общим token budget.

### Протокол оценки

**Пользовательская документация** — парные сравнения LLM-as-judge по трём осям: *релевантность*, *чистота*, *достаточность*. Контроль смещений: обе перестановки предъявления (нейтрализация position bias), фиксированный seed порядка, **три независимых судьи из разных семейств** (DeepSeek-V4-Pro, Qwen3.5-Plus, Gemini-3.1-Flash-Lite через шлюз CometAPI; температура 0, reasoning off). Итог по паре — большинство голосов (≥4 из 6: 3 судьи × 2 перестановки, с инверсией BA-перестановки).

**Benchmark-корпуса** — стандартные retrieval-метрики Recall@k, MRR, nDCG@k (для BioASQ дополнительно context noise@k). Результаты на трёх корпусах не агрегируются в единую метрику из-за различий в задачах и разметке.

### Ключевые результаты

**Пользовательская документация** (150 запросов; доли win SWAGA / tie / win baseline, overall = усреднение по трём осям). Прогон изолирует ровно один фактор: `swaga_chunks` и `swaga_windows` считаются на **идентичном** конфиге drill/expand/score (различие только в оконной упаковке), а все embedding-методы — на одном энкодере (mpnet), `dense` пересобран на тех же чанках со сменой лишь энкодера; BM25 — лексический.

| Сравнение | win SWAGA | tie | win baseline |
|-----------|:---------:|:---:|:------------:|
| `swaga_windows` vs BM25 | 0.73 | 0.07 | 0.20 |
| `swaga_windows` vs Dense | 0.82 | 0.04 | 0.14 |
| `swaga_windows` vs BM25 + heuristic | 0.69 | 0.07 | 0.24 |
| `swaga_windows` vs Dense + heuristic | 0.70 | 0.07 | 0.23 |

Финальная конфигурация устойчиво превосходит и классические baseline (BM25, Dense), и их эвристически усиленные варианты — без ручной подстройки эвристик под корпус.

**Вклад этапа формирования окон** (`swaga_windows` vs `swaga_chunks` при идентичном retrieval): **90% побед против 0%** (tie 10%) в overall, с наиболее выраженным эффектом по чистоте (94% против 1%). Так как единственное различие между конфигурациями — оконная упаковка, этот выигрыш целиком относится к ней: окна не добавляют новой информации, но передают отобранные узлы в форме, согласованной с локальной логикой документа. Согласованность судей: inter-judge 0.73, permutation consistency 0.65–0.87.

**Энкодер — главный рычаг на бенчмарках, и он поднимает все методы**: Qasper hybrid-chunks strict Recall@10 mpnet 0.084 → bge 0.094 → **e5 0.108**; BioASQ Recall@10 mpnet 0.237 → pubmedbert 0.259 → **bge 0.294**. Поэтому dense-baseline считается на **том же энкодере**, что SWAGA-индекс (иначе «прирост» смешивается со сменой модели).

**Qasper** (single-document, слабая иерархия): на сильном энкодере dense — сильный baseline по recall (dense·e5 Recall@10 **0.134**), но SWAGA полезен для **уточнённого ранжирования** внутри найденной области — same_section nDCG@10 hybrid·e5 **0.201** (окна **0.221**) против dense 0.148 / BM25 0.138.

**BioASQ** (multi-document): dense·bge даёт лучший recall (Recall@10 **0.362**), а SWAGA-гибрид — лучшую раннюю точность: MRR hybrid·bge до **0.540** против dense 0.430. First-stage корпусно-зависим по принципу: single-doc Qasper → BM25-first, multi-doc BioASQ → dense-first.

**Абляции** (16 конфигураций, диагностика overlap@20 / Jaccard@20): большинство параметров влияют на выдачу лишь количественно; режимную перестройку дают отключение структурной локализации (drill-down) и графовой дистанции — это несущие компоненты метода.

**Итог трактовки**: SWAGA-RAG — precision/локализующий компонент поверх сильной первой стадии. Его вклад — не «сырой» recall (там лидирует сильный dense-энкодер), а точность ранжирования (MRR / same-section nDCG) и качество контекста для чтения/генерации (LLM-судья). Полные таблицы по энкодерам, first-stage и окнам — в [`results/`](results/).

---

## Структура репозитория

```text
SWAGA-RAG/
├── src/
│   ├── swaga_rag/        ядро метода: drill-down, expand, score, pipeline, индексация
│   ├── bm25_rag/         лексический baseline (+ эвристический вариант)
│   ├── classic_rag/      dense baseline (+ эвристический вариант)
│   ├── judge_prep/       подготовка payload/промптов для слепой LLM-оценки
│   └── rag_common/       общие утилиты baseline-ов (text hygiene)
├── scripts/              построение индексов, прогоны retrieval, абляции, LLM-судьи
├── configs/
│   ├── ablations/        конфигурации абляционного анализа (stable_*)
│   ├── qasper/           конфигурации SWAGA-RAG для Qasper
│   ├── judge_prep/       конфигурации подготовки судейских пар
│   └── judge_v2/         судьи, методы, пары для парного протокола
├── data/
│   ├── eval/             наборы запросов и gold-разметка (tracked)
│   ├── raw/              исходный граф документации (tracked)
│   └── processed/        очищенный граф для воспроизводимости baseline (tracked)
├── tests/                юнит- и интеграционные тесты ядра и раннеров
├── artifacts/            индексы и результаты прогонов (в .gitignore)
├── pyproject.toml
└── README.md
```

---

## Установка

Требуется Python `3.12`.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_env.ps1
```

Скрипт создаёт виртуальное окружение и устанавливает пакет в editable-режиме. Запускать скрипты следует интерпретатором окружения (`venv\Scripts\python.exe`). Пути по умолчанию разрешаются относительно корня репозитория.

LLM-судьи требуют ключа `COMETAPI_KEY` (см. `.env.example`; можно задать и как переменную окружения). Без ключа доступны только retrieval-прогоны и benchmark-метрики; парная LLM-оценка не выполняется.

---

## Запуск

### 1. Построение офлайн-индексов

```powershell
venv\Scripts\python.exe scripts\build_bm25_index.py
venv\Scripts\python.exe scripts\build_classic_rag_index.py
venv\Scripts\python.exe scripts\build_swaga_index.py
```

### 2. Прогоны retrieval

```powershell
venv\Scripts\python.exe scripts\run_queries_bm25.py
venv\Scripts\python.exe scripts\run_queries_classic_rag.py
venv\Scripts\python.exe scripts\run_queries_swaga.py --config configs\ablations\stable_baseline.json
```

### 3. Абляционные прогоны

```powershell
venv\Scripts\python.exe scripts\run_param_experiments.py --configs_dir configs\ablations
```

### 4. Парная LLM-оценка (judge_v2)

Чистый конфиг без конфаундов — `configs/judge_v2_clean/` (единый drill/expand/score для `swaga_chunks`/`swaga_windows`, dense на mpnet). Унификация под token budget → сборка пар → судейство → агрегация:

```powershell
venv\Scripts\python.exe scripts\judge_v2_unify.py      --methods_config configs\judge_v2_clean\methods.json --out_dir artifacts\judge_v2_clean\unified
venv\Scripts\python.exe scripts\judge_v2_make_pairs.py --methods_config configs\judge_v2_clean\methods.json --unified_dir artifacts\judge_v2_clean\unified --out_path artifacts\judge_v2_clean\pairs_full150.jsonl
$env:COMETAPI_KEY="<KEY>"
venv\Scripts\python.exe scripts\judge_v2_full_run.py   --pairs artifacts\judge_v2_clean\pairs_full150.jsonl --out_dir artifacts\judge_v2_clean\full_run
venv\Scripts\python.exe scripts\judge_v2_aggregate.py  --pairs artifacts\judge_v2_clean\pairs_full150.jsonl --full_dir artifacts\judge_v2_clean\full_run --out_dir artifacts\judge_v2_clean\agg
```

Полный runbook (включая ретрив-прогоны 6 методов) — в [`REPRODUCE.md`](REPRODUCE.md).

### Benchmark-корпуса

Прогоны и метрики для Qasper и BioASQ выполняются профильными скриптами (`run_qasper_hybrid_retrieval.py`, `run_bioasq_hybrid_retrieval.py`, `evaluate_retrieval_metrics.py`, `evaluate_bioasq_retrieval.py`).

### Тесты

```powershell
venv\Scripts\python.exe -m pytest -q
```

---

## Воспроизводимость

- Версия Python — ровно `3.12`; все скрипты запускаются через `venv\Scripts\python.exe`.
- `configs/ablations/stable_baseline.json` остаётся неизменным для baseline-прогонов.
- Каждый прогон хранится со снимком своего `config.json`; выдачи разных методов не смешиваются в одной папке.
- LLM-оценка ведётся слепой (имена методов скрыты в payload/промптах); для каждой таблицы работы фиксируется commit-хэш.
- Крупные артефакты не версионируются в Git: `artifacts/`, локальные окружения, кэши, тяжёлые `datasets/` и `data/*`. Лёгкие входы оценки под `data/eval/` и графовые файлы под `data/raw/` и `data/processed/` — версионируются.

Доказательная база состоит из артефактов, выведенных из публичной пользовательской документации. Эталонные тест-планы команды тестирования (NDA) в репозиторий не включены.

---

## Замечание об именовании

SWAGA-RAG — текущее название метода, ранее реализованного как OntologyRAG. Код мигрирован на новое именование (`swaga_rag`, `run_queries_swaga.py`, `build_swaga_index.py`).

---

## Лицензия

MIT. См. файл [`LICENSE`](LICENSE).
