# Результаты экспериментальной оценки

Лёгкие, версионируемые артефакты результатов (метрики, отчёты судей) — чтобы
результаты статьи можно было просмотреть прямо в репозитории, без пересборки
тяжёлых индексов и без повторного запуска LLM-судей. Крупные артефакты (индексы,
кэши моделей, датасеты, сырые предсказания и пофайловые прогоны) намеренно не
версионируются — см. `.gitignore` и раздел «Воспроизводимость» в корневом README
и `docs/`.

## Состав

```
results/
├── retrieval/                     retrieval-метрики (Recall@k, MRR, nDCG, noise)
│   ├── summary.md                 СВОДНАЯ таблица: корпус × энкодер × first-stage × вариант
│   ├── qasper_encoders.{csv,json}         Qasper: mpnet / bge / e5 (chunks)
│   ├── qasper_fair_e5.{csv,json}          Qasper: BM25 / dense(e5) / hybrid(e5)
│   ├── qasper_firststage_sweep.{csv,json} Qasper: top-m {5,10,20} × {bm25,dense}-first
│   ├── qasper_canon_chunks_vs_windows.*   Qasper: chunks vs windows (каноничная конфиг.)
│   ├── qasper_windows_by_encoder.*        Qasper: chunks vs windows × mpnet/bge/e5 (вариант A)
│   ├── {qasper,bioasq}_windowlevel.*      окно как единица выдачи (вариант C) × энкодеры
│   ├── qasper_threshold_axis.{csv,json}   Qasper: absolute/percentile/rank пороги
│   └── bioasq_enc_{mpnet,pubmedbert,bge}.*  BioASQ: по энкодерам (snippet-level)
└── llm_judge/                     парная LLM-оценка (корпус документации, 150 запросов)
    ├── report.md                  Таблицы 7–9: win/tie/loss, согласованность, стоимость
    ├── pair_aggregates.json       по сравнениям × осям (≥4/6, инверсия перестановки)
    ├── agreement_metrics.json     inter-judge agreement + permutation consistency
    ├── operational_metrics.json   вызовы/токены/стоимость
    ├── calibration_agreement.json согласие судей с экспертной разметкой (5 запросов)
    ├── expert_labels.jsonl        экспертная разметка (перепривязанная), 90 пар
    └── csv/                       те же таблицы в CSV
```

## Retrieval — главное (`retrieval/summary.md`)

Метрики изолированного этапа retrieval, единый token budget, без reranking. `dense`
= векторный baseline на **том же энкодере**, что SWAGA-индекс (честное сравнение).
`hybrid` = first-stage doc-recall (BM25 или dense) + SWAGA-локализация внутри.

- **Энкодер — главный рычаг**, и он поднимает все методы: Qasper hybrid-chunks
  strict R@10 mpnet 0.084 → bge 0.094 → **e5 0.108**; BioASQ R@10 mpnet 0.237 →
  pubmedbert 0.259 → **bge 0.294** (общий bge обходит доменный PubMedBert).
- **На сильном энкодере dense — сильный baseline**: dense·e5 (Qasper) R@10 **0.134**,
  dense·bge (BioASQ) R@10 **0.362** — выше SWAGA-hybrid по recall (у гибрида recall
  ограничен doc-restriction первой стадии).
- **Ценность SWAGA — точность ранжирования, не recall**: Qasper same_section nDCG
  hybrid·e5 **0.201** (окна 0.221) против dense 0.148 / BM25 0.138; BioASQ MRR
  hybrid·bge до **0.540** (dense-first) против dense 0.430.
- **First-stage корпусно-зависим** (принцип, не подбор максимума): single-doc
  Qasper → BM25-first (шире → хуже); multi-doc BioASQ → dense-first (шире → лучше).
- **Окна — эффект стабилен по всем энкодерам** (не артефакт одной модели): в chunk-режиме
  (вариант A) нейтральны по strict recall и стабильно поднимают Qasper same_section nDCG
  (+0.012…0.024 для mpnet/bge/e5); как единица выдачи (вариант C) поднимают раннюю
  релевантность — BioASQ MRR·bge **0.595** против 0.527 chunk-level, noise падает с силой
  энкодера (0.69→0.62). Основной выигрыш окон — в качестве контекста (LLM-судья, 69–90% побед).
- **Пороги drill-down** (absolute/percentile/rank) практически инертны на
  каноничной конфигурации → выдача устойчива к параметризации.

## LLM-судья — главное (`llm_judge/report.md`)

Корпус пользовательской документации, 150 запросов, парный протокол (3 судьи ×
2 перестановки, мажоритарное решение ≥4/6), три оси: релевантность, чистота,
достаточность. Панель судей — CometAPI (DeepSeek-V4-Pro, Qwen3.5-Plus,
Gemini-3.1-Flash-Lite), откалибрована по экспертной разметке (согласие ~0.62–0.65).
Прогон **чистый — сняты два конфаунда**: (1) `swaga_chunks` и `swaga_windows` идут на
идентичном конфиге drill/expand/score (проверено ассертом), различие только в
упаковке (окна вкл/выкл); (2) все embedding-методы — на одном энкодере
**mpnet** (`dense` пересобран на тех же 1696 chunk-id, сменён только энкодер).
Значения — доля побед SWAGA / tie / поражений.

| Сравнение | win SWAGA | tie | win baseline |
|---|--:|--:|--:|
| swaga_windows vs BM25 | **0.73** | 0.07 | 0.20 |
| swaga_windows vs dense | **0.82** | 0.04 | 0.14 |
| swaga_windows vs BM25+heuristic | **0.69** | 0.07 | 0.24 |
| swaga_windows vs dense+heuristic | **0.70** | 0.07 | 0.23 |
| swaga_windows vs swaga_chunks | **0.90** | 0.10 | 0.00 |

Оконная конфигурация устойчиво превосходит все baseline и chunks (по чистоте
81–95% побед). При **идентичном** retrieval и одном энкодере выигрыш `swaga_windows`
над `swaga_chunks` достигает **90% / 10% / 0%** — то есть его даёт именно оконная
сборка контекста, а не параметры или модель эмбеддингов. Сам `swaga_chunks` без
окон уступает baseline'ам по relevance/sufficiency, но выигрывает по чистоте (против
`dense·mpnet` — паритет overall 44/15/41). Согласованность: inter-judge 0.73,
permutation consistency 0.65–0.87. Фактическая стоимость прогона — $7.96.

## Итоговая трактовка

SWAGA-RAG — **precision/локализующий компонент поверх сильной первой стадии** (dense
для multi-document, BM25 для single-document): выигрыш в ранжирующей точности
(MRR / same-section nDCG) и в качестве контекста для чтения/генерации (LLM-судья),
а не в «сыром» recall, где сильный dense-энкодер уже лидирует.

Как воспроизвести числа — см. `../REPRODUCE.md` и `../docs/windows_benchmarks.md`.
