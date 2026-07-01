# SWAGA-окна на Qasper и BioASQ — runbook

Прогон benchmark-корпусов (Qasper, BioASQ) в конфигурации **`swaga_windows`** (связные
текстовые окна вокруг якорных чанков) вместо `swaga_chunks`, с получением метрик.

Оконная сборка реализована в самом пакете `swaga_rag` (`src/swaga_rag/rag/subgraph.py`) —
внешних компонентов для прогона не требуется.

## Что добавлено

- `src/swaga_rag/rag/subgraph.py` — `SubgraphAssembler`/`SubgraphConfig` + хелперы
  `windows_to_ranked_ids`, `window_output_items`, `window_chunk_id`. Порядок чанков
  внутри секции (`_chunk_order_key`) обобщён на схемы id docs / Qasper / BioASQ.
- `scripts/run_qasper_hybrid_retrieval.py`, `scripts/run_bioasq_hybrid_retrieval.py` —
  флаги `--windows` и `--windows-order {doc,anchors_first}`.
- `scripts/adapt_hybrid_predictions_for_eval.py` — пробрасывает `output_items` с
  метаданными окна (нужно для варианта C).
- `scripts/evaluate_windows_retrieval.py` — **новый** window-level evaluator (вариант C).
- `configs/qasper/swaga_windows.json`, `configs/ablations/bioasq_windows_baseline.json` —
  копии baseline-конфигов + блок `subgraph` (отличие только в оконной упаковке).

## Два варианта метрик

Один оконный прогон даёт данные для обоих вариантов сразу:

- **Вариант A (chunk-level, сравнимо с baseline/chunks).** `output_ids` = развёрнутые
  `window_node_ids` всех окон (окна по score, внутри окна — порядок документа, глобальный
  дедуп). Считается **существующими** evaluator'ами (`evaluate_retrieval_metrics.py` для
  Qasper, `evaluate_bioasq_retrieval.py` для BioASQ) — то же id-пространство и gold, что у
  BM25/Dense/`swaga_chunks`. Это цифры для общих таблиц.
- **Вариант C (window-level, диагностика).** Окно = единица; relevant, если его
  `window_node_ids` пересекаются с gold; Recall@k = покрытие gold объединением окон top-k.
  Считается `evaluate_windows_retrieval.py`. **Не** сравнимо напрямую с chunk-level baseline
  (другая единица измерения) — только для пары `swaga_windows` vs `swaga_chunks`.

`--windows-order doc` (нейтрально) или `anchors_first` (якоря держат ранг, ближе к
`swaga_chunks` по MRR) — влияет только на вариант A.

---

## 0. Предусловия

- Запускать из основного рабочего дерева, где лежат `datasets/` (в worktree их нет).
  Перенести этот код в основной checkout: `git checkout claude/vigilant-spence-6e2690`
  (или смержить ветку в свою рабочую).
- Окружение: `venv\Scripts\python.exe` (Python 3.12). Без GPU добавляйте `--device cpu`
  к сборке индексов и прогонам (по умолчанию у swaga-индексаторов `--device cuda`).
- **Индексы Qasper/BioASQ в репозитории отсутствуют** (в `.gitignore`) — их нужно
  построить заново (шаги 1.A / 2.A). Это самая тяжёлая часть.

---

## 1. Qasper

### 1.A. Построение индексов (если ещё нет)

```powershell
# граф из датасета
venv\Scripts\python.exe scripts\build_qasper_graph.py
# ontology/swaga индекс (эмбеддинги); без GPU: --device cpu
venv\Scripts\python.exe scripts\build_qasper_swaga_index.py --device cpu
# BM25 индекс по qasper-узлам (имя файла важно — его ждёт раннер)
venv\Scripts\python.exe scripts\build_bm25_index.py `
  --nodes-path data\processed\qasper_nodes.cleaned.json `
  --out-path artifacts\indexes\qasper_bm25_index.pkl
# queries/gold уже в data/eval/ (есть в репо); при необходимости пересобрать:
# venv\Scripts\python.exe scripts\build_qasper_validation_eval.py
```

### 1.B. Оконный прогон (вариант A + метаданные для C)

```powershell
venv\Scripts\python.exe scripts\run_qasper_hybrid_retrieval.py `
  --config configs\qasper\swaga_windows.json `
  --windows --windows-order doc `
  --device cpu `
  --out-dir artifacts\hybrid_rag_results\qasper_windows
```

(Для контрольного chunk-прогона — тот же скрипт без `--windows`, с
`--out-dir artifacts\hybrid_rag_results\qasper_chunks`.)

### 1.C. Метрики

```powershell
# Вариант A — существующий evaluator (strict + same_section)
venv\Scripts\python.exe scripts\evaluate_retrieval_metrics.py `
  --gold-path data\eval\qasper_validation_gold.jsonl `
  --run swaga_windows=artifacts\hybrid_rag_results\qasper_windows `
  --match-mode both `
  --out-json artifacts\reports\qasper_windows_metrics.json `
  --out-csv  artifacts\reports\qasper_windows_metrics.csv

# Вариант C — window-level evaluator
venv\Scripts\python.exe scripts\evaluate_windows_retrieval.py `
  --gold-path data\eval\qasper_validation_gold.jsonl `
  --run swaga_windows=artifacts\hybrid_rag_results\qasper_windows `
  --match-mode both `
  --out-json artifacts\reports\qasper_windows_metrics_windowlevel.json `
  --out-csv  artifacts\reports\qasper_windows_metrics_windowlevel.csv
```

---

## 2. BioASQ

### 2.A. Построение индексов (если ещё нет) — тяжёлый PMC-пайплайн (нужна сеть)

```powershell
venv\Scripts\python.exe scripts\prepare_bioasq_pmids.py
venv\Scripts\python.exe scripts\match_bioasq_pmcs.py
venv\Scripts\python.exe scripts\prepare_pmc_download_list.py
venv\Scripts\python.exe scripts\download_pmc_xml.py          # скачивает PMC XML
venv\Scripts\python.exe scripts\validate_pmc_xml.py
venv\Scripts\python.exe scripts\build_pmc_structured_corpus.py
venv\Scripts\python.exe scripts\build_bioasq_pmc_graph.py
venv\Scripts\python.exe scripts\build_bioasq_pmc_swaga_index.py --device cpu
venv\Scripts\python.exe scripts\build_bioasq_bm25_index.py
# gold (snippet-level): сопоставление сниппетов с чанками + сборка eval
venv\Scripts\python.exe scripts\match_bioasq_snippets_to_chunks.py
venv\Scripts\python.exe scripts\build_bioasq_retrieval_eval.py
```

Источник датасета — `datasets/bioasq/bioasq12b_eval.jsonl` (скрипты сами берут fallback-путь
`datasets/bioasq/...`, если нет `data/datasets/bioasq/...`).

### 2.B. Оконный прогон (вариант A + метаданные для C)

```powershell
venv\Scripts\python.exe scripts\run_bioasq_hybrid_retrieval.py `
  --config configs\ablations\bioasq_windows_baseline.json `
  --windows --windows-order doc `
  --device cpu `
  --output data\artifacts\bioasq_windows_predictions.jsonl
```

Раннер пишет один JSONL. Преобразуем в per-query файлы (метаданные окна сохраняются):

```powershell
venv\Scripts\python.exe scripts\adapt_hybrid_predictions_for_eval.py `
  --input data\artifacts\bioasq_windows_predictions.jsonl `
  --out-dir artifacts\swaga_rag_results\bioasq_windows
```

### 2.C. Метрики

```powershell
# Вариант A — существующий BioASQ evaluator (snippet-level, точное совпадение)
venv\Scripts\python.exe scripts\evaluate_bioasq_retrieval.py `
  --gold-path data\artifacts\bioasq_retrieval_eval.jsonl `
  --run-dir artifacts\swaga_rag_results\bioasq_windows `
  --out-json artifacts\reports\bioasq_windows_metrics.json `
  --out-csv  artifacts\reports\bioasq_windows_metrics.csv

# Вариант C — window-level evaluator
venv\Scripts\python.exe scripts\evaluate_windows_retrieval.py `
  --gold-path data\artifacts\bioasq_retrieval_eval.jsonl `
  --run bioasq_windows=artifacts\swaga_rag_results\bioasq_windows `
  --match-mode strict `
  --out-json artifacts\reports\bioasq_windows_metrics_windowlevel.json `
  --out-csv  artifacts\reports\bioasq_windows_metrics_windowlevel.csv
```

---

## Параметры окна

Блок `subgraph` в конфиге (значения по умолчанию совпадают с конфигурацией окон для
пользовательской документации):

```json
"subgraph": { "tail_after": 2, "max_window_chunks": 8, "fallback_before": 2, "fallback_after": 2 }
```

- `tail_after` — сколько чанков после якоря добавлять (окно по умолчанию идёт от начала секции до якоря + хвост);
- `max_window_chunks` — порог, при превышении окно сжимается до симметричного вокруг якоря;
- `fallback_before` / `fallback_after` — размеры симметричного окна при срабатывании порога.
