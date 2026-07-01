# Воспроизводимость

Готовые результаты (метрики, отчёты судей) лежат в [`results/`](results/) и не
требуют пересборки. Ниже — как получить их заново. Тяжёлые артефакты (индексы,
кэши моделей HF, датасеты, пофайловые прогоны) в репозиторий не входят
(`.gitignore`) — они детерминированно строятся приведёнными скриптами.

Требуется Python 3.12. Установка окружения:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_env.ps1
```
Все скрипты запускать интерпретатором окружения (`venv\Scripts\python.exe`); пути
разрешаются от корня репозитория.

## 1. Корпус документации + парная LLM-оценка (150 запросов)

Индексы корпуса (BM25 / dense / SWAGA) из версионируемого графа
`data/processed/graphrag_*.cleaned.json`:
```powershell
venv\Scripts\python.exe scripts\build_bm25_index.py
venv\Scripts\python.exe scripts\build_classic_rag_index.py
venv\Scripts\python.exe scripts\build_swaga_index.py
```
Прогоны 6 методов на `data/eval/queries_5rag.jsonl`, унификация под token budget
и сборка пар. **Чистая конфигурация без конфаундов** — `configs/judge_v2_clean/`:
`swaga_chunks` и `swaga_windows` считаются на **идентичном** drill/expand/score-конфиге
(`swaga_unified_{chunks,windows}.json` — различие только в окнах), а все
embedding-методы — на одном энкодере **mpnet** (`dense` пересобирается на тех же
1696 chunk-id, что и раньше, меняется только энкодер MiniLM→mpnet):
```powershell
# SWAGA: один конфиг, разница только в упаковке (окна вкл/выкл)
venv\Scripts\python.exe scripts\run_queries_swaga.py            --config configs\judge_v2_clean\swaga_unified_chunks.json  --index_dir artifacts\indexes\swaga_index_dir
venv\Scripts\python.exe scripts\run_queries_swaga_subgraphs.py  --config configs\judge_v2_clean\swaga_unified_windows.json --index_dir artifacts\indexes\swaga_index_dir
# dense/dense_heuristic — на mpnet (перекодировать тексты classic-индекса энкодером mpnet); bm25/bm25_heuristic — лексические
venv\Scripts\python.exe scripts\judge_v2_unify.py       --methods_config configs\judge_v2_clean\methods.json --out_dir artifacts\judge_v2_clean\unified
venv\Scripts\python.exe scripts\judge_v2_make_pairs.py  --methods_config configs\judge_v2_clean\methods.json --unified_dir artifacts\judge_v2_clean\unified --out_path artifacts\judge_v2_clean\pairs_full150.jsonl
```
Оценка судьями (CometAPI, ключ `COMETAPI_KEY`). Модели/промпт/цены — в
`configs/judge_v2/judges.json` (DeepSeek-V4-Pro, Qwen3.5-Plus, Gemini-3.1-Flash-Lite;
temperature 0, thinking off); смета без вызовов — `judge_v2_cost_estimate.py`:
```powershell
$env:COMETAPI_KEY="<KEY>"
venv\Scripts\python.exe scripts\judge_v2_full_run.py --pairs artifacts\judge_v2_clean\pairs_full150.jsonl --out_dir artifacts\judge_v2_clean\full_run
venv\Scripts\python.exe scripts\judge_v2_aggregate.py --pairs artifacts\judge_v2_clean\pairs_full150.jsonl --full_dir artifacts\judge_v2_clean\full_run --out_dir artifacts\judge_v2_clean\agg
venv\Scripts\python.exe scripts\judge_v2_report_md.py --agg_dir artifacts\judge_v2_clean\agg --out results\llm_judge\report.md
```

## 2. Benchmark-корпуса (Qasper, BioASQ) + сменные энкодеры

Сборка ontology-индексов под конкретный энкодер (`--model-name`; e5/bge получают
query/passage-префиксы автоматически). Дефолт — mpnet (воспроизводит старые числа):
```powershell
venv\Scripts\python.exe scripts\build_qasper_swaga_index.py       --model-name "intfloat/e5-base-v2"
venv\Scripts\python.exe scripts\build_bioasq_pmc_swaga_index.py   --model-name "BAAI/bge-base-en-v1.5"
```
Гибридные прогоны (first-stage {bm25,dense}, top-m, пороги) и метрики — подробный
runbook: [`docs/windows_benchmarks.md`](docs/windows_benchmarks.md). Пример
(каноничная конфигурация Qasper = e5, bm25-first m5):
```powershell
venv\Scripts\python.exe scripts\run_qasper_hybrid_retrieval.py --model "intfloat/e5-base-v2" `
  --first-stage bm25 --bm25-top-m-docs 5 --threshold-mode absolute `
  --index-dir artifacts\indexes\qasper__e5-base-v2 --bm25-index artifacts\indexes\qasper_bm25_index.pkl `
  --out-dir artifacts\hybrid_rag_results\qasper_e5_chunks
venv\Scripts\python.exe scripts\evaluate_retrieval_metrics.py --run x=artifacts\hybrid_rag_results\qasper_e5_chunks --match-mode both
```
`--first-stage dense` требует `--dense-index` (ClassicRAGIndex на том же энкодере;
собирается из swaga-индекса, см. runbook). Сводная таблица по всем прогонам:
```powershell
venv\Scripts\python.exe scripts\build_stage3_summary.py       # -> results/retrieval/summary.md
```

## Заметки о честности сравнения

- `dense` baseline считается на **том же энкодере**, что SWAGA-индекс (иначе «прирост»
  спутывается со сменой энкодера). BM25 лексический — от энкодера не зависит; при
  пересборке индексов сверялся ассертом, что gold/метрики совпадают.
- `threshold_mode=absolute` — бит-в-бит прежний алгоритм drill-down (проверено на
  синтетике).
- Крупные датасеты (BioASQ PMC, Qasper) в репозиторий не включены.
