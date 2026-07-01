# LLM-as-judge — сводный отчёт (judge_v2)

> **Чистый прогон (сняты два конфаунда).** `swaga_chunks` и `swaga_windows`
> считаются на **идентичном** конфиге drill/expand/score (проверено ассертом) —
> единственное различие между ними это этап упаковки контекста (окна вкл/выкл),
> поэтому выигрыш `swaga_windows` над `swaga_chunks` (**overall 90%**) — чистый
> эффект оконной сборки. Все embedding-методы (`swaga_*`, `dense`, `dense+heur`) —
> на одном энкодере **paraphrase-multilingual-mpnet-base-v2**; `dense`
> пересобран на тех же 1696 chunk-id, сменён только энкодер. BM25 — лексический.
> Панель судей: DeepSeek-V4-Pro, Qwen3.5-Plus, Gemini-3.1-Flash-Lite
> (temp 0, thinking off, 2 перестановки, решение ≥4/6 с инверсией BA).

## Таблица 7–8. Парные сравнения (win A / tie / win B)

| Сравнение (A vs B) | Ось | win A | tie | win B | n |
|---|---|---:|---:|---:|---:|
| swaga_chunks vs bm25 | relevance | 23% | 11% | 66% | 150 |
| swaga_chunks vs bm25 | cleanliness | 65% | 13% | 21% | 150 |
| swaga_chunks vs bm25 | sufficiency | 22% | 12% | 66% | 150 |
| **swaga_chunks vs bm25** | **overall** | **37%** | **12%** | **51%** | |
| swaga_chunks vs bm25_heuristic | relevance | 23% | 11% | 67% | 150 |
| swaga_chunks vs bm25_heuristic | cleanliness | 51% | 13% | 36% | 150 |
| swaga_chunks vs bm25_heuristic | sufficiency | 23% | 11% | 66% | 150 |
| **swaga_chunks vs bm25_heuristic** | **overall** | **32%** | **12%** | **56%** | |
| swaga_chunks vs dense | relevance | 23% | 21% | 56% | 150 |
| swaga_chunks vs dense | cleanliness | 84% | 5% | 11% | 150 |
| swaga_chunks vs dense | sufficiency | 24% | 20% | 56% | 150 |
| **swaga_chunks vs dense** | **overall** | **44%** | **15%** | **41%** | |
| swaga_chunks vs dense_heuristic | relevance | 19% | 14% | 67% | 150 |
| swaga_chunks vs dense_heuristic | cleanliness | 53% | 17% | 31% | 150 |
| swaga_chunks vs dense_heuristic | sufficiency | 20% | 13% | 67% | 150 |
| **swaga_chunks vs dense_heuristic** | **overall** | **31%** | **14%** | **55%** | |
| swaga_windows vs bm25 | relevance | 67% | 9% | 23% | 150 |
| swaga_windows vs bm25 | cleanliness | 85% | 2% | 13% | 150 |
| swaga_windows vs bm25 | sufficiency | 67% | 9% | 23% | 150 |
| **swaga_windows vs bm25** | **overall** | **73%** | **7%** | **20%** | |
| swaga_windows vs bm25_heuristic | relevance | 62% | 9% | 29% | 150 |
| swaga_windows vs bm25_heuristic | cleanliness | 81% | 5% | 15% | 150 |
| swaga_windows vs bm25_heuristic | sufficiency | 64% | 8% | 28% | 150 |
| **swaga_windows vs bm25_heuristic** | **overall** | **69%** | **7%** | **24%** | |
| swaga_windows vs dense | relevance | 77% | 5% | 19% | 150 |
| swaga_windows vs dense | cleanliness | 95% | 2% | 3% | 150 |
| swaga_windows vs dense | sufficiency | 75% | 6% | 19% | 150 |
| **swaga_windows vs dense** | **overall** | **82%** | **4%** | **14%** | |
| swaga_windows vs dense_heuristic | relevance | 63% | 8% | 29% | 150 |
| swaga_windows vs dense_heuristic | cleanliness | 84% | 6% | 10% | 150 |
| swaga_windows vs dense_heuristic | sufficiency | 64% | 7% | 29% | 150 |
| **swaga_windows vs dense_heuristic** | **overall** | **70%** | **7%** | **23%** | |
| swaga_windows vs swaga_chunks | relevance | 87% | 13% | 0% | 150 |
| swaga_windows vs swaga_chunks | cleanliness | 94% | 5% | 1% | 150 |
| swaga_windows vs swaga_chunks | sufficiency | 89% | 11% | 0% | 150 |
| **swaga_windows vs swaga_chunks** | **overall** | **90%** | **10%** | **0%** | |

## Таблица 9. Согласованность судей

### Inter-judge agreement (доля пар, где все судьи согласны)

| Ось | rate |
|---|---:|
| relevance | 73% |
| cleanliness | 71% |
| sufficiency | 74% |
| **mean** | **73%** |

### Permutation consistency (BA == invert(AB)) по судьям

| Судья | relevance | cleanliness | sufficiency | mean |
|---|---:|---:|---:|---:|
| deepseek_v4_pro | 61% | 73% | 62% | 65% |
| qwen3_5_plus | 88% | 85% | 88% | 87% |
| gemini_3_1_flash_lite | 75% | 77% | 77% | 76% |

## Операционная сводка / стоимость

| Судья | calls | ok | failed | retry | in_tok | out_tok | $ est |
|---|---:|---:|---:|---:|---:|---:|---:|
| deepseek_v4_pro | 2700 | 2700 | 0 | 0 | 8147016 | 60366 | $3.439 |
| qwen3_5_plus | 2700 | 2700 | 0 | 0 | 7340710 | 61485 | $3.010 |
| gemini_3_1_flash_lite | 2700 | 2696 | 4 | 4 | 7226974 | 53920 | $1.510 |
| **TOTAL** | 8100 | 8096 | 4 | 4 | 22714700 | 175771 | $7.960 |

> cost_usd_estimated считается по публичным ценам из configs/judge_v2/judges.json. Все три судьи (DeepSeek-V4-Pro, Qwen3.5-Plus, Gemini-3.1-Flash-Lite) идут через шлюз CometAPI, поэтому фактически списанное может отличаться от оценки из-за наценки шлюза (фактический billed этого прогона — $7.96).
