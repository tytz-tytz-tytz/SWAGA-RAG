# Этап 3–4 — сводная таблица retrieval (encoder / first-stage / threshold)

## Qasper (888 запросов; strict ‖ same_section)

| Метод | s R@5 | s R@10 | s MRR | s nDCG | ss R@5 | ss R@10 | ss MRR | ss nDCG |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| BM25 (lexical) | 0.085 | 0.123 | 0.082 | 0.076 | 0.137 | 0.189 | 0.108 | 0.138 |
| dense · e5 | 0.100 | 0.134 | 0.102 | 0.091 | 0.139 | 0.183 | 0.119 | 0.148 |
| hybrid chunks · mpnet · bm25-first m5 | 0.065 | 0.084 | 0.067 | 0.062 | 0.084 | 0.090 | 0.083 | 0.168 |
| hybrid chunks · bge · bm25-first m5 | 0.073 | 0.094 | 0.075 | 0.071 | 0.093 | 0.102 | 0.092 | 0.188 |
| hybrid chunks · e5 · bm25-first m5 | 0.091 | 0.108 | 0.075 | 0.076 | 0.109 | 0.116 | 0.088 | 0.201 |
| hybrid chunks · e5 · bm25-first m10 | 0.083 | 0.103 | 0.071 | 0.073 | 0.101 | 0.109 | 0.085 | 0.183 |
| hybrid chunks · e5 · bm25-first m20 | 0.080 | 0.099 | 0.071 | 0.071 | 0.098 | 0.106 | 0.086 | 0.177 |
| hybrid chunks · e5 · dense-first m5 | 0.067 | 0.088 | 0.063 | 0.062 | 0.082 | 0.092 | 0.073 | 0.153 |
| hybrid chunks · e5 · dense-first m10 | 0.063 | 0.081 | 0.062 | 0.060 | 0.078 | 0.085 | 0.072 | 0.147 |
| hybrid chunks · e5 · dense-first m20 | 0.060 | 0.076 | 0.059 | 0.056 | 0.073 | 0.080 | 0.070 | 0.137 |
| hybrid windows · mpnet · bm25-first m5 | 0.065 | 0.080 | 0.063 | 0.060 | 0.075 | 0.084 | 0.078 | 0.192 |
| hybrid windows · bge · bm25-first m5 | 0.068 | 0.091 | 0.063 | 0.064 | 0.078 | 0.096 | 0.085 | 0.200 |
| hybrid windows · e5 · bm25-first m5 | 0.075 | 0.106 | 0.063 | 0.069 | 0.090 | 0.109 | 0.080 | 0.221 |

## BioASQ (280 вопросов; snippet-level strict)

| Метод | R@5 | R@10 | MRR | nDCG@10 | noise |
|---|--:|--:|--:|--:|--:|
| BM25 (lexical) | 0.179 | 0.290 | 0.397 | 0.257 | 0.845 |
| dense · bge | 0.220 | 0.362 | 0.430 | 0.304 | 0.815 |
| hybrid chunks · mpnet · bm25-first m5 | 0.205 | 0.237 | 0.483 | 0.253 | 0.839 |
| hybrid chunks · pubmedbert · bm25-first m5 | 0.220 | 0.259 | 0.457 | 0.251 | 0.838 |
| hybrid chunks · bge · bm25-first m5 | 0.249 | 0.294 | 0.497 | 0.285 | 0.817 |
| hybrid chunks · bge · bm25-first m10 | 0.258 | 0.298 | 0.524 | 0.295 | 0.805 |
| hybrid chunks · bge · bm25-first m20 | 0.274 | 0.316 | 0.537 | 0.311 | 0.793 |
| hybrid chunks · bge · dense-first m5 | 0.272 | 0.315 | 0.527 | 0.311 | 0.791 |
| hybrid chunks · bge · dense-first m10 | 0.272 | 0.315 | 0.533 | 0.313 | 0.789 |
| hybrid chunks · bge · dense-first m20 | 0.269 | 0.312 | 0.540 | 0.312 | 0.790 |
| hybrid windows · mpnet · dense-first m5 | 0.161 | 0.193 | 0.418 | 0.211 | — |
| hybrid windows · pubmedbert · dense-first m5 | 0.179 | 0.236 | 0.418 | 0.231 | — |
| hybrid windows · bge · dense-first m5 | 0.226 | 0.292 | 0.459 | 0.280 | 0.800 |

## Окна как единица выдачи (вариант C)

Метрики, где единица ранжирования — окно целиком (а не развёрнутые chunk-id). Тот же
каноничный first-stage: Qasper bm25-first m5 (same_section), BioASQ dense-first m5 (strict).

| Корпус | Энкодер | R@5 | R@10 | MRR | nDCG@10 | noise |
|---|---|--:|--:|--:|--:|--:|
| Qasper | mpnet | 0.087 | 0.090 | 0.084 | 0.079 | 0.961 |
| Qasper | bge   | 0.099 | 0.102 | 0.094 | 0.089 | 0.958 |
| Qasper | e5    | 0.114 | 0.116 | 0.089 | 0.092 | 0.954 |
| BioASQ | mpnet      | 0.216 | 0.216 | 0.509 | 0.246 | 0.686 |
| BioASQ | pubmedbert | 0.267 | 0.267 | 0.534 | 0.275 | 0.667 |
| BioASQ | bge        | 0.316 | 0.316 | 0.595 | 0.322 | 0.619 |

_Примечания: dense = classic_rag (тот же энкодер, что SWAGA-индекс). hybrid = BM25/dense
first-stage doc-recall + SWAGA in-doc localization. Вариант A: окна разворачиваются в
chunk-id → те же chunk-level метрики (нейтральны/хуже на exact-match, но поднимают
Qasper same_section nDCG на всех энкодерах). Вариант C: окно как единица — BioASQ MRR
на bge доходит до 0.595 (выше chunk-level 0.527), noise падает с силой энкодера. Полный
эффект окон проявляется в качестве контекста (LLM-судья, окна 68–84% побед). threshold_mode
инертен на канонической конфигурации (см. отдельную сводку)._
