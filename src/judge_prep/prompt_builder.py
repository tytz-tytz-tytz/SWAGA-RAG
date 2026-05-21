from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Legacy letters (multi-way)
LETTERS: List[str] = ["A", "B", "C", "D", "E"]
PAIRWISE_LETTERS: List[str] = ["A", "B"]

# -----------------------------
# Helpers
# -----------------------------
def _as_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def _is_blank(s: str) -> bool:
    return len(s.strip()) == 0


def _normalize_contexts_dict(raw: Dict[str, Any]) -> Dict[str, str]:
    """Normalize contexts dict values to strings."""
    out: Dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(k, str):
            out[k] = _as_str(v)
    return out


def _extract_contexts_raw(payload: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Return raw contexts dict if present (any of known keys).
    Does NOT pad to A–E; returns exactly what payload provides.
    """
    raw = payload.get("contexts_for_judge")
    if isinstance(raw, dict):
        return _normalize_contexts_dict(raw)

    for key in ("retrieved_contexts", "contexts", "candidates"):
        raw2 = payload.get(key)
        if isinstance(raw2, dict):
            return _normalize_contexts_dict(raw2)

    return None


def _detect_mode_and_letters(payload: Dict[str, Any], mode: str) -> Tuple[str, List[str]]:
    """
    Decide whether to use multi-way (A–E) or pairwise (A–B).
    - mode: "auto" | "multiway" | "pairwise"
    """
    mode = (mode or "auto").strip().lower()
    if mode not in ("auto", "multiway", "pairwise"):
        raise ValueError("mode must be one of: auto | multiway | pairwise")

    if mode == "multiway":
        return "multiway", LETTERS
    if mode == "pairwise":
        return "pairwise", PAIRWISE_LETTERS

    # auto
    raw = _extract_contexts_raw(payload)
    if raw is None:
        # fall back to legacy multi-way
        return "multiway", LETTERS

    present = [k for k in raw.keys() if k in LETTERS]
    # If payload has exactly A and B (or generally 2 among A–E), treat as pairwise.
    if len(present) == 2 and set(present) == set(PAIRWISE_LETTERS):
        return "pairwise", PAIRWISE_LETTERS

    # Default: multi-way
    return "multiway", LETTERS


def _extract_contexts_for_letters(payload: Dict[str, Any], letters: List[str]) -> Dict[str, str]:
    """
    Extract contexts for specified letters.
    Pads missing letters with "" (only within requested letters).
    """
    raw = _extract_contexts_raw(payload)
    if raw is None:
        return {k: "" for k in letters}

    return {k: _as_str(raw.get(k, "")) for k in letters}


def _select_prompt_variant(prompt_variant: str) -> Tuple[str, str]:
    """
    Return (system_prompt, multiway_instructions) for the requested variant.
    """
    variant = (prompt_variant or "legacy").strip().lower()
    if variant in ("legacy", "default", "old"):
        return SYSTEM_PROMPT, USER_INSTRUCTIONS_MULTIWAY
    if variant in ("reasoning", "new", "downstream"):
        return SYSTEM_PROMPT_REASONING, USER_INSTRUCTIONS_MULTIWAY_REASONING
    raise ValueError("prompt_variant must be one of: legacy | reasoning")


# -----------------------------
# Prompt templates
# -----------------------------
SYSTEM_PROMPT = (
    "Ты — беспристрастный судья, оценивающий результаты поиска по продуктовой документации.\n\n"
    "Цель поиска — найти фрагменты документации, пригодные для последующего извлечения "
    "бизнес-логики, правил, ограничений, условий и сценариев поведения системы.\n\n"
    "Это НЕ задача ответа на вопрос пользователя и НЕ оценка качества сгенерированного ответа.\n"
    "Оценивай только качество найденного контекста.\n\n"
    "Все запросы и контексты написаны на русском языке. Оценивай их как есть."
)

SYSTEM_PROMPT_REASONING = SYSTEM_PROMPT

USER_INSTRUCTIONS_MULTIWAY = """Задача

Дано:
- поисковый запрос
- пять найденных контекстов-кандидатов A-E

Нужно оценить, какой контекст лучше подходит как evidence для последующего извлечения бизнес-логики системы.

Ключевой принцип оценки

Хороший контекст должен не просто быть тематически похожим на запрос.
Он должен попадать в правильный механизм / функцию / сценарий и содержать конкретную полезную логику.

Приоритеты оценки, от самого важного к менее важному:

1. Правильный anchor
Контекст должен относиться именно к запрошенному механизму, а не к соседней теме.
Например:
- запрос про "ошибки внешнего API" хуже покрывается общим описанием API без ошибок
- запрос про "формирование аудитории" хуже покрывается экспортом аудитории или терминами
- запрос про "настройку запуска" хуже покрывается только общими условиями старта

2. Наличие бизнес-логики
Контекст лучше, если содержит правила, условия, ограничения, шаги, переходы, исключения или поведение системы.
Общий обзор, определение или справочный текст хуже, даже если он тематически похож.

3. Логическая связность
Контекст лучше, если фрагменты поддерживают друг друга и позволяют восстановить цельную логику.
Набор случайных похожих кусков хуже.

4. Шум
Дополнительный текст допустим только если он структурно связан и помогает восстановить логику.
Нерелевантный, обзорный, повторяющийся или уводящий в сторону текст является шумом.

Что считать хорошим контекстом:
- попадает в нужный механизм или сценарий
- содержит конкретные правила, условия, ограничения или шаги
- даёт достаточно информации для дальнейшего reasoning / генерации
- сохраняет локальную связность
- не состоит только из заголовков или общих описаний

Что считать плохим контекстом:
- уходит в соседний механизм
- совпадает только по общей теме, но не по аспекту запроса
- содержит в основном определения, обзор или справочный текст
- теряет важные условия или ограничения
- содержит много нерелевантных фрагментов
- состоит из обрывков без связной логики

Важно:
- НЕ отвечай на запрос.
- НЕ оценивай стиль текста.
- НЕ выбирай автоматически самый длинный контекст.
- НЕ выбирай автоматически самый короткий контекст.
- Длина сама по себе не является преимуществом.
- Полезность определяется тем, насколько контекст помогает восстановить бизнес-логику.
- Если кандидат содержит только заголовки без содержательных фрагментов, это слабый контекст.
- Если кандидат содержит описание + связанные условия/список шагов, это может быть сильным контекстом, если всё относится к запросу.

Специальные правила:
- Строки вида '---' внутри кандидата считай частью контента, а не разделителем между кандидатами.
- Если кандидат пустой или содержит '[NO RELEVANT CONTEXT FOUND]', поставь ему 0 по всем метрикам и добавь в failure_letters.

Метрики 0-5:

1) relevance
Насколько контекст попадает именно в нужный механизм / функцию / сценарий запроса.

0 = полностью не по теме
1 = очень далёкая связь
2 = соседняя тема, но не нужный механизм
3 = нужная область, но не точный аспект
4 = почти точный механизм
5 = точно про запрошенный механизм / сценарий

2) usefulness_for_logic
Насколько контекст содержит извлекаемую бизнес-логику: правила, условия, ограничения, шаги, исключения, поведение.

0 = логики нет
1 = почти только обзор/определения
2 = есть отдельные намёки на логику
3 = есть полезная, но неполная логика
4 = достаточно конкретной логики
5 = прямо пригоден для downstream reasoning / генерации

3) coherence
Насколько контекст является связным evidence-фрагментом.

0 = несвязный набор обрывков
1 = в основном разрозненные куски
2 = частичная связность
3 = понятная локальная связность
4 = хорошо связанный фрагмент
5 = цельный логический фрагмент

4) noise
Сколько нерелевантной или отвлекающей информации.

0 = почти нет шума
1 = немного лишнего
2 = умеренный шум
3 = много лишнего
4 = большая часть текста шумная
5 = почти весь контекст шумный

5) overall
Итоговая полезность retrieval-контекста.

При выставлении overall используй такой порядок:
- если relevance <= 2, overall не должен быть выше 2
- если usefulness_for_logic <= 2, overall обычно не должен быть выше 3
- высокий coherence не компенсирует wrong anchor
- низкий noise не компенсирует отсутствие бизнес-логики
- лучший кандидат — тот, где одновременно хороший anchor, есть логика, связность и умеренный шум

Агрегация:
- winner: одна буква с максимальным overall; если точное равенство по лучшему overall, winner = ""
- ranking: все буквы A-E по убыванию overall; при равенстве по алфавиту
- failure_letters: пустые кандидаты или кандидаты с '[NO RELEVANT CONTEXT FOUND]'
- confidence: уверенность в ranking и winner

Формат ответа:
Верни один JSON строго такой структуры:

{
  "relevance": {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0},
  "usefulness_for_logic": {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0},
  "coherence": {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0},
  "noise": {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0},
  "overall": {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0},
  "winner": "",
  "ranking": ["A", "B", "C", "D", "E"],
  "failure_letters": [],
  "confidence": 0,
  "rationales": {
    "A": "",
    "B": "",
    "C": "",
    "D": "",
    "E": ""
  }
}

Рационалы:
- 1-2 предложения на кандидата
- объясняй через: anchor, business logic, coherence, noise
- не добавляй других полей
- не оборачивай JSON в markdown
"""

USER_INSTRUCTIONS_MULTIWAY_REASONING = USER_INSTRUCTIONS_MULTIWAY

# ---- New pairwise (A–B) instructions ----
USER_INSTRUCTIONS_PAIRWISE = """Task
Given:
- A user query
- Two candidate retrieved contexts labeled A and B (top-k retrieval results)

Choose which context better supports answering the user query using ONLY the information contained in that context.

Important rules
- Do NOT answer the query.
- Do NOT reward writing style, fluency, or phrasing.
- Do NOT reward longer contexts. More text is not better.
- Do NOT use any external knowledge. Judge only what is present in the contexts.
- Penalize irrelevant content, repetitions, and topic drift.
- Preserve the idea of top-k retrieval: if necessary information is not present in the provided context,
  that candidate is worse, even if such information might exist elsewhere.
- If both contexts are equally useful OR equally useless for answering the query, you MUST output Tie.
- IMPORTANT: Inside a candidate context you may see lines like '---'. Treat them as part of the
  retrieved content (chunk separators), NOT as separators between candidates.
- If a candidate contains the literal marker '[NO RELEVANT CONTEXT FOUND]' OR is empty/whitespace-only,
  treat it as providing no support.

Decision
Return one of:
- "A"  (A is better)
- "B"  (B is better)
- "Tie" (no clear advantage)

Output format (STRICT)
Return a single JSON object with EXACTLY this schema and keys:
{
  "decision": "A" | "B" | "Tie",
  "reason": "<1-3 sentences, content-based; do not invent facts; no spoilers>"
}

Do not add any other keys.
Do not wrap the JSON in markdown code fences.

---
"""


# -----------------------------
# Data model
# -----------------------------
@dataclass(frozen=True)
class JudgePrompt:
    """
    Internal representation of a single judge prompt.

    id: payload id (e.g., Q001)
    query: Russian query string
    candidates: dict letter -> context text (letters depends on mode)
    mode: "multiway" | "pairwise"
    letters: list of letters used in this prompt
    """
    id: str
    query: str
    candidates: Dict[str, str]
    mode: str
    letters: List[str]


# -----------------------------
# Builders
# -----------------------------
def build_prompts(
    payload: Any,
    mode: str = "auto",
    prompt_variant: str = "legacy",
) -> List[JudgePrompt]:
    """
    Build prompts from:
    - a single payload dict
    - OR a list of payload dicts

    Supported payload shape (recommended):
    {
      "id": "Q001",
      "query": "...",
      "contexts_for_judge": {"A": "...", ..., "E": "..."} OR {"A": "...", "B": "..."},
      "private_mapping": {"A": "...", ...}  # ignored here on purpose
    }

    mode:
      - "auto": choose by number of candidates present in payload
      - "multiway": force A–E
      - "pairwise": force A–B
    """
    items: List[Dict[str, Any]] = []

    if isinstance(payload, list):
        for it in payload:
            if isinstance(it, dict):
                items.append(it)
            else:
                raise TypeError(f"Expected list[dict] payload, got element type: {type(it)}")
    elif isinstance(payload, dict):
        items = [payload]
    else:
        raise TypeError(f"Expected payload dict or list[dict], got: {type(payload)}")

    # Validate the requested prompt variant for backward compatibility.
    _select_prompt_variant(prompt_variant)

    prompts: List[JudgePrompt] = []
    for item in items:
        pid = _as_str(item.get("id", "")).strip() or "UNKNOWN"
        query = _as_str(item.get("query", "")).strip()

        detected_mode, letters = _detect_mode_and_letters(item, mode=mode)
        candidates = _extract_contexts_for_letters(item, letters)

        # Ensure all requested letters exist and are strings
        candidates = {k: _as_str(candidates.get(k, "")) for k in letters}

        prompts.append(
            JudgePrompt(
                id=pid,
                query=query,
                candidates=candidates,
                mode=detected_mode,
                letters=list(letters),
            )
        )

    return prompts


def _render_candidates_block(candidates: Dict[str, str], letters: List[str]) -> str:
    parts: List[str] = []
    for letter in letters:
        ctx = _as_str(candidates.get(letter, ""))
        parts.append(
            f"===== CANDIDATE {letter} =====\n{ctx}\n===== END CANDIDATE {letter} =====\n"
        )
    return "\n".join(parts).rstrip() + "\n"


def prompts_to_markdown(
    prompts: Sequence[JudgePrompt],
    *,
    title: Optional[str] = None,
    prompt_variant: str = "legacy",
) -> str:
    """
    Render one or many prompts into a Markdown file for inspection.

    Note: The model prompt itself is not wrapped in code fences on purpose,
    because you later embed the same content into chat messages.
    """
    lines: List[str] = []
    if title:
        lines.append(f"# {title}\n")

    system_prompt, multiway_instructions = _select_prompt_variant(prompt_variant)

    for p in prompts:
        lines.append(f"## {p.id}\n")
        lines.append(system_prompt.strip() + "\n")

        if p.mode == "pairwise":
            lines.append(USER_INSTRUCTIONS_PAIRWISE.rstrip())
        else:
            lines.append(multiway_instructions.rstrip())

        lines.append("User query (RU)\n" + p.query + "\n\n")
        lines.append("Retrieved contexts\n")
        lines.append(_render_candidates_block(p.candidates, p.letters))
        lines.append("\n---\n")

    return "\n".join(lines).rstrip() + "\n"


def prompts_to_messages(
    prompts: Sequence[JudgePrompt],
    *,
    prompt_variant: str = "legacy",
) -> List[Dict[str, str]]:
    """
    Convert prompts into OpenAI Chat Completions-style messages.

    If you pass multiple prompts, they will be concatenated into a single user message.
    For your pipeline, you usually want one payload per file anyway.
    """
    system_prompt, multiway_instructions = _select_prompt_variant(prompt_variant)

    user_chunks: List[str] = []
    for p in prompts:
        instructions = USER_INSTRUCTIONS_PAIRWISE if p.mode == "pairwise" else multiway_instructions
        chunk = (
            instructions
            + "User query (RU)\n"
            + p.query
            + "\n\nRetrieved contexts\n"
            + _render_candidates_block(p.candidates, p.letters)
        )
        user_chunks.append(chunk.rstrip())

    user_content = "\n\n".join(user_chunks).rstrip() + "\n"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def dumps_messages_json(
    id_: str,
    prompts: Sequence[JudgePrompt],
    *,
    prompt_variant: str = "legacy",
) -> str:
    """
    Convenience helper to produce {"id": ..., "messages": [...]} as JSON text.
    """
    obj = {"id": id_, "messages": prompts_to_messages(prompts, prompt_variant=prompt_variant)}
    return json.dumps(obj, ensure_ascii=False, indent=2)

