"""
Sanity probe for parse_judge_response — feeds it the typical wrappings each
provider tends to emit and prints the parse result. No API calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from judge_v2_client import parse_judge_response  # noqa: E402


CASES = [
    # OpenAI / Gemini with structured output: pure JSON
    (
        "openai_clean",
        '{"relevance":"A","cleanliness":"B","sufficiency":"equal"}',
    ),
    # Anthropic: markdown fence, json hint
    (
        "anthropic_fence_json",
        '```json\n{"relevance": "A", "cleanliness": "B", "sufficiency": "equal"}\n```',
    ),
    # Anthropic: fence without language hint
    (
        "anthropic_fence_plain",
        '```\n{"relevance":"A","cleanliness":"B","sufficiency":"equal"}\n```',
    ),
    # Anthropic: chatty preamble
    (
        "anthropic_preamble",
        'Here is my evaluation in JSON:\n\n{"relevance":"equal","cleanliness":"A","sufficiency":"B"}',
    ),
    # Anthropic: trailing comment after JSON
    (
        "anthropic_trailing",
        '{"relevance":"A","cleanliness":"A","sufficiency":"A"}\n\nLet me know if you need more detail.',
    ),
    # Gemini: pretty-printed JSON with whitespace
    (
        "gemini_pretty",
        '\n  {\n    "relevance": "B",\n    "cleanliness": "equal",\n    "sufficiency": "A"\n  }\n',
    ),
    # JSON value containing braces inside a string (defensive)
    (
        "defensive_braces_in_string",
        '{"relevance":"A","cleanliness":"B","sufficiency":"equal","_note":"contains } brace"}',
    ),
    # Bad: missing axis
    (
        "bad_missing_axis",
        '{"relevance":"A","cleanliness":"B"}',
    ),
    # Bad: invalid value
    (
        "bad_value",
        '{"relevance":"yes","cleanliness":"B","sufficiency":"equal"}',
    ),
    # Bad: no JSON at all
    (
        "bad_no_json",
        'Unable to evaluate this comparison.',
    ),
]


def main() -> None:
    print(f"{'case':30s} {'status':6s} {'detail'}")
    for name, payload in CASES:
        labels, err = parse_judge_response(payload)
        if labels is not None:
            print(f"{name:30s} OK     {labels}")
        else:
            print(f"{name:30s} FAIL   {err}")


if __name__ == "__main__":
    main()
