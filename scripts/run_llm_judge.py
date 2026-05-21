#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run LLM-as-judge prompts against OpenAI-compatible APIs and save strict JSON outputs.

Supports two OpenAI-compatible endpoint styles:
- Responses API:     POST {base_url}/responses
- Chat Completions:  POST {base_url}/chat/completions

Supports two judge schemas (determined by prompt content):
1) Fiveway (A–E): relevance/usefulness_for_logic/noise/overall + winner/ranking/failure/confidence/rationales
2) Pairwise (A/B): {"decision": "A"|"B"|"" , "reason": "..."}

Prompt files are expected as:
  artifacts/judge_prompts/<group>/Q001.messages.json

Outputs are written as:
  artifacts/judge_outputs/<group>/<judge_name>/Q001_1.json
  artifacts/judge_outputs/<group>/<judge_name>/Q001_2.json
  ...

Output envelope schema:
{
  "qid": "Q001",
  "judge": {...},
  "meta": {...},
  "status": "OK" | "ERROR" | "DRY_RUN",
  "raw_response_text": "...",
  "parsed_json": {schema...} | null,
  "error": null | "..."
}

Notes:
- Comments inside code are in English (as requested).
- .env is loaded automatically via python-dotenv.
- A strict system prefix is prepended to reduce format drift.
- Automatic retries are performed on JSON/schema violations.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

# Load environment variables from .env in current working directory (repo root).
load_dotenv()


LETTERS_5 = ["A", "B", "C", "D", "E"]
LETTERS_2 = ["A", "B"]

STRICT_JSON_SYSTEM: Dict[str, str] = {
    "role": "system",
    "content": (
        "Return ONLY a single JSON object and nothing else. "
        "No markdown code fences. No prose. "
        "Use EXACTLY the required keys and schema as specified in the prompt. "
        "Do NOT add extra keys."
    ),
}


# ---------------------------
# Config
# ---------------------------

@dataclass
class RunConfig:
    input_dir: Path
    output_root: Path
    group: str
    judge_name: str
    replicas: int
    temperature: float
    max_output_tokens: int
    dry_run: bool
    overwrite: bool

    provider: str
    api_key_env: str
    base_url: str
    api_style: str  # "responses" | "chat_completions"

    max_retries: int
    retry_temperature: float
    timeout_s: int = 120


# ---------------------------
# IO helpers
# ---------------------------

def _write_json(path: Path, obj: Dict[str, Any], overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_prompt_file(p: Path) -> bool:
    return bool(re.match(r"^Q[0-9A-Za-z]+\.messages\.json$", p.name))


def _parse_qid_from_prompt_filename(p: Path) -> str:
    m = re.match(r"^(Q[0-9A-Za-z]+)\.messages\.json$", p.name)
    if not m:
        raise ValueError(f"Unexpected prompt filename: {p.name}")
    return m.group(1)


# ---------------------------
# JSON extraction
# ---------------------------

def _extract_first_json_object(text: str) -> str:
    """
    Extract the first top-level JSON object from a string.
    Works even if the model wraps it with ```json ... ``` or includes extra text.
    """
    if not isinstance(text, str):
        raise ValueError("raw response is not a string")

    s = text.strip()
    if not s:
        raise ValueError("empty response text")

    start = s.find("{")
    if start < 0:
        raise ValueError("no JSON object start '{' found")

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]

    raise ValueError("unterminated JSON object (brace matching failed)")


# ---------------------------
# Prompt loading
# ---------------------------

def _normalize_messages(data: Any, path: Path) -> List[Dict[str, Any]]:
    """
    Accept multiple possible shapes:
    1) List of {"role": ..., "content": ...}
    2) Dict with "messages": [ ... ]
    3) Dict with "input": [ ... ] (Responses-style)
    """
    if isinstance(data, list):
        messages = data
    elif isinstance(data, dict):
        if isinstance(data.get("messages"), list):
            messages = data["messages"]
        elif isinstance(data.get("input"), list):
            messages = data["input"]
        else:
            raise ValueError(
                f"Expected a list or dict with 'messages'/'input' in {path}, got keys: {list(data.keys())}"
            )
    else:
        raise ValueError(f"Expected list/dict in {path}, got: {type(data)}")

    norm: List[Dict[str, Any]] = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            raise ValueError(f"Message {i} in {path} is not a dict")
        role = m.get("role")
        content = m.get("content")
        if role not in ("system", "user", "assistant", "developer"):
            raise ValueError(f"Message {i} in {path} has invalid role: {role}")
        if content is None:
            raise ValueError(f"Message {i} in {path} missing content")
        norm.append({"role": role, "content": content})
    return norm


def _load_messages(path: Path) -> List[Dict[str, Any]]:
    data = _load_json(path)
    return _normalize_messages(data, path)


def _detect_schema_mode(messages: List[Dict[str, Any]]) -> str:
    """
    Detect which schema the prompt expects.
    Heuristics:
    - if prompt mentions sufficiency_for_reasoning or coherence, use fiveway_reasoning
    - else if prompt mentions candidates C/D/E, assume legacy fiveway
    - otherwise pairwise
    """
    full = "\n".join(str(m.get("content", "")) for m in messages)
    if ("sufficiency_for_reasoning" in full) or ("coherence" in full):
        return "fiveway_reasoning"
    if ("CANDIDATE C" in full) or ("CANDIDATE D" in full) or ("CANDIDATE E" in full):
        return "fiveway"
    # Your ablation_pairs always use A/B.
    return "pairwise"


# ---------------------------
# Schema validation
# ---------------------------

def _validate_metric_map(d: Any, key: str) -> None:
    if not isinstance(d, dict):
        raise ValueError(f"'{key}' must be an object")
    for L in LETTERS_5:
        if L not in d:
            raise ValueError(f"'{key}' missing letter {L}")
        v = d[L]
        if not isinstance(v, int):
            raise ValueError(f"'{key}.{L}' must be int")
        if v < 0 or v > 5:
            raise ValueError(f"'{key}.{L}' must be in [0,5]")


def _validate_fiveway(obj: Dict[str, Any]) -> None:
    """
    Validate fiveway judge schema strictly.
    """
    required_keys = [
        "relevance",
        "noise",
        "overall",
        "winner",
        "ranking",
        "failure_letters",
        "confidence",
        "rationales",
    ]
    metric_key = None
    if "usefulness_for_logic" in obj:
        metric_key = "usefulness_for_logic"
    elif "answerability" in obj:
        metric_key = "answerability"
    else:
        required_keys.append("usefulness_for_logic")
        metric_key = "usefulness_for_logic"

    missing = [k for k in required_keys if k not in obj]
    if missing:
        raise ValueError(f"Missing keys: {missing}")

    allowed_keys = set(required_keys)
    allowed_keys.add("answerability")
    allowed_keys.add("usefulness_for_logic")
    extra = sorted([k for k in obj.keys() if k not in allowed_keys])
    if extra:
        raise ValueError(f"Unexpected extra keys: {extra}")

    _validate_metric_map(obj["relevance"], "relevance")
    _validate_metric_map(obj[metric_key], metric_key)
    _validate_metric_map(obj["noise"], "noise")
    _validate_metric_map(obj["overall"], "overall")

    winner = obj["winner"]
    if not isinstance(winner, str):
        raise ValueError("'winner' must be a string")
    if winner != "" and winner not in LETTERS_5:
        raise ValueError("'winner' must be empty string or one of A-E")

    ranking = obj["ranking"]
    if not isinstance(ranking, list):
        raise ValueError("'ranking' must be a list")
    ranking_norm = [x for x in ranking if isinstance(x, str)]
    if len(ranking_norm) != 5 or sorted(set(ranking_norm)) != sorted(LETTERS_5):
        raise ValueError("'ranking' must contain exactly 5 unique letters A-E")

    failure = obj["failure_letters"]
    if not isinstance(failure, list):
        raise ValueError("'failure_letters' must be a list")


def _validate_fiveway_reasoning(obj: Dict[str, Any]) -> None:
    """
    Validate fiveway reasoning judge schema strictly.
    """
    required_keys = [
        "relevance",
        "sufficiency_for_reasoning",
        "coherence",
        "noise",
        "overall",
        "winner",
        "ranking",
        "failure_letters",
        "confidence",
        "rationales",
    ]

    missing = [k for k in required_keys if k not in obj]
    if missing:
        raise ValueError(f"Missing keys: {missing}")

    extra = sorted([k for k in obj.keys() if k not in required_keys])
    if extra:
        raise ValueError(f"Unexpected extra keys: {extra}")

    _validate_metric_map(obj["relevance"], "relevance")
    _validate_metric_map(obj["sufficiency_for_reasoning"], "sufficiency_for_reasoning")
    _validate_metric_map(obj["coherence"], "coherence")
    _validate_metric_map(obj["noise"], "noise")
    _validate_metric_map(obj["overall"], "overall")

    winner = obj["winner"]
    if not isinstance(winner, str):
        raise ValueError("'winner' must be a string")
    if winner != "" and winner not in LETTERS_5:
        raise ValueError("'winner' must be empty string or one of A-E")

    ranking = obj["ranking"]
    if not isinstance(ranking, list):
        raise ValueError("'ranking' must be a list")
    ranking_norm = [x for x in ranking if isinstance(x, str)]
    if len(ranking_norm) != 5 or sorted(set(ranking_norm)) != sorted(LETTERS_5):
        raise ValueError("'ranking' must contain exactly 5 unique letters A-E")

    failure = obj["failure_letters"]
    if not isinstance(failure, list):
        raise ValueError("'failure_letters' must be a list")

    confidence = obj["confidence"]
    if not isinstance(confidence, int):
        raise ValueError("'confidence' must be an int")
    if confidence < 0 or confidence > 5:
        raise ValueError("'confidence' must be in [0,5]")

    rationales = obj["rationales"]
    if not isinstance(rationales, dict):
        raise ValueError("'rationales' must be an object")
    for L in LETTERS_5:
        if L not in rationales:
            raise ValueError(f"'rationales' missing letter {L}")
        if not isinstance(rationales[L], str):
            raise ValueError(f"'rationales.{L}' must be a string")
    for x in failure:
        if not isinstance(x, str) or x not in LETTERS_5:
            raise ValueError("'failure_letters' must contain only letters A-E")

    conf = obj["confidence"]
    if not isinstance(conf, int):
        raise ValueError("'confidence' must be int")
    if conf < 0 or conf > 5:
        raise ValueError("'confidence' must be in [0,5]")

    rats = obj["rationales"]
    if not isinstance(rats, dict):
        raise ValueError("'rationales' must be an object")
    for L in LETTERS_5:
        if L not in rats:
            raise ValueError(f"'rationales' missing letter {L}")
        if not isinstance(rats[L], str):
            raise ValueError(f"'rationales.{L}' must be a string")


def _validate_pairwise(obj: Dict[str, Any]) -> None:
    """
    Validate pairwise judge schema strictly:
      {"decision": "A"|"B"|"" , "reason": "..."}
    """
    required_keys = ["decision", "reason"]
    missing = [k for k in required_keys if k not in obj]
    if missing:
        raise ValueError(f"Missing keys: {missing}")

    extra = sorted([k for k in obj.keys() if k not in required_keys])
    if extra:
        raise ValueError(f"Unexpected extra keys: {extra}")

    decision = obj["decision"]
    if not isinstance(decision, str):
        raise ValueError("'decision' must be a string")
    if decision not in ("A", "B", ""):
        raise ValueError("'decision' must be 'A', 'B', or ''")

    reason = obj["reason"]
    if not isinstance(reason, str):
        raise ValueError("'reason' must be a string")


# ---------------------------
# API calls
# ---------------------------

def _post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
    return resp.json()


def _openai_responses_call(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_output_tokens: int,
    timeout_s: int,
) -> str:
    """
    Call OpenAI-compatible Responses API and return output text.

    Enforce JSON output via:
      "text": {"format": {"type": "json_object"}}
    """
    url = base_url.rstrip("/") + "/responses"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": model,
        "input": messages,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "text": {"format": {"type": "json_object"}},
    }

    data = _post_json(url, headers, payload, timeout_s)

    # Prefer output_text if available.
    if isinstance(data.get("output_text"), str) and data["output_text"].strip():
        return data["output_text"].strip()

    # Try standard "output[0].content[*].text" structure.
    output = data.get("output")
    if isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, dict):
            content = first.get("content")
            if isinstance(content, list) and content:
                for c in content:
                    if isinstance(c, dict) and isinstance(c.get("text"), str) and c["text"].strip():
                        return c["text"].strip()

    raise ValueError(f"Unexpected Responses API shape: {data}")


def _openai_chat_completions_call(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_output_tokens: int,
    timeout_s: int,
) -> str:
    """
    Call OpenAI-compatible Chat Completions API and return assistant message content text.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_output_tokens,
    }

    data = _post_json(url, headers, payload, timeout_s)

    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        raise ValueError(f"Unexpected chat.completions response shape: {data}")

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Empty assistant content from chat.completions")

    return content.strip()


# ---------------------------
# Runner logic
# ---------------------------

def _iter_prompt_files(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")
    return [p for p in sorted(input_dir.glob("Q*.messages.json")) if p.is_file() and _is_prompt_file(p)]


def _output_path(cfg: RunConfig, qid: str, replica: int) -> Path:
    # Make judge name safe as a directory name.
    safe_judge = re.sub(r"[^A-Za-z0-9._-]+", "_", cfg.judge_name)
    return cfg.output_root / cfg.group / safe_judge / f"{qid}_{replica}.json"


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="artifacts/judge_outputs")
    parser.add_argument("--group", type=str, required=True)
    parser.add_argument("--judge_name", type=str, required=True)

    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_output_tokens", type=int, default=1200)

    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--provider", type=str, default="openai", choices=["openai"])
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1")
    parser.add_argument(
        "--api_style",
        type=str,
        default="responses",
        choices=["responses", "chat_completions"],
        help="Which OpenAI-compatible endpoint to use (responses or chat_completions).",
    )

    parser.add_argument("--max_retries", type=int, default=2, help="Retries on invalid JSON/schema.")
    parser.add_argument("--retry_temperature", type=float, default=0.0, help="Temperature used on retry attempts.")
    parser.add_argument("--timeout_s", type=int, default=120)
    parser.add_argument(
        "--log_every",
        type=int,
        default=1,
        help="Print a progress line every N prompt files (default: 1).",
    )

    args = parser.parse_args()

    cfg = RunConfig(
        input_dir=Path(args.input_dir),
        output_root=Path(args.output_root),
        group=str(args.group).replace("\\", "/").strip("/"),
        judge_name=str(args.judge_name),
        replicas=int(args.replicas),
        temperature=float(args.temperature),
        max_output_tokens=int(args.max_output_tokens),
        dry_run=bool(args.dry_run),
        overwrite=bool(args.overwrite),
        provider=str(args.provider),
        api_key_env=str(args.api_key_env),
        base_url=str(args.base_url),
        api_style=str(args.api_style),
        max_retries=int(args.max_retries),
        retry_temperature=float(args.retry_temperature),
        timeout_s=int(args.timeout_s),
    )

    api_key = os.environ.get(cfg.api_key_env, "").strip()
    if not cfg.dry_run and not api_key:
        raise RuntimeError(f"API key not found in environment variable: {cfg.api_key_env}")

    prompt_files = _iter_prompt_files(cfg.input_dir)
    print(f"Found prompts: {len(prompt_files)}", flush=True)
    planned = len(prompt_files) * max(cfg.replicas, 1)
    print(f"Planned jobs: {planned} (replicas={cfg.replicas})", flush=True)

    written = 0

    for prompt_idx, pf in enumerate(prompt_files, start=1):
        qid = _parse_qid_from_prompt_filename(pf)
        messages = _load_messages(pf)

        # Determine schema mode from prompt contents.
        schema_mode = _detect_schema_mode(messages)

        # Prepend strict system instruction to reduce format drift.
        messages_to_send = [STRICT_JSON_SYSTEM] + messages

        for replica in range(1, cfg.replicas + 1):
            out_path = _output_path(cfg, qid, replica)
            if out_path.exists() and not cfg.overwrite:
                if args.log_every > 0 and (prompt_idx == 1 or prompt_idx % args.log_every == 0):
                    print(f"[SKIP] {prompt_idx}/{len(prompt_files)} {qid} replica={replica} -> exists", flush=True)
                continue

            if args.log_every > 0 and (prompt_idx == 1 or prompt_idx % args.log_every == 0):
                print(
                    f"[RUN ] {prompt_idx}/{len(prompt_files)} {qid} replica={replica} "
                    f"model={cfg.judge_name} schema={schema_mode}",
                    flush=True,
                )

            # Dry run: write envelope only.
            if cfg.dry_run:
                obj = {
                    "qid": qid,
                    "judge": {
                        "name": cfg.judge_name,
                        "replica": replica,
                        "temperature": cfg.temperature,
                        "max_output_tokens": cfg.max_output_tokens,
                    },
                    "meta": {
                        "created_at_unix": int(time.time()),
                        "input_messages_count": len(messages),
                        "schema_mode": schema_mode,
                    },
                    "status": "DRY_RUN",
                    "raw_response_text": "",
                    "parsed_json": None,
                    "error": None,
                }
                _write_json(out_path, obj, overwrite=True)
                written += 1
                if args.log_every > 0 and (prompt_idx == 1 or prompt_idx % args.log_every == 0):
                    print(f"[DONE] {qid} replica={replica} -> DRY_RUN", flush=True)
                continue

            raw_text = ""
            last_err: Optional[Exception] = None

            # Retry loop for schema/JSON drift.
            for attempt in range(cfg.max_retries + 1):
                try:
                    temp = cfg.temperature if attempt == 0 else cfg.retry_temperature
                    if args.log_every > 0 and (prompt_idx == 1 or prompt_idx % args.log_every == 0):
                        print(
                            f"[TRY ] {qid} replica={replica} attempt={attempt + 1}/{cfg.max_retries + 1} temp={temp}",
                            flush=True,
                        )

                    if cfg.provider != "openai":
                        raise RuntimeError(f"Unsupported provider: {cfg.provider}")

                    if cfg.api_style == "responses":
                        raw_text = _openai_responses_call(
                            base_url=cfg.base_url,
                            api_key=api_key,
                            model=cfg.judge_name,
                            messages=messages_to_send,
                            temperature=temp,
                            max_output_tokens=cfg.max_output_tokens,
                            timeout_s=cfg.timeout_s,
                        )
                    elif cfg.api_style == "chat_completions":
                        raw_text = _openai_chat_completions_call(
                            base_url=cfg.base_url,
                            api_key=api_key,
                            model=cfg.judge_name,
                            messages=messages_to_send,
                            temperature=temp,
                            max_output_tokens=cfg.max_output_tokens,
                            timeout_s=cfg.timeout_s,
                        )
                    else:
                        raise RuntimeError(f"Unsupported api_style: {cfg.api_style}")

                    json_str = _extract_first_json_object(raw_text)
                    parsed = json.loads(json_str)
                    if not isinstance(parsed, dict):
                        raise ValueError("Parsed JSON is not an object")

                    if schema_mode == "fiveway":
                        _validate_fiveway(parsed)
                    elif schema_mode == "fiveway_reasoning":
                        _validate_fiveway_reasoning(parsed)
                    else:
                        _validate_pairwise(parsed)

                    ok_obj = {
                        "qid": qid,
                        "judge": {
                            "name": cfg.judge_name,
                            "replica": replica,
                            "temperature": cfg.temperature,
                            "max_output_tokens": cfg.max_output_tokens,
                        },
                        "meta": {
                            "created_at_unix": int(time.time()),
                            "input_messages_count": len(messages),
                            "schema_mode": schema_mode,
                            "attempts": attempt + 1,
                        },
                        "status": "OK",
                        "raw_response_text": raw_text,
                        "parsed_json": parsed,
                        "error": None,
                    }
                    _write_json(out_path, ok_obj, overwrite=True)
                    written += 1
                    last_err = None
                    if args.log_every > 0 and (prompt_idx == 1 or prompt_idx % args.log_every == 0):
                        print(f"[DONE] {qid} replica={replica} -> OK", flush=True)
                    break

                except Exception as e:
                    last_err = e
                    if args.log_every > 0 and (prompt_idx == 1 or prompt_idx % args.log_every == 0):
                        print(f"[ERR ] {qid} replica={replica} attempt={attempt + 1}: {type(e).__name__}: {e}", flush=True)
                    continue

            if last_err is not None:
                err_obj = {
                    "qid": qid,
                    "judge": {
                        "name": cfg.judge_name,
                        "replica": replica,
                        "temperature": cfg.temperature,
                        "max_output_tokens": cfg.max_output_tokens,
                    },
                    "meta": {
                        "created_at_unix": int(time.time()),
                        "input_messages_count": len(messages),
                        "schema_mode": schema_mode,
                        "attempts": cfg.max_retries + 1,
                    },
                    "status": "ERROR",
                    "raw_response_text": raw_text,
                    "parsed_json": None,
                    "error": str(last_err),
                }
                _write_json(out_path, err_obj, overwrite=True)
                written += 1
                if args.log_every > 0 and (prompt_idx == 1 or prompt_idx % args.log_every == 0):
                    print(f"[DONE] {qid} replica={replica} -> ERROR", flush=True)

    print(f"Outputs written: {written} (overwrite={cfg.overwrite}, dry_run={cfg.dry_run})", flush=True)
    # Print a stable output root hint.
    print(f"Output root: {(cfg.output_root / cfg.group).resolve()}", flush=True)


if __name__ == "__main__":
    main()
