#!/usr/bin/env python3
"""Build LLaMA-Factory DPO pairs from the provided on-policy VQA answers.

The script only consumes `answers.jsonl` plus the image paths referenced by those
answer records. It never reads AMBER evaluation assets. Use `--judge-mode mock`
for schema/smoke tests without API credentials.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import mimetypes
import os
from io import BytesIO
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_JUDGE_MODEL = "gpt-5.4-mini"
DEFAULT_REFINE_MODEL = "gpt-5.5"
DEFAULT_MAX_ACCEPTED = 1000
DEFAULT_CONFIDENCE_THRESHOLD = 0.70
DEFAULT_REFINE_THRESHOLD = 0.78
DEFAULT_MIN_SCORE_GAP = 1.0
AMBER_MARKERS = ("llava/data/eval/amber", "amber_gen", "amber_dis")


@dataclass(frozen=True)
class Candidate:
    question_id: str
    image_path: str
    prompt: str
    text: str
    answer_id: str
    sample_idx: int | None


@dataclass(frozen=True)
class QuestionGroup:
    question_id: str
    image_path: str
    prompt: str
    candidates: list[Candidate]


@dataclass
class JudgeDecision:
    chosen_answer_id: str
    rejected_answer_id: str
    confidence: float
    candidate_scores: list[dict[str, Any]]
    reasoning: str
    needs_refinement: bool
    raw_response: dict[str, Any]


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE lines before argparse reads environment defaults."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def parse_args() -> argparse.Namespace:
    load_env_file(Path(os.environ.get("DPO_ENV_FILE", ".env")))
    parser = argparse.ArgumentParser(
        description="Create LLaMA-Factory DPO JSON from answers.jsonl using an auditable multimodal judge."
    )
    parser.add_argument("--answers", default=env_str("DPO_ANSWERS_PATH", "answers.jsonl"), help="Input answers.jsonl path.")
    parser.add_argument("--output", default=env_str("DPO_OUTPUT_PATH", "data/processed/qwen2_5_vl_dpo_local_1000.json"), help="Output LLaMA-Factory DPO JSON path.")
    parser.add_argument("--audit", default=env_str("DPO_AUDIT_PATH", "data/audit/qwen2_5_vl_dpo_local_1000.audit.jsonl"), help="Audit JSONL path.")
    parser.add_argument("--sample-output", default=None, help="Optional extra small sample JSON copied from accepted rows.")
    parser.add_argument("--sample-size", type=int, default=5, help="Rows to write to --sample-output.")
    parser.add_argument("--image-root", default=".", help="Root used to resolve relative image paths for validation/API upload.")
    parser.add_argument("--judge-mode", choices=("api", "mock"), default=env_str("DPO_JUDGE_MODE", "api"), help="Use real OpenAI-compatible API or deterministic local mock judge.")
    parser.add_argument("--judge-model", default=env_str("DPO_JUDGE_MODEL", DEFAULT_JUDGE_MODEL), help="Primary multimodal judge model.")
    parser.add_argument("--refine-model", default=env_str("DPO_REFINE_MODEL", DEFAULT_REFINE_MODEL), help="Optional refinement model.")
    parser.add_argument("--enable-refine", action="store_true", default=env_bool("DPO_ENABLE_REFINE", False), help="Use refine model for low-confidence/inconsistent decisions.")
    parser.add_argument("--max-accepted", type=int, default=env_int("DPO_MAX_ACCEPTED", DEFAULT_MAX_ACCEPTED), help="Maximum accepted DPO pairs to output; default is 1000.")
    parser.add_argument("--max-questions", type=int, default=env_int("DPO_MAX_QUESTIONS", 0), help="Maximum question groups to attempt before stopping; 0 means no cap.")
    parser.add_argument("--seed", type=int, default=env_int("DPO_SEED", 0), help="Deterministic shuffle seed.")
    parser.add_argument("--confidence-threshold", type=float, default=env_float("DPO_CONFIDENCE_THRESHOLD", DEFAULT_CONFIDENCE_THRESHOLD), help="Minimum confidence for an accepted pair.")
    parser.add_argument("--refine-threshold", type=float, default=env_float("DPO_REFINE_THRESHOLD", DEFAULT_REFINE_THRESHOLD), help="Refine decisions below this confidence when refinement is enabled.")
    parser.add_argument("--min-score-gap", type=float, default=env_float("DPO_MIN_SCORE_GAP", DEFAULT_MIN_SCORE_GAP), help="Minimum chosen-vs-rejected score gap when scores are supplied.")
    parser.add_argument("--request-timeout", type=float, default=env_float("DPO_REQUEST_TIMEOUT", 120.0), help="API request timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=env_int("DPO_MAX_RETRIES", 3), help="API retries per judge request.")
    parser.add_argument("--workers", type=int, default=env_int("DPO_WORKERS", 1), help="Number of parallel question-level judge workers.")
    parser.add_argument("--max-image-side", type=int, default=env_int("DPO_MAX_IMAGE_SIDE", 0), help="Resize API-uploaded images so the longest side is at most this many pixels; 0 keeps originals.")
    parser.add_argument("--image-jpeg-quality", type=int, default=env_int("DPO_IMAGE_JPEG_QUALITY", 85), help="JPEG quality used when --max-image-side resizes API images.")
    parser.add_argument("--resume", action="store_true", help="Resume by skipping question IDs already present in the audit file.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output/audit instead of resuming.")
    parser.add_argument("--require-exactly-ten-candidates", action="store_true", help="Skip groups that do not have exactly 10 candidates.")
    parser.add_argument("--dry-run", action="store_true", help="Validate/group input and print a summary without judging or writing outputs.")
    return parser.parse_args()


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield value


def normalize_rel_path(raw_path: str) -> str:
    normalized = Path(raw_path).as_posix().lstrip("/")
    lowered = normalized.lower()
    if any(marker in lowered for marker in AMBER_MARKERS):
        raise ValueError(f"AMBER path is not allowed in training data: {raw_path}")
    if ".." in Path(normalized).parts:
        raise ValueError(f"Parent-directory image path is not allowed: {raw_path}")
    return normalized


def candidate_from_record(record: dict[str, Any]) -> Candidate:
    required = ("question_id", "image_path", "prompt", "text", "answer_id")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"Answer record missing fields {missing}: {record}")
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    sample_idx = metadata.get("sample_idx")
    return Candidate(
        question_id=str(record["question_id"]),
        image_path=normalize_rel_path(str(record["image_path"])),
        prompt=str(record["prompt"]),
        text=str(record["text"]),
        answer_id=str(record["answer_id"]),
        sample_idx=int(sample_idx) if isinstance(sample_idx, int) else None,
    )


def read_groups(answers_path: Path) -> dict[str, list[Candidate]]:
    groups: dict[str, list[Candidate]] = {}
    for record in load_jsonl(answers_path):
        candidate = candidate_from_record(record)
        groups.setdefault(candidate.question_id, []).append(candidate)
    return groups


def validate_group(
    question_id: str,
    candidates: list[Candidate],
    image_root: Path,
    require_exactly_ten: bool,
) -> tuple[QuestionGroup | None, str | None]:
    if not candidates:
        return None, "empty_group"
    image_paths = {candidate.image_path for candidate in candidates}
    prompts = {candidate.prompt for candidate in candidates}
    answer_ids = [candidate.answer_id for candidate in candidates]
    if len(image_paths) != 1:
        return None, "multiple_image_paths"
    if len(prompts) != 1:
        return None, "multiple_prompts"
    if len(answer_ids) != len(set(answer_ids)):
        return None, "duplicate_answer_ids"
    if require_exactly_ten and len(candidates) != 10:
        return None, f"candidate_count_{len(candidates)}"
    image_path = next(iter(image_paths))
    resolved_image = image_root / image_path
    if not resolved_image.is_file():
        return None, "missing_image"
    return QuestionGroup(question_id=question_id, image_path=image_path, prompt=next(iter(prompts)), candidates=candidates), None


def build_dpo_row(group: QuestionGroup, decision: JudgeDecision) -> dict[str, Any]:
    by_id = {candidate.answer_id: candidate for candidate in group.candidates}
    chosen = by_id[decision.chosen_answer_id]
    rejected = by_id[decision.rejected_answer_id]
    return {
        "instruction": f"<image>{group.prompt}",
        "input": "",
        "chosen": chosen.text,
        "rejected": rejected.text,
        "images": [group.image_path],
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    tmp_path.replace(path)


def append_audit(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_resume_state(audit_path: Path, output_path: Path) -> tuple[set[str], list[dict[str, Any]]]:
    completed_question_ids: set[str] = set()
    accepted_rows: list[dict[str, Any]] = []
    if output_path.is_file():
        with output_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if not isinstance(existing, list):
            raise ValueError(f"Existing output is not a JSON array: {output_path}")
        accepted_rows = [row for row in existing if isinstance(row, dict)]
    if audit_path.is_file():
        for record in load_jsonl(audit_path):
            question_id = record.get("question_id")
            if question_id:
                completed_question_ids.add(str(question_id))
            if not output_path.is_file() and record.get("status") == "accepted" and isinstance(record.get("dpo_row"), dict):
                accepted_rows.append(record["dpo_row"])
    return completed_question_ids, accepted_rows


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fenced_match:
        stripped = fenced_match.group(1)
    else:
        first = stripped.find("{")
        last = stripped.rfind("}")
        if first != -1 and last != -1 and last > first:
            stripped = stripped[first : last + 1]
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("Judge response JSON must be an object")
    return parsed


def resolve_answer_id(value: Any, group: QuestionGroup, field_name: str) -> str:
    raw = str(value).strip()
    valid_ids = {candidate.answer_id for candidate in group.candidates}
    if raw in valid_ids:
        return raw
    match = re.fullmatch(r"(?:candidate|answer|option|#)?\s*(\d+)", raw, flags=re.IGNORECASE)
    if match:
        index = int(match.group(1))
        if 1 <= index <= len(group.candidates):
            return group.candidates[index - 1].answer_id
        if 0 <= index < len(group.candidates):
            return group.candidates[index].answer_id
    raise ValueError(f"{field_name} is not one of the candidates: {raw}")


def parse_judge_decision(payload: dict[str, Any], group: QuestionGroup) -> JudgeDecision:
    chosen = resolve_answer_id(payload.get("chosen_answer_id", payload.get("chosen", "")), group, "chosen_answer_id")
    rejected = resolve_answer_id(payload.get("rejected_answer_id", payload.get("rejected", "")), group, "rejected_answer_id")
    if chosen == rejected:
        raise ValueError("chosen_answer_id and rejected_answer_id must differ")
    confidence = float(payload.get("confidence", 0.0))
    if confidence < 0 or confidence > 1:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
    candidate_scores = payload.get("candidate_scores", payload.get("scores", []))
    if isinstance(candidate_scores, dict):
        candidate_scores = [
            {"answer_id": answer_id, **(score if isinstance(score, dict) else {"score": score})}
            for answer_id, score in candidate_scores.items()
        ]
    if not isinstance(candidate_scores, list):
        candidate_scores = []
    return JudgeDecision(
        chosen_answer_id=chosen,
        rejected_answer_id=rejected,
        confidence=confidence,
        candidate_scores=[score for score in candidate_scores if isinstance(score, dict)],
        reasoning=str(payload.get("reasoning", payload.get("reason", ""))),
        needs_refinement=bool(payload.get("needs_refinement", False)),
        raw_response=payload,
    )


def score_gap(decision: JudgeDecision) -> float | None:
    scores: dict[str, float] = {}
    for item in decision.candidate_scores:
        answer_id = item.get("answer_id")
        if answer_id is None:
            continue
        try:
            scores[str(answer_id)] = float(item.get("score"))
        except (TypeError, ValueError):
            continue
    if decision.chosen_answer_id in scores and decision.rejected_answer_id in scores:
        return scores[decision.chosen_answer_id] - scores[decision.rejected_answer_id]
    return None


def decision_needs_refinement(decision: JudgeDecision, refine_threshold: float, min_score_gap: float) -> bool:
    gap = score_gap(decision)
    return decision.needs_refinement or decision.confidence < refine_threshold or (gap is not None and gap < min_score_gap)


def mock_judge(group: QuestionGroup, model: str, refined: bool = False) -> JudgeDecision:
    """Deterministic local judge for smoke tests; not a quality labeler."""
    ordered = sorted(group.candidates, key=lambda item: (len(item.text), item.answer_id))
    rejected = ordered[0]
    chosen = ordered[-1]
    max_len = max(len(chosen.text), 1)
    gap = max(1, len(chosen.text) - len(rejected.text))
    confidence = min(0.95, 0.72 + gap / max_len * 0.2)
    payload = {
        "chosen_answer_id": chosen.answer_id,
        "rejected_answer_id": rejected.answer_id,
        "confidence": confidence,
        "candidate_scores": [
            {
                "answer_id": candidate.answer_id,
                "score": round(1.0 + 9.0 * len(candidate.text) / max_len, 3),
                "reason": "Mock score based only on answer length for no-API schema testing.",
            }
            for candidate in group.candidates
        ],
        "reasoning": f"Mock {model} decision for smoke testing; do not use as training labels.",
        "needs_refinement": False,
        "refined": refined,
    }
    return parse_judge_decision(payload, group)


def image_data_url(image_path: Path, max_side: int = 0, jpeg_quality: int = 85) -> str:
    if max_side > 0:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for --max-image-side image resizing.") from exc
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image.thumbnail((max_side, max_side))
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
            data = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{data}"
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None:
        mime_type = "image/jpeg"
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"


def judge_prompt(group: QuestionGroup, refined: bool) -> str:
    candidates_text = "\n".join(
        f"[{idx}] answer_id={candidate.answer_id}\n{candidate.text}"
        for idx, candidate in enumerate(group.candidates, start=1)
    )
    refinement_note = " This is a refinement pass; be stricter about uncertainty and visual grounding." if refined else ""
    return f"""You are judging candidate answers for a multimodal VQA DPO dataset.{refinement_note}

Use only the provided image, question, and candidate answers. Do not use AMBER data or any external benchmark content.
Select the answer that is most factually grounded in the image as chosen_answer_id and the answer with the highest hallucination risk as rejected_answer_id. Prefer visual correctness over style or verbosity.

Question: {group.prompt}

Candidate answers:
{candidates_text}

Return a single JSON object with this schema. `score` is a 0-10 visual-grounding score where higher means more faithful to the image/question and lower means higher hallucination risk:
{{
  "chosen_answer_id": "string",
  "rejected_answer_id": "string",
  "confidence": 0.0,
  "candidate_scores": [{{"answer_id": "string", "score": 0.0, "reason": "short reason"}}],
  "reasoning": "short audit explanation",
  "needs_refinement": false
}}
"""


def api_judge(
    group: QuestionGroup,
    image_root: Path,
    model: str,
    timeout: float,
    max_retries: int,
    max_image_side: int,
    image_jpeg_quality: int,
    refined: bool = False,
) -> JudgeDecision:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is required for --judge-mode api. Use pixi or --judge-mode mock.") from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for --judge-mode api")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not base_url or "example.com" in base_url:
        raise RuntimeError("Set OPENAI_BASE_URL in the environment or .env before using --judge-mode api")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    messages = [
        {
            "role": "system",
            "content": "You are a careful multimodal preference judge. Return valid JSON only.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": judge_prompt(group, refined=refined)},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url(
                            image_root / group.image_path,
                            max_side=max_image_side,
                            jpeg_quality=image_jpeg_quality,
                        )
                    },
                },
            ],
        },
    ]
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty judge response")
            return parse_judge_decision(extract_json_object(content), group)
        except Exception as exc:  # noqa: BLE001 - audit and retry arbitrary API/parser failures.
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Judge request failed after {max_retries} attempts: {last_error}")


def run_judge(args: argparse.Namespace, group: QuestionGroup, refined: bool = False) -> JudgeDecision:
    model = args.refine_model if refined else args.judge_model
    if args.judge_mode == "mock":
        return mock_judge(group, model=model, refined=refined)
    return api_judge(
        group=group,
        image_root=Path(args.image_root),
        model=model,
        timeout=args.request_timeout,
        max_retries=args.max_retries,
        max_image_side=args.max_image_side,
        image_jpeg_quality=args.image_jpeg_quality,
        refined=refined,
    )


def audit_base(group: QuestionGroup) -> dict[str, Any]:
    return {
        "question_id": group.question_id,
        "image_path": group.image_path,
        "prompt": group.prompt,
        "candidate_count": len(group.candidates),
        "candidate_answer_ids": [candidate.answer_id for candidate in group.candidates],
    }


def accepted_audit_record(
    group: QuestionGroup,
    decision: JudgeDecision,
    args: argparse.Namespace,
    refined: bool,
    dpo_row: dict[str, Any],
) -> dict[str, Any]:
    return {
        **audit_base(group),
        "status": "accepted",
        "judge_model": args.judge_model,
        "refine_model": args.refine_model if refined else None,
        "refinement_status": "refined" if refined else "not_refined",
        "chosen_answer_id": decision.chosen_answer_id,
        "rejected_answer_id": decision.rejected_answer_id,
        "confidence": decision.confidence,
        "score_gap": score_gap(decision),
        "candidate_scores": decision.candidate_scores,
        "reasoning": decision.reasoning,
        "raw_response": decision.raw_response,
        "dpo_row": dpo_row,
    }


def failure_audit_record(group: QuestionGroup, args: argparse.Namespace, status: str, error: str, decision: JudgeDecision | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        **audit_base(group),
        "status": status,
        "judge_model": args.judge_model,
        "refine_model": args.refine_model if args.enable_refine else None,
        "error": error,
    }
    if decision is not None:
        record.update(
            {
                "chosen_answer_id": decision.chosen_answer_id,
                "rejected_answer_id": decision.rejected_answer_id,
                "confidence": decision.confidence,
                "score_gap": score_gap(decision),
                "candidate_scores": decision.candidate_scores,
                "reasoning": decision.reasoning,
                "raw_response": decision.raw_response,
            }
        )
    return record


def process_group(args: argparse.Namespace, group: QuestionGroup) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    try:
        decision = run_judge(args, group, refined=False)
        refined = False
        if args.enable_refine and decision_needs_refinement(decision, args.refine_threshold, args.min_score_gap):
            decision = run_judge(args, group, refined=True)
            refined = True
        gap = score_gap(decision)
        if decision.confidence < args.confidence_threshold:
            return None, failure_audit_record(
                group,
                args,
                "skipped_low_confidence",
                f"confidence {decision.confidence:.3f} < threshold {args.confidence_threshold:.3f}",
                decision,
            )
        if gap is not None and gap < args.min_score_gap:
            return None, failure_audit_record(
                group,
                args,
                "skipped_low_score_gap",
                f"score gap {gap:.3f} < threshold {args.min_score_gap:.3f}",
                decision,
            )
        dpo_row = build_dpo_row(group, decision)
        return dpo_row, accepted_audit_record(group, decision, args, refined, dpo_row)
    except Exception as exc:  # noqa: BLE001 - failure must be auditable and resumable.
        return None, failure_audit_record(group, args, "failed", str(exc))


def main() -> int:
    args = parse_args()
    answers_path = Path(args.answers)
    output_path = Path(args.output)
    audit_path = Path(args.audit)
    image_root = Path(args.image_root)

    if args.max_accepted < 1:
        raise ValueError("--max-accepted must be at least 1")
    if not answers_path.is_file():
        raise FileNotFoundError(f"Answers file not found: {answers_path}")
    if args.overwrite:
        for path in (output_path, audit_path):
            if path.exists():
                path.unlink()
    elif not args.resume and (output_path.exists() or audit_path.exists()):
        raise FileExistsError("Output or audit exists. Use --resume or --overwrite.")

    raw_groups = read_groups(answers_path)
    valid_groups: list[QuestionGroup] = []
    validation_failures: list[tuple[str, str]] = []
    for question_id, candidates in raw_groups.items():
        group, error = validate_group(question_id, candidates, image_root, args.require_exactly_ten_candidates)
        if group is None:
            validation_failures.append((question_id, error or "invalid_group"))
        else:
            valid_groups.append(group)

    rng = random.Random(args.seed)
    valid_groups.sort(key=lambda item: item.question_id)
    rng.shuffle(valid_groups)
    if args.max_questions > 0:
        valid_groups = valid_groups[: args.max_questions]

    print(
        f"Loaded {len(raw_groups)} question groups; {len(valid_groups)} valid selected; "
        f"{len(validation_failures)} validation failures.",
        file=sys.stderr,
    )
    if args.dry_run:
        return 0

    completed_ids: set[str] = set()
    accepted_rows: list[dict[str, Any]] = []
    if args.resume and not args.overwrite:
        completed_ids, accepted_rows = load_resume_state(audit_path, output_path)
        print(f"Resume state: {len(completed_ids)} audited questions, {len(accepted_rows)} accepted rows.", file=sys.stderr)

    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    attempted = 0
    groups_to_process = [group for group in valid_groups if group.question_id not in completed_ids]

    def handle_result(group: QuestionGroup, dpo_row: dict[str, Any] | None, audit_record: dict[str, Any]) -> None:
        nonlocal accepted_rows
        append_audit(audit_path, audit_record)
        if audit_record.get("status") == "failed":
            print(f"Failed question_id={group.question_id}: {audit_record.get('error')}", file=sys.stderr)
        if dpo_row is not None and len(accepted_rows) < args.max_accepted:
            accepted_rows.append(dpo_row)
            write_json(output_path, accepted_rows)

    if args.workers == 1:
        for group in groups_to_process:
            if len(accepted_rows) >= args.max_accepted:
                break
            attempted += 1
            dpo_row, audit_record = process_group(args, group)
            handle_result(group, dpo_row, audit_record)
    else:
        print(f"Using {args.workers} parallel judge workers.", file=sys.stderr)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
        futures: dict[concurrent.futures.Future[tuple[dict[str, Any] | None, dict[str, Any]]], QuestionGroup] = {}
        group_iter = iter(groups_to_process)
        try:
            while len(futures) < args.workers and len(accepted_rows) < args.max_accepted:
                group = next(group_iter)
                futures[executor.submit(process_group, args, group)] = group
                attempted += 1
            while futures and len(accepted_rows) < args.max_accepted:
                done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    group = futures.pop(future)
                    dpo_row, audit_record = future.result()
                    handle_result(group, dpo_row, audit_record)
                while len(futures) < args.workers and len(accepted_rows) < args.max_accepted:
                    try:
                        group = next(group_iter)
                    except StopIteration:
                        break
                    futures[executor.submit(process_group, args, group)] = group
                    attempted += 1
        except StopIteration:
            pass
        finally:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)

    if accepted_rows:
        write_json(output_path, accepted_rows[: args.max_accepted])
    else:
        write_json(output_path, [])
    if args.sample_output:
        write_json(Path(args.sample_output), accepted_rows[: args.sample_size])

    print(
        f"Attempted {attempted} new questions; wrote {min(len(accepted_rows), args.max_accepted)} accepted DPO rows to {output_path}.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
