# tools/fix_submission.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _extract_questions(raw: Any) -> List[Dict[str, Any]]:
    """
    Supports common formats:
    - list[{"question_text": "...", ...}, ...]
    - {"questions": [ ... ]}
    """
    if isinstance(raw, list):
        return [q for q in raw if isinstance(q, dict)]
    if isinstance(raw, dict):
        q = raw.get("questions")
        if isinstance(q, list):
            return [x for x in q if isinstance(x, dict)]
    raise ValueError("Unsupported questions.json format. Expected list[...] or {'questions': [...]}.")


def _normalize_top_level(sub: Dict[str, Any]) -> Dict[str, Any]:
    # Server expects team_email (your first error: Field required team_email)
    if "team_email" not in sub:
        if "email" in sub:
            sub["team_email"] = sub.pop("email")
    return sub


def _clean_nones(obj: Any) -> Any:
    """Recursively remove keys with None values from dicts."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if v is None:
                continue
            out[k] = _clean_nones(v)
        return out
    if isinstance(obj, list):
        return [_clean_nones(x) for x in obj]
    return obj


def _coerce_reference(ref: Any) -> Dict[str, Any]:
    if not isinstance(ref, dict):
        return {}

    pdf_sha1 = ref.get("pdf_sha1")
    page_index = ref.get("page_index")

    out: Dict[str, Any] = {}
    if isinstance(pdf_sha1, str) and pdf_sha1:
        out["pdf_sha1"] = pdf_sha1

    # page_index must be int (no floats/strings)
    if isinstance(page_index, bool):
        # bool is subclass of int; ignore
        pass
    elif isinstance(page_index, int):
        out["page_index"] = page_index
    else:
        # try to parse numeric string
        if isinstance(page_index, str):
            try:
                out["page_index"] = int(page_index.strip())
            except Exception:
                pass

    return out


def _normalize_answers(
    answers: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    issues: List[str] = []
    out_answers: List[Dict[str, Any]] = []

    if len(answers) != len(questions):
        issues.append(
            f"answers count ({len(answers)}) != questions count ({len(questions)}). "
            f"This may indicate misalignment."
        )

    n = min(len(answers), len(questions))
    for i in range(n):
        a = answers[i] if isinstance(answers[i], dict) else {}
        q = questions[i]

        q_text = (
            q.get("question_text")
            or q.get("question")
            or q.get("text")
            or q.get("query")
        )

        # Fill question_text if missing/None/empty
        if not a.get("question_text"):
            if isinstance(q_text, str) and q_text.strip():
                a["question_text"] = q_text.strip()
            else:
                issues.append(f"Question {i}: cannot find question text in questions.json item.")

        # Optional: fill kind if you want (only if present in questions.json)
        if not a.get("kind"):
            q_kind = q.get("kind") or q.get("answer_kind")
            if isinstance(q_kind, str) and q_kind.strip():
                a["kind"] = q_kind.strip()

        # Normalize references
        refs = a.get("references", [])
        if not isinstance(refs, list):
            refs = []
        norm_refs = []
        for r in refs:
            rr = _coerce_reference(r)
            if rr.get("pdf_sha1") is not None and rr.get("page_index") is not None:
                norm_refs.append(rr)
        a["references"] = norm_refs

        out_answers.append(a)

    # If answers longer than questions, keep tail but warn
    if len(answers) > n:
        issues.append("answers has extra items beyond questions length; extra answers were dropped.")
    # If questions longer, keep missing answers as-is? Better warn only.
    if len(questions) > n:
        issues.append("questions has extra items beyond answers length; missing answers are not created.")

    return out_answers, issues


def fix_submission(submission_path: Path, questions_path: Path, out_path: Path | None = None) -> None:
    sub_raw = _load_json(submission_path)
    if not isinstance(sub_raw, dict):
        raise ValueError("Submission JSON must be an object at top-level.")

    questions_raw = _load_json(questions_path)
    questions = _extract_questions(questions_raw)

    sub = _normalize_top_level(sub_raw)

    if "submission_name" not in sub or not isinstance(sub["submission_name"], str):
        raise ValueError("submission_name is required and must be a string.")

    if "team_email" not in sub or not isinstance(sub["team_email"], str):
        raise ValueError("team_email is required and must be a string.")

    answers = sub.get("answers")
    if not isinstance(answers, list):
        raise ValueError("answers must be a list.")

    answers_norm, issues = _normalize_answers(answers, questions)
    sub["answers"] = answers_norm

    # Remove None keys recursively (so question_text is never null)
    sub = _clean_nones(sub)

    # Save
    target = out_path or submission_path
    _dump_json(target, sub)

    if issues:
        print("Fixed with warnings:")
        for x in issues:
            print(" -", x)
    else:
        print("Fixed: OK")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True, help="Path to submission_*.json")
    p.add_argument("--questions", dest="questions_path", default="data/questions.json", help="Path to questions.json")
    p.add_argument("--out", dest="out_path", default="", help="Output path (optional). If empty, overwrites input.")
    args = p.parse_args()

    in_path = Path(args.in_path)
    q_path = Path(args.questions_path)
    out_path = Path(args.out_path) if args.out_path.strip() else None

    fix_submission(in_path, q_path, out_path)


if __name__ == "__main__":
    main()
