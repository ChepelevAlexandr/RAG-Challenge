import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.pdf_index import PdfIndex
from src.gigachat_client import GigaChatClient


@dataclass
class Reference:
    pdf_sha1: str
    page_index: int


def _normalize_value_for_schema(value: Any, kind: str) -> Any:
    """
    Приводим ответ к типу, который ожидается для конкретного вопроса.
    """
    kind = (kind or "").strip().lower()

    if kind == "boolean":
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in {"true", "yes", "1"}:
            return True
        if s in {"false", "no", "0"}:
            return False
        return False

    if kind == "number":
        if value is None:
            return "N/A"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return "N/A"
            return float(value)
        s = str(value).strip()
        if not s or s.lower() in {"n/a", "na", "none", "null"}:
            return "N/A"
        m = re.search(r"[-+]?\d[\d,]*\.?\d*", s.replace(" ", ""))
        if not m:
            return "N/A"
        num = m.group(0).replace(",", "")
        try:
            return float(num)
        except Exception:
            return "N/A"

    # string / name / etc.
    if value is None:
        return "N/A"
    s = str(value).strip()
    if not s or s.lower() in {"n/a", "na", "none", "null"}:
        return "N/A"
    return s


def _dedup_refs(refs: List[Reference]) -> List[Reference]:
    seen = set()
    out: List[Reference] = []
    for r in refs:
        key = (r.pdf_sha1, int(r.page_index))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _build_context(index: PdfIndex, query: str, top_k_passages: int = 6) -> Tuple[str, List[Reference]]:
    hits = index.search(query, top_k=top_k_passages)

    refs: List[Reference] = []
    blocks: List[str] = []
    for passage, score in hits:
        refs.append(Reference(pdf_sha1=passage.pdf_sha1, page_index=int(passage.page_index)))
        blocks.append(
            f"[{passage.pdf_sha1} | page {passage.page_index + 1} | score={score:.3f}]\n{passage.text}"
        )

    return "\n\n---\n\n".join(blocks), _dedup_refs(refs)


def answer_question(
    index: PdfIndex,
    question_text: str,
    kind: str,
    top_k_refs: int = 3,
    top_k_passages: int = 6,
    gigachat_api_base: Optional[str] = None,
) -> Tuple[Any, List[Reference]]:
    context, refs_all = _build_context(index, question_text, top_k_passages=top_k_passages)
    refs = refs_all[: max(0, int(top_k_refs))]

    client = GigaChatClient(api_base=(gigachat_api_base or "").strip() or "https://gigachat.devices.sberbank.ru/api/v1")

    prompt = (
        "You are a precise assistant for annual reports QA.\n"
        "Answer the question using ONLY the provided context.\n"
        "Return ONLY the final answer, no explanations.\n\n"
        f"QUESTION:\n{question_text}\n\n"
        f"CONTEXT:\n{context}\n"
    )

    raw = client.ask(prompt).strip()
    if not raw:
        if (kind or "").lower() == "boolean" and "return false" in question_text.lower():
            return False, refs
        return "N/A", refs

    value = _normalize_value_for_schema(raw, kind)
    return value, refs


def build_submission(
    index: PdfIndex,
    questions_path: Path,
    team_email: str,
    submission_name: str,
    top_k_refs: int = 3,
    top_k_passages: int = 6,
    gigachat_api_base: Optional[str] = None,
) -> Dict[str, Any]:
    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))

    answers_out: List[Dict[str, Any]] = []
    for q in questions:
        q_text = q.get("text") or q.get("question_text") or ""
        q_kind = q.get("kind") or "string"

        value, refs = answer_question(
            index=index,
            question_text=q_text,
            kind=q_kind,
            top_k_refs=top_k_refs,
            top_k_passages=top_k_passages,
            gigachat_api_base=gigachat_api_base,
        )

        answers_out.append(
            {
                "value": value,
                "references": [{"pdf_sha1": r.pdf_sha1, "page_index": int(r.page_index)} for r in refs],
            }
        )

    # ВАЖНО: сервер требует team_email
    return {
        "team_email": team_email,
        "submission_name": submission_name,
        "answers": answers_out,
    }
