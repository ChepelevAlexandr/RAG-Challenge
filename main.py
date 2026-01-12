import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.pdf_index import PdfIndex
from src.rag_answer import build_submission
from src.submission_api import submit_submission, get_leaderboard


DEFAULT_API_BASE = "http://5.35.3.130:800"


def _pick_file(args) -> str:
    if getattr(args, "file", None):
        return args.file
    if getattr(args, "file_opt", None):
        return args.file_opt
    raise SystemExit("Missing --file (or positional file).")


def build_index(args) -> None:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = PdfIndex.build_from_dir(data_dir)
    index.save(out_dir)

    print(json.dumps({"status": "ok", "index_dir": str(out_dir)}, ensure_ascii=False, indent=2))


def make_submission_cmd(args) -> None:
    index_dir = Path(args.index_dir)
    questions_file = Path(args.questions_file)
    out_file = Path(args.out_file)

    # оставляем флаг --email (как раньше), но кладём в team_email
    team_email = args.email or os.environ.get("TEAM_EMAIL") or os.environ.get("EMAIL")
    submission_name = args.submission_name or os.environ.get("SUBMISSION_NAME")

    if not team_email:
        raise SystemExit("No team email provided. Use --email or set TEAM_EMAIL in .env")
    if not submission_name:
        raise SystemExit("No submission name provided. Use --submission-name or set SUBMISSION_NAME in .env")

    index = PdfIndex.load(index_dir)
    submission = build_submission(
        index=index,
        questions_path=questions_file,
        team_email=team_email,
        submission_name=submission_name,
        top_k_refs=args.top_k_refs,
        top_k_passages=args.top_k_passages,
        gigachat_api_base=args.gigachat_api_base,
    )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "file": str(out_file)}, ensure_ascii=False, indent=2))


def check_cmd(args) -> None:
    file_path = _pick_file(args)
    payload = json.loads(Path(file_path).read_text(encoding="utf-8"))

    issues = []
    if "team_email" not in payload:
        issues.append("Missing team_email at top level (server requires team_email).")
    if "submission_name" not in payload:
        issues.append("Missing submission_name at top level.")
    if "answers" not in payload or not isinstance(payload["answers"], list):
        issues.append("Missing answers list at top level.")
    else:
        for i, a in enumerate(payload["answers"]):
            if not isinstance(a, dict):
                issues.append(f"Answer #{i} is not an object.")
                continue
            if "value" not in a:
                issues.append(f"Answer #{i}: missing value.")
            if "references" not in a or not isinstance(a["references"], list):
                issues.append(f"Answer #{i}: missing references list.")

    if issues:
        print(json.dumps({"status": "issues found", "issues": issues}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"status": "ok"}, ensure_ascii=False, indent=2))


def submit_cmd(args) -> None:
    file_path = _pick_file(args)
    resp = submit_submission(file_path, api_base=args.api_base)
    print(json.dumps(resp, ensure_ascii=False, indent=2))


def leaderboard_cmd(args) -> None:
    resp = get_leaderboard(api_base=args.api_base)
    if isinstance(resp, dict) and resp.get("status") == "error":
        print(json.dumps(resp, ensure_ascii=False, indent=2))
        return

    rows = resp
    if not rows:
        print(json.dumps({"status": "ok", "rows": 0}, ensure_ascii=False, indent=2))
        return

    def to_float(x):
        try:
            return float(x)
        except Exception:
            return float("-inf")

    rows_sorted = sorted(rows, key=lambda r: to_float(r.get("Score")), reverse=True)

    top_n = args.top
    out = rows_sorted[:top_n]

    print(f"Leaderboard (sorted by Score), top {top_n}:")
    for r in out:
        rank = r.get("rank", "?")
        team = r.get("team", "?")
        score = r.get("Score", "?")
        val = r.get("Val Accuracy", r.get("Val_Accuracy", r.get("ValAccuracy", "?")))
        sig = r.get("signature", "")
        print(f"{rank:>4} | {team:<24} | Score={score:<6} | ValAcc={val:<8} | sig={sig}")


def main():
    load_dotenv()

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_bi = sub.add_parser("build-index")
    p_bi.add_argument("--data-dir", required=True)
    p_bi.add_argument("--out-dir", required=True)
    p_bi.set_defaults(func=build_index)

    p_ms = sub.add_parser("make-submission")
    p_ms.add_argument("--index-dir", required=True)
    p_ms.add_argument("--questions-file", required=True)
    p_ms.add_argument("--email", default=None)  # станет team_email в JSON
    p_ms.add_argument("--submission-name", default=None)
    p_ms.add_argument("--out-file", required=True)
    p_ms.add_argument("--top-k-refs", type=int, default=3)
    p_ms.add_argument("--top-k-passages", type=int, default=6)
    p_ms.add_argument("--gigachat-api-base", default="https://gigachat.devices.sberbank.ru/api/v1")
    p_ms.set_defaults(func=make_submission_cmd)

    p_ck = sub.add_parser("check")
    p_ck.add_argument("file_opt", nargs="?", default=None)
    p_ck.add_argument("--file", dest="file", default=None)
    p_ck.set_defaults(func=check_cmd)

    p_sb = sub.add_parser("submit")
    p_sb.add_argument("file_opt", nargs="?", default=None)
    p_sb.add_argument("--file", dest="file", default=None)
    p_sb.add_argument("--api-base", default=DEFAULT_API_BASE)
    p_sb.set_defaults(func=submit_cmd)

    p_lb = sub.add_parser("leaderboard")
    p_lb.add_argument("--api-base", default=DEFAULT_API_BASE)
    p_lb.add_argument("--top", type=int, default=30)
    p_lb.set_defaults(func=leaderboard_cmd)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
