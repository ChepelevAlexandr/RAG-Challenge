# -*- coding: utf-8 -*-
"""
PDF parsing + hybrid retrieval index.

Dependencies:
- PyMuPDF (fitz) for text extraction
- rank-bm25 for BM25
- scikit-learn for TF-IDF (recommended; otherwise BM25-only fallback)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hashlib
import json
import pickle
import re

import fitz  # PyMuPDF
from rank_bm25 import BM25Okapi

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:  # pragma: no cover
    np = None  # type: ignore
    TfidfVectorizer = None  # type: ignore


_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[.,][0-9]+)?")
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")


def compute_sha1(path: Path) -> str:
    """
    If filename stem already looks like sha1 (40 hex chars) — use it.
    Otherwise compute sha1 of the file bytes.
    """
    stem = path.stem.lower()
    if _HEX40_RE.match(stem):
        return stem

    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean_text(text: str) -> str:
    # Normalize hyphenation across line breaks, multiple spaces, etc.
    text = text.replace("\u00ad", "")  # soft hyphen
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    text = text.lower()
    return _WORD_RE.findall(text)


@dataclass(frozen=True)
class PdfPage:
    pdf_sha1: str
    page_index: int
    text: str


class HybridRetriever:
    """
    BM25 + optional TF-IDF (word + char_wb).
    """

    def __init__(
        self,
        *,
        bm25: BM25Okapi,
        tokenized_pages: List[List[str]],
        tfidf_word: Any = None,
        tfidf_char: Any = None,
        X_word: Any = None,
        X_char: Any = None,
    ) -> None:
        self._bm25 = bm25
        self._tokenized_pages = tokenized_pages

        self._tfidf_word = tfidf_word
        self._tfidf_char = tfidf_char
        self._X_word = X_word
        self._X_char = X_char

    @staticmethod
    def build(pages: List[PdfPage]) -> "HybridRetriever":
        tokenized = [tokenize(p.text) for p in pages]
        bm25 = BM25Okapi(tokenized)

        # TF-IDF optional
        if TfidfVectorizer is None or np is None:
            return HybridRetriever(bm25=bm25, tokenized_pages=tokenized)

        texts = [p.text for p in pages]

        tfidf_word = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            token_pattern=r"(?u)\b\w+\b",
        )
        X_word = tfidf_word.fit_transform(texts)

        tfidf_char = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.98,
        )
        X_char = tfidf_char.fit_transform(texts)

        return HybridRetriever(
            bm25=bm25,
            tokenized_pages=tokenized,
            tfidf_word=tfidf_word,
            tfidf_char=tfidf_char,
            X_word=X_word,
            X_char=X_char,
        )

    def search(self, query: str, *, top_k: int = 40) -> List[Tuple[int, float]]:
        """
        Returns list of (page_idx_in_corpus, score) sorted desc.
        """
        q_tokens = tokenize(query)
        bm = self._bm25.get_scores(q_tokens)

        def _norm(arr):
            if np is None:
                mx = max(arr)
                mn = min(arr)
                if mx <= mn:
                    return [0.0 for _ in arr]
                return [(float(x) - mn) / (mx - mn) for x in arr]

            arr = np.asarray(arr, dtype=float)
            mx = float(arr.max())
            mn = float(arr.min())
            if mx <= mn:
                return np.zeros_like(arr)
            return (arr - mn) / (mx - mn)

        bm_n = _norm(bm)

        # BM25-only fallback
        if self._tfidf_word is None or self._X_word is None or np is None:
            scores = bm_n
            idxs = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:top_k]
            return [(i, float(scores[i])) for i in idxs]

        q_word = self._tfidf_word.transform([query])
        w = (self._X_word @ q_word.T).toarray().ravel()

        q_char = self._tfidf_char.transform([query])
        c = (self._X_char @ q_char.T).toarray().ravel()

        w_n = _norm(w)
        c_n = _norm(c)

        # Weighted fusion (annual reports are table-heavy)
        scores = 0.50 * bm_n + 0.35 * w_n + 0.15 * c_n

        idxs = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in idxs]


class PdfIndex:
    def __init__(self, pages: List[PdfPage], retriever: HybridRetriever) -> None:
        self.pages = pages
        self.retriever = retriever

    @staticmethod
    def parse_pdf(path: Path) -> List[PdfPage]:
        sha1 = compute_sha1(path)
        doc = fitz.open(str(path))
        out: List[PdfPage] = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            text = _clean_text(text)
            out.append(PdfPage(pdf_sha1=sha1, page_index=i, text=text))
        doc.close()
        return out

    @classmethod
    def build_from_dir(cls, data_dir: Path) -> "PdfIndex":
        data_dir = Path(data_dir)
        pdf_paths = sorted([p for p in data_dir.glob("**/*.pdf") if p.is_file()])
        if not pdf_paths:
            raise FileNotFoundError(f"No PDFs found in {data_dir}")

        pages: List[PdfPage] = []
        for p in pdf_paths:
            pages.extend(cls.parse_pdf(p))

        retriever = HybridRetriever.build(pages)
        return cls(pages=pages, retriever=retriever)

    def search(
        self,
        query: str,
        *,
        top_k: int = 18,
        per_pdf_limit: int = 4,
        prefetch_k: int = 60,
    ) -> List[Tuple[PdfPage, float]]:
        """
        Get top_k pages with a soft cap per pdf to improve multi-company questions.
        """
        hits = self.retriever.search(query, top_k=prefetch_k)
        out: List[Tuple[PdfPage, float]] = []
        counts: Dict[str, int] = {}

        for idx, score in hits:
            page = self.pages[idx]
            cnt = counts.get(page.pdf_sha1, 0)
            if cnt >= per_pdf_limit:
                continue
            counts[page.pdf_sha1] = cnt + 1
            out.append((page, score))
            if len(out) >= top_k:
                break

        return out

    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        pages_path = out_dir / "pages.jsonl"
        with pages_path.open("w", encoding="utf-8") as f:
            for p in self.pages:
                f.write(
                    json.dumps(
                        {"pdf_sha1": p.pdf_sha1, "page_index": p.page_index, "text": p.text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        idx_path = out_dir / "index.pkl"
        with idx_path.open("wb") as f:
            pickle.dump(self.retriever, f)

    @classmethod
    def load(cls, index_dir: Path) -> "PdfIndex":
        index_dir = Path(index_dir)
        pages_path = index_dir / "pages.jsonl"
        idx_path = index_dir / "index.pkl"
        if not pages_path.exists() or not idx_path.exists():
            raise FileNotFoundError(
                f"Index not found in {index_dir} (expected pages.jsonl and index.pkl)"
            )

        pages: List[PdfPage] = []
        with pages_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                pages.append(
                    PdfPage(
                        pdf_sha1=obj["pdf_sha1"],
                        page_index=int(obj["page_index"]),
                        text=obj["text"],
                    )
                )

        with idx_path.open("rb") as f:
            retriever = pickle.load(f)

        return cls(pages=pages, retriever=retriever)
