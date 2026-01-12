# -*- coding: utf-8 -*-
"""
Minimal HTTP client for the RAG Challenge server.

If you don't need server submission from CLI, you can ignore this module.
"""

from __future__ import annotations

from typing import Any, Dict
import os
import requests


def submit_submission(payload: Dict[str, Any], *, api_base: str) -> Dict[str, Any]:
    """
    POST /submissions
    Auth: Bearer token from env RAG_CHALLENGE_TOKEN (if required).
    """
    token = os.getenv("RAG_CHALLENGE_TOKEN", "").strip()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = api_base.rstrip("/") + "/submissions"
    r = requests.post(url, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()


def get_leaderboard(*, api_base: str) -> Any:
    """
    GET /leaderboard
    """
    token = os.getenv("RAG_CHALLENGE_TOKEN", "").strip()
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = api_base.rstrip("/") + "/leaderboard"
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()
