from __future__ import annotations

import logging

import httpx

IPINFO_URL = "https://ipinfo.io/json"

logger = logging.getLogger(__name__)


def run_benchmark() -> str | None:
    try:
        resp = httpx.get(IPINFO_URL, timeout=15)
        resp.raise_for_status()
        country = (resp.json().get("country") or "").strip().upper()
    except Exception as e:
        logger.warning(f"localization lookup failed")
        return None
    return country or None
