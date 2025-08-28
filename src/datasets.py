#!/usr/bin/env python3
"""
src/datasets.py
===============

Minimal helper to **download TLE histories from Space‑Track** and
store them under  `data/raw/`.

Usage (CLI)
-----------
# All history for one spacecraft
$ python -m src.datasets fetch 37746

# Last 90 days only
$ python -m src.datasets fetch 37746 --days 90
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

# ────────────────────────────────────────────────────────────
LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
TLE_URL = (
    "https://www.space-track.org/basicspacedata/query/"
    "class/tle/NORAD_CAT_ID/{id}{date_filter}/orderby/EPOCH desc/format/tle"
)

RAW_DIR = Path("data/raw")          # central raw‑data folder
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────── small helpers ──────────────────────────
def _date_filter(days: int | None) -> str:
    """Return URL fragment to filter TLEs newer than <days> ago."""
    if not days:
        return ""
    start = (datetime.utcnow() - timedelta(days=days)).date().isoformat()
    return f"/EPOCH/%3E{start}"      # '%3E' is URL‑encoded '>'


# ─────────────────── public API ─────────────────────────────
def fetch_tle(norad_id: int, days: int | None = None) -> str:
    """
    Download TLE set for a given NORAD catalogue ID.

    Parameters
    ----------
    norad_id : int
        NORAD catalogue number.
    days : int | None
        If provided, limit to last <days> of data.

    Returns
    -------
    str
        Raw multi‑line TLE text (pairs of lines).
    """
    load_dotenv()                                    # read .env once
    user, pwd = os.getenv("ST_USERNAME"), os.getenv("ST_PASSWORD")
    if not (user and pwd):
        sys.exit("❌  Space‑Track credentials missing in .env")

    with requests.Session() as session:
        session.post(LOGIN_URL, data={"identity": user, "password": pwd}).raise_for_status()
        url = TLE_URL.format(id=norad_id, date_filter=_date_filter(days))
        resp = session.get(url)
        resp.raise_for_status()
        return resp.text


def save_tle(tle_text: str, norad_id: int) -> Path:
    """
    Save raw TLE text to `data/raw/` and return the path.

    Filename format:  <NORAD>_tle_<UTC‑timestamp>.txt
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = RAW_DIR / f"{norad_id}_tle_{timestamp}.txt"
    out_path.write_text(tle_text, encoding="utf-8")
    return out_path


# ─────────────────── CLI entry ‑ “fetch” subcommand ────────
def _cli():
    ap = argparse.ArgumentParser(
        prog="python -m src.datasets",
        description="Dataset utilities (currently: fetch TLEs)."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # fetch ----------------------------------------------------------------
    fetch = sub.add_parser("fetch", help="Download TLE history from Space‑Track")
    fetch.add_argument("norad_id", type=int, help="NORAD catalogue number")
    fetch.add_argument(
        "--days", type=int, metavar="N",
        help="Only last N days (omit for full history)"
    )

    args = ap.parse_args()

    if args.cmd == "fetch":
        try:
            tle_txt = fetch_tle(args.norad_id, args.days)
            path = save_tle(tle_txt, args.norad_id)
            print(f"✅  TLE saved → {path}")
        except requests.HTTPError as e:
            sys.exit(f"❌  HTTP error from Space‑Track: {e}")
        except Exception as e:
            sys.exit(f"❌  Unexpected error: {e}")


if __name__ == "__main__":
    _cli()
