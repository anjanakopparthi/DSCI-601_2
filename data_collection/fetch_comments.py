"""
YouTube comment collector for domain-adaptive pretraining data.

Collects Tamil / romanized-Tamil YouTube comments on the same topic domains as
the HopeEDI dataset (equality, women in STEM, COVID-19 encouragement) using the
official YouTube Data API v3. The output corpus is UNLABELED — it is used for
domain-adaptive pretraining (MLM) and/or self-training, not directly as
supervised data.

Privacy & compliance:
    - Official Data API only (no HTML scraping).
    - Comments are ANONYMIZED at collection time: author names/ids are never
      written to disk. Only the comment text, like count, and a video id (for
      dedup/provenance) are kept.
    - The output CSV is written to data_collection/collected/ which is
      gitignored — raw comments must not be committed.

Quota:
    - Default daily quota is 10,000 units. search.list costs 100 units/call;
      commentThreads.list costs 1 unit/call (up to 100 comments each).
    - The script tracks units spent and stops at QUOTA_BUDGET.
    - State (seen video ids) is persisted, so rerunning tomorrow resumes
      instead of re-fetching.

Setup:
    $ pip install requests
    .env in the repo root:  YOUTUBE_API_KEY=AIza...

Usage (from anywhere):
    $ python data_collection/fetch_comments.py
    $ python data_collection/fetch_comments.py --max-videos 30 --budget 5000
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / "collected"
STATE_PATH = OUT_DIR / "state.json"
CSV_PATH = OUT_DIR / "tamil_comments_raw.csv"

API_BASE = "https://www.googleapis.com/youtube/v3"
COST_SEARCH = 100
COST_COMMENTS = 1

# Search queries mirroring the HopeEDI topic domains, in Tamil + romanized
# Tamil + English-with-Tamil-audience phrasing. Edit freely.
SEARCH_QUERIES = [
    "நம்பிக்கை motivation tamil",
    "வாழ்க்கை motivation tamil speech",
    "tamil motivation never give up",
    "பெண்கள் சாதனை tamil",                 # women's achievement
    "women in engineering tamil",
    "corona positive வாழ்க்கை tamil",       # covid encouragement
    "tamil students motivation exam",
    "சமத்துவம் பேச்சு tamil",               # equality speech
    "vazhkai nambikkai tamil",              # romanized: life hope
    "tamil inspirational speech",
]


# ============================================================
# 1. Environment / API plumbing
# ============================================================

def load_api_key() -> str:
    """Read YOUTUBE_API_KEY from the repo-root .env (no external deps)."""
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        sys.exit(f".env not found at {env_path} — create it with YOUTUBE_API_KEY=...")
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("YOUTUBE_API_KEY=") and not line.startswith("#"):
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
            if key:
                return key
    sys.exit("YOUTUBE_API_KEY not set in .env")


class QuotaTracker:
    def __init__(self, budget: int):
        self.budget = budget
        self.spent = 0

    def charge(self, units: int) -> bool:
        """Return True if the call fits in the budget (and charge it)."""
        if self.spent + units > self.budget:
            return False
        self.spent += units
        return True


def yt_get(endpoint: str, params: dict, api_key: str) -> dict | None:
    """One GET to the Data API with basic error handling."""
    params = dict(params, key=api_key)
    try:
        r = requests.get(f"{API_BASE}/{endpoint}", params=params, timeout=30)
    except requests.RequestException as e:
        print(f"  network error: {e} — skipping")
        return None
    if r.status_code == 403:
        reason = ""
        try:
            reason = r.json()["error"]["errors"][0].get("reason", "")
        except Exception:
            pass
        if reason in ("quotaExceeded", "dailyLimitExceeded"):
            print("  API quota exceeded — stopping for today (state is saved).")
            return {"_quota_exceeded": True}
        if reason == "commentsDisabled":
            return None
        print(f"  403 ({reason}) — skipping")
        return None
    if r.status_code != 200:
        print(f"  HTTP {r.status_code} — skipping")
        return None
    return r.json()


# ============================================================
# 2. Collection
# ============================================================

def search_videos(query: str, api_key: str, max_results: int = 25) -> list[str]:
    """Video ids for a search query (1 call = 100 units)."""
    data = yt_get("search", {
        "part": "id", "q": query, "type": "video",
        "maxResults": min(max_results, 50),
        "relevanceLanguage": "ta", "regionCode": "IN",
        "safeSearch": "none",
    }, api_key)
    if not data or data.get("_quota_exceeded"):
        return []
    return [item["id"]["videoId"] for item in data.get("items", [])
            if item.get("id", {}).get("videoId")]


def fetch_video_comments(video_id: str, api_key: str, quota: QuotaTracker,
                         max_pages: int = 5) -> list[dict]:
    """
    Up to max_pages * 100 top-level comments for one video.
    ANONYMIZATION happens here: only text + like count are extracted;
    author fields are never read.
    """
    comments, page_token = [], None
    for _ in range(max_pages):
        if not quota.charge(COST_COMMENTS):
            break
        params = {"part": "snippet", "videoId": video_id,
                  "maxResults": 100, "textFormat": "plainText",
                  "order": "relevance"}
        if page_token:
            params["pageToken"] = page_token
        data = yt_get("commentThreads", params, api_key)
        if not data:
            break
        if data.get("_quota_exceeded"):
            quota.spent = quota.budget
            break
        for item in data.get("items", []):
            sn = item["snippet"]["topLevelComment"]["snippet"]
            text = WS_RE.sub(" ", str(sn.get("textDisplay", ""))).strip()
            if text:
                comments.append({
                    "video_id": video_id,
                    "text": text,
                    "like_count": sn.get("likeCount", 0),
                })
        page_token = data.get("nextPageToken")
        if not page_token:
            break
        time.sleep(0.1)
    return comments


WS_RE = re.compile(r"\s+")


# ============================================================
# 3. State + output
# ============================================================

def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"seen_videos": [], "n_comments": 0}


def save_state(state: dict):
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def append_comments(rows: list[dict]):
    new_file = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "text", "like_count"])
        if new_file:
            w.writeheader()
        w.writerows(rows)


# ============================================================
# 4. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Collect Tamil YouTube comments")
    parser.add_argument("--max-videos", type=int, default=20,
                        help="videos per search query (default 20)")
    parser.add_argument("--pages-per-video", type=int, default=3,
                        help="comment pages per video, 100 each (default 3)")
    parser.add_argument("--budget", type=int, default=9000,
                        help="API unit budget for this run (default 9000)")
    args = parser.parse_args()

    api_key = load_api_key()
    OUT_DIR.mkdir(exist_ok=True)
    state = load_state()
    seen = set(state["seen_videos"])
    quota = QuotaTracker(args.budget)
    total_new = 0

    print(f"Resuming with {len(seen)} videos already collected, "
          f"{state['n_comments']} comments so far")

    for query in SEARCH_QUERIES:
        if not quota.charge(COST_SEARCH):
            print("Budget exhausted before search — stopping.")
            break
        print(f"\nSearch: {query!r}")
        video_ids = [v for v in search_videos(query, api_key, args.max_videos)
                     if v not in seen]
        print(f"  {len(video_ids)} new videos")

        for vid in video_ids:
            if quota.spent >= quota.budget:
                break
            rows = fetch_video_comments(vid, api_key, quota,
                                        max_pages=args.pages_per_video)
            seen.add(vid)
            if rows:
                append_comments(rows)
                total_new += len(rows)
                print(f"  {vid}: +{len(rows)} comments "
                      f"(total new: {total_new}, units: {quota.spent})")
            # persist incrementally so interruptions lose nothing
            state["seen_videos"] = sorted(seen)
            state["n_comments"] = state.get("n_comments", 0) + len(rows)
            save_state(state)

        if quota.spent >= quota.budget:
            print("\nBudget exhausted — rerun tomorrow to continue (resumes).")
            break

    print(f"\nDone. New comments this run: {total_new}")
    print(f"API units spent: {quota.spent}/{args.budget}")
    print(f"Corpus file: {CSV_PATH}")
    print("Reminder: collected/ is raw user text — keep it out of git.")


if __name__ == "__main__":
    main()
