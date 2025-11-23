# fetch_comments.py
import os
import time
import json
import pandas as pd
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
if not API_KEY:
    raise ValueError("Put your YOUTUBE_API_KEY in a .env file")

youtube = build("youtube", "v3", developerKey=API_KEY)

def fetch_comments_from_video(video_id, max_comments=2000):
    comments = []
    next_page_token = None
    while len(comments) < max_comments:
        req = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token,
            textFormat="plainText"
        )
        res = req.execute()
        for item in res.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "comment_id": item["id"],
                "text": snippet.get("textDisplay", ""),
                "likeCount": snippet.get("likeCount", 0),
                "publishedAt": snippet.get("publishedAt"),
                "authorDisplayName": snippet.get("authorDisplayName"),
            })
            if len(comments) >= max_comments:
                break
        next_page_token = res.get("nextPageToken")
        if not next_page_token:
            break
        time.sleep(0.1)
    return pd.DataFrame(comments)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_id", required=True)
    parser.add_argument("--out", default="comments.csv")
    parser.add_argument("--max", type=int, default=1000)
    args = parser.parse_args()
    df = fetch_comments_from_video(args.video_id, max_comments=args.max)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} comments to {args.out}")
