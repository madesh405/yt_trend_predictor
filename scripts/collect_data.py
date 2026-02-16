import os
import csv
from datetime import datetime
from googleapiclient.discovery import build
from dotenv import load_dotenv


# =========================================================
# CONFIG
# =========================================================
CHANNEL_SAMPLE_SIZE = 150
VIDEOS_PER_CHANNEL = 25
MIN_RATIO = 2.0
MAX_RATIO = 0.5


# =========================================================
# Load API
# =========================================================
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

if not API_KEY:
    raise ValueError("Missing YOUTUBE_API_KEY")

youtube = build("youtube", "v3", developerKey=API_KEY)
region = input("Enter region code (IN, US, GB, etc.): ").upper()

dataset = []


# =========================================================
# Safe Execute
# =========================================================
def safe_execute(request):
    try:
        return request.execute()
    except Exception as e:
        print("API Error:", e)
        return None


def parse_duration(duration):
    import re
    hours = minutes = seconds = 0
    h = re.search(r'(\d+)H', duration)
    m = re.search(r'(\d+)M', duration)
    s = re.search(r'(\d+)S', duration)

    if h: hours = int(h.group(1))
    if m: minutes = int(m.group(1))
    if s: seconds = int(s.group(1))

    return hours*3600 + minutes*60 + seconds


# =========================================================
# STEP 1 — Collect Channels (Limited search usage)
# =========================================================
print("\nCollecting channels...\n")

channels = set()
next_page_token = None

while len(channels) < CHANNEL_SAMPLE_SIZE:

    request = youtube.search().list(
        part="snippet",
        q="a",
        type="channel",
        regionCode=region,
        maxResults=50,
        pageToken=next_page_token
    )

    response = safe_execute(request)
    if not response:
        break

    for item in response["items"]:
        channels.add(item["snippet"]["channelId"])
        if len(channels) >= CHANNEL_SAMPLE_SIZE:
            break

    next_page_token = response.get("nextPageToken")
    if not next_page_token:
        break

print("Collected channels:", len(channels))


# =========================================================
# STEP 2 — Fetch Videos Using Upload Playlist (CHEAP)
# =========================================================
print("\nCollecting videos per channel...\n")

for channel_id in channels:

    # ---- Get uploads playlist ID ----
    channel_request = youtube.channels().list(
        part="contentDetails",
        id=channel_id
    )

    channel_response = safe_execute(channel_request)
    if not channel_response or not channel_response["items"]:
        continue

    uploads_playlist = channel_response["items"][0] \
        ["contentDetails"]["relatedPlaylists"]["uploads"]

    # ---- Fetch videos from uploads playlist ----
    playlist_request = youtube.playlistItems().list(
        part="contentDetails",
        playlistId=uploads_playlist,
        maxResults=VIDEOS_PER_CHANNEL
    )

    playlist_response = safe_execute(playlist_request)
    if not playlist_response:
        continue

    video_ids = [
        item["contentDetails"]["videoId"]
        for item in playlist_response["items"]
    ]

    if not video_ids:
        continue

    # ---- Fetch video stats in batch ----
    videos_request = youtube.videos().list(
        part="snippet,statistics,contentDetails",
        id=",".join(video_ids)
    )

    videos_response = safe_execute(videos_request)
    if not videos_response:
        continue

    videos = videos_response["items"]

    # ---- Compute channel average ----
    view_counts = [
        int(v["statistics"].get("viewCount", 0))
        for v in videos
    ]

    if not view_counts:
        continue

    channel_avg_views = sum(view_counts) / len(view_counts)

    # ---- Label videos ----
    for video in videos:

        snippet = video.get("snippet", {})
        stats = video.get("statistics", {})
        content_details = video.get("contentDetails", {})

        duration = content_details.get("duration")
        if not duration:
            continue

        title = snippet.get("title", "")
        if not title:
            continue

        views = int(stats.get("viewCount", 0))
        likes = int(stats.get("likeCount", 0))
        comments = int(stats.get("commentCount", 0))

        if channel_avg_views == 0:
            continue

        ratio = views / channel_avg_views

        if ratio >= MIN_RATIO:
            label = 1
        elif ratio <= MAX_RATIO:
            label = 0
        else:
            continue

        duration_seconds = parse_duration(duration)
        if duration_seconds < 60:
            continue

        published_at = snippet.get("publishedAt")
        if not published_at:
            continue

        publish_time = datetime.fromisoformat(
            published_at.replace("Z", "+00:00")
        )

        dataset.append({
            "title": title,
            "title_length": len(title),
            "caps_ratio": sum(1 for c in title if c.isupper()) / len(title) if len(title) > 0 else 0,
            "views": views,
            "channel_avg_views": channel_avg_views,
            "performance_ratio": ratio,
            "likes": likes,
            "comments": comments,
            "duration_sec": duration_seconds,
            "publish_hour": publish_time.hour,
            "viral": label
        })


# =========================================================
# SAVE DATASET
# =========================================================
print("\nTotal labeled samples:", len(dataset))

if dataset:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "trendpulse_channel_relative.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=dataset[0].keys())
        writer.writeheader()
        writer.writerows(dataset)

    print("\nDataset saved at:", csv_path)
else:
    print("No labeled samples collected.")
