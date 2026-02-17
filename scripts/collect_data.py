import os
import csv
import re
from datetime import datetime, timezone
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv


# =========================================================
# CONFIG
# =========================================================
MIN_RATIO = 2.0
MAX_RATIO = 0.5
CHANNEL_SAMPLE_SIZE = 400
VIDEOS_PER_CHANNEL = 40


# =========================================================
# Load API
# =========================================================
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

if not API_KEY:
    raise ValueError("Missing YOUTUBE_API_KEY")

youtube = build("youtube", "v3", developerKey=API_KEY)
region = input("Enter region code (IN, US, GB, etc.): ").upper()

now = datetime.now(timezone.utc)


# =========================================================
# Safe Execute
# =========================================================
def safe_execute(request):
    try:
        return request.execute()
    except HttpError:
        return None
    except Exception:
        return None


# =========================================================
# Duration Parser
# =========================================================
def parse_duration(duration):
    hours = minutes = seconds = 0
    h = re.search(r'(\d+)H', duration)
    m = re.search(r'(\d+)M', duration)
    s = re.search(r'(\d+)S', duration)

    if h: hours = int(h.group(1))
    if m: minutes = int(m.group(1))
    if s: seconds = int(s.group(1))

    return hours*3600 + minutes*60 + seconds


# =========================================================
# STEP 1 — Collect Channels from Trending Videos
# =========================================================
print("\nCollecting active channels from trending videos...\n")

channels = set()
page_token = None

while len(channels) < CHANNEL_SAMPLE_SIZE:

    request = youtube.videos().list(
        part="snippet",
        chart="mostPopular",
        regionCode=region,
        maxResults=50,
        pageToken=page_token
    )

    response = safe_execute(request)
    if not response:
        break

    for item in response["items"]:
        channels.add(item["snippet"]["channelId"])

    page_token = response.get("nextPageToken")
    if not page_token:
        break

print("Collected channels:", len(channels))


# =========================================================
# STEP 2 — Collect Videos Per Channel
# =========================================================
print("\nCollecting videos per channel...\n")

viral_samples = []
nonviral_samples = []

for channel_id in channels:

    # ---------------------------------------------
    # Channel Statistics
    # ---------------------------------------------
    channel_request = youtube.channels().list(
        part="statistics",
        id=channel_id
    )

    channel_response = safe_execute(channel_request)
    if not channel_response or not channel_response["items"]:
        continue

    channel_stats = channel_response["items"][0]["statistics"]

    subscriber_count = int(channel_stats.get("subscriberCount", 0))
    total_channel_views = int(channel_stats.get("viewCount", 0))
    channel_video_count = int(channel_stats.get("videoCount", 0))

    views_per_video = total_channel_views / (channel_video_count + 1)

    # ---------------------------------------------
    # Fetch Videos (Recent Uploads)
    # ---------------------------------------------
    search_request = youtube.search().list(
        part="id",
        channelId=channel_id,
        type="video",
        order="date",
        maxResults=VIDEOS_PER_CHANNEL
    )

    search_response = safe_execute(search_request)
    if not search_response:
        continue

    video_ids = [
        item["id"]["videoId"]
        for item in search_response["items"]
    ]

    if not video_ids:
        continue

    videos_request = youtube.videos().list(
        part="snippet,statistics,contentDetails",
        id=",".join(video_ids)
    )

    videos_response = safe_execute(videos_request)
    if not videos_response:
        continue

    videos = videos_response["items"]

    view_counts = [
        int(v.get("statistics", {}).get("viewCount", 0))
        for v in videos
    ]

    if not view_counts:
        continue

    channel_avg_views = sum(view_counts) / len(view_counts)

    for video in videos:

        snippet = video.get("snippet", {})
        stats = video.get("statistics", {})
        content_details = video.get("contentDetails", {})

        duration = content_details.get("duration")
        if not duration:
            continue

        duration_seconds = parse_duration(duration)

        # Remove Shorts
        if duration_seconds <= 90:
            continue

        title = snippet.get("title", "")
        if not title:
            continue

        # Remove obvious music spam
        if "official music video" in title.lower():
            continue

        description = snippet.get("description", "")
        tags = " ".join(snippet.get("tags", []))

        full_text = title + " " + description + " " + tags

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

        published_at = snippet.get("publishedAt")
        if not published_at:
            continue

        publish_time = datetime.fromisoformat(
            published_at.replace("Z", "+00:00")
        )

        age_hours = (now - publish_time).total_seconds() / 3600
        velocity = views / (age_hours + 1)

        like_ratio = likes / (views + 1)
        comment_ratio = comments / (views + 1)

        sample = {
            "full_text": full_text,
            "title_length": len(title),
            "caps_ratio": sum(1 for c in title if c.isupper()) / len(title),
            "views": views,
            "channel_avg_views": channel_avg_views,
            "performance_ratio": ratio,
            "likes": likes,
            "comments": comments,
            "like_ratio": like_ratio,
            "comment_ratio": comment_ratio,
            "velocity": velocity,
            "subscriber_count": subscriber_count,
            "views_per_video": views_per_video,
            "duration_sec": duration_seconds,
            "publish_hour": publish_time.hour,
            "viral": label
        }

        if label == 1:
            viral_samples.append(sample)
        else:
            nonviral_samples.append(sample)


# =========================================================
# STEP 3 — Balance Dataset
# =========================================================
min_size = min(len(viral_samples), len(nonviral_samples))
dataset = viral_samples[:min_size] + nonviral_samples[:min_size]

print("\nViral samples:", len(viral_samples))
print("Non-viral samples:", len(nonviral_samples))
print("Balanced dataset size:", len(dataset))


# =========================================================
# SAVE DATASET
# =========================================================
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
