import os
import math
from googleapiclient.discovery import build
from dotenv import load_dotenv
from datetime import datetime, timezone

# -------------------------------------------------
# Load API Key
# -------------------------------------------------
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

if not API_KEY:
    raise ValueError("API key not found.")

youtube = build("youtube", "v3", developerKey=API_KEY)

region = input("Enter region code (IN, US, GB, etc.): ").upper()

print("\nFiltered Trending Candidates\n")

# -------------------------------------------------
# Fetch Category Mapping
# -------------------------------------------------
category_request = youtube.videoCategories().list(
    part="snippet",
    regionCode=region
)

category_response = category_request.execute()

category_map = {}
if "items" in category_response:
    for item in category_response["items"]:
        category_map[item["id"]] = item["snippet"]["title"]

# -------------------------------------------------
# Fetch Trending Videos
# -------------------------------------------------
request = youtube.videos().list(
    part="snippet,statistics,contentDetails",
    chart="mostPopular",
    regionCode=region,
    maxResults=50
)

response = request.execute()

if "items" not in response:
    print("Error fetching trending videos.")
    exit()

# -------------------------------------------------
# Batch Fetch Channel Statistics
# -------------------------------------------------
channel_ids = list(set(
    item["snippet"]["channelId"]
    for item in response["items"]
))

channel_request = youtube.channels().list(
    part="statistics",
    id=",".join(channel_ids)
)

channel_response = channel_request.execute()

channel_map = {}

if "items" in channel_response:
    for ch in channel_response["items"]:
        stats = ch.get("statistics", {})

        # Skip channels hiding subscriber count
        if "subscriberCount" not in stats:
            continue

        channel_map[ch["id"]] = int(stats["subscriberCount"])

# -------------------------------------------------
# Reaction Detection
# -------------------------------------------------
def looks_like_reaction(title):

    title_lower = title.lower()

    reaction_keywords = [
        "reaction", "reacts", "live reaction",
        "first time watching",
        "watching for the first time",
        "first listen",
        "first time hearing",
        "my honest thoughts",
        "this made me cry",
        "i was not ready",
        "this broke me",
        "speechless",
        "i lost it",
        "this hit different",
        "mind blown",
        "unbelievable",
        "did not expect",
        "didn't expect",
        "no way",
        "my thoughts on",
        "thoughts on",
        "breaking down",
        "let's talk about",
        "reviewing",
        "live watch"
    ]

    emotional_patterns = [
        "??", "!!",
        "what", "crazy",
        "insane", "shocked",
        "unexpected", "wild"
    ]

    score = 0

    if any(word in title_lower for word in reaction_keywords):
        score += 2

    if any(pattern in title_lower for pattern in emotional_patterns):
        score += 1

    words = title.split()
    if any(word.isupper() and len(word) > 3 for word in words):
        score += 1

    if title.count("!") >= 2 or title.count("?") >= 2:
        score += 1

    return score >= 2

# -------------------------------------------------
# Filters
# -------------------------------------------------
blocked_categories = ["Music"]

blacklist_keywords = [
    "official music video",
    "lyrics",
    "cover",
    "remix",
    "live performance",
    "concert",
    "music video",
    "trailer",
    "teaser",
    "song"
]

# -------------------------------------------------
# Process Videos
# -------------------------------------------------
videos_data = []

for item in response["items"]:

    title = item["snippet"]["title"]
    title_lower = title.lower()

    category_id = item["snippet"].get("categoryId")
    category_name = category_map.get(category_id, "Unknown")

    duration = item["contentDetails"]["duration"]
    channel_id = item["snippet"]["channelId"]

    # Skip if channel stats missing (hidden subs)
    subscriber_count = channel_map.get(channel_id)
    if subscriber_count is None:
        continue

    # Validate video stats
    stats = item.get("statistics", {})
    if "viewCount" not in stats:
        continue

    views = int(stats["viewCount"])
    if views == 0:
        continue

    likes = int(stats.get("likeCount", 0))
    comments = int(stats.get("commentCount", 0))

    # Time calculation
    published = item["snippet"]["publishedAt"]
    published_time = datetime.fromisoformat(
        published.replace("Z", "+00:00")
    )
    now = datetime.now(timezone.utc)
    hours_since_publish = (
        now - published_time
    ).total_seconds() / 3600

    # ---------------- FILTERS ----------------

    is_reaction = looks_like_reaction(title)

    if category_name in blocked_categories and not is_reaction:
        continue

    if "M" not in duration and "H" not in duration:
        continue

    if any(word in title_lower for word in blacklist_keywords) and not is_reaction:
        continue

    if subscriber_count > 5_000_000:
        continue

    if hours_since_publish > 96:
        continue

    # ---------------- TrendPulse Calculation ----------------

    relative_velocity = (
        math.log1p(views) /
        (math.log1p(subscriber_count + 1) * max(hours_since_publish, 1))
    )

    engagement_rate = (likes + comments) / views

    size_boost = 1 - min(subscriber_count / 5_000_000, 1)

    trend_pulse = (
        (relative_velocity * 40) +
        (engagement_rate * 30) +
        (size_boost * 20)
    )

    videos_data.append({
        "title": title,
        "category": category_name,
        "subs": subscriber_count,
        "views": views,
        "likes": likes,
        "comments": comments,
        "hours": hours_since_publish,
        "score": trend_pulse
    })

# -------------------------------------------------
# Sort by TrendPulse Score
# -------------------------------------------------
videos_data.sort(key=lambda x: x["score"], reverse=True)

# -------------------------------------------------
# Print Results
# -------------------------------------------------
for video in videos_data:

    print("=" * 80)
    print("Title:", video["title"])
    print("Category:", video["category"])
    print("Subscribers:", video["subs"])
    print("Views:", video["views"])
    print("Likes:", video["likes"])
    print("Comments:", video["comments"])
    print("Hours Since Publish:", round(video["hours"], 2))
    print("TrendPulse Score:", round(video["score"], 2))
