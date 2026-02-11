import os
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

blocked_categories = [
    "Music",
    "Entertainment",
    "News & Politics",
    "Sports",
    "Film & Animation"
]

blacklist_keywords = [
    "live", "match", "vs", "election",
    "breaking", "budget", "full movie",
    "episode", "season", "trailer", "teaser"
]

for item in response["items"]:

    title = item["snippet"]["title"]
    title_lower = title.lower()

    category_id = item["snippet"].get("categoryId", "Unknown")
    category_name = category_map.get(category_id, "Unknown")

    duration = item["contentDetails"]["duration"]

    channel_id = item["snippet"]["channelId"]

    # Fetch channel stats
    channel_request = youtube.channels().list(
        part="statistics",
        id=channel_id
    )

    channel_response = channel_request.execute()

    subscriber_count = 0
    if channel_response["items"]:
        subscriber_count = int(
            channel_response["items"][0]["statistics"].get("subscriberCount", 0)
        )

    views = int(item["statistics"].get("viewCount", 0))

    # Calculate hours since publish
    published = item["snippet"]["publishedAt"]
    published_time = datetime.fromisoformat(published.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    hours_since_publish = (now - published_time).total_seconds() / 3600

    # ---------------- FILTERS ----------------

    # Remove blocked categories
    if category_name in blocked_categories:
        continue

    # Remove Shorts
    if "M" not in duration and "H" not in duration:
        continue

    # Remove blacklisted keywords
    if any(word in title_lower for word in blacklist_keywords):
        continue

    # Remove very large channels
    if subscriber_count > 1_000_000:
        continue

    # Remove older videos
    if hours_since_publish > 72:
        continue

    # ------------------------------------------

    # If it reaches here, it passed everything
    print("=" * 80)
    print("Title:", title)
    print("Category:", category_name)
    print("Subscribers:", subscriber_count)
    print("Views:", views)
    print("Hours Since Publish:", round(hours_since_publish, 2))
