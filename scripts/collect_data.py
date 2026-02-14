import os
import math
import csv
from googleapiclient.discovery import build
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta


# -------------------------------------------------
# Reaction Detection
# -------------------------------------------------
def looks_like_reaction(title):

    title_lower = title.lower()

    reaction_keywords = [
        "reaction","reacts","live reaction",
        "first time watching","watching for the first time",
        "first listen","first time hearing",
        "my honest thoughts","this made me cry",
        "i was not ready","this broke me",
        "speechless","i lost it",
        "this hit different","mind blown",
        "unbelievable","did not expect",
        "didn't expect","no way",
        "my thoughts on","thoughts on",
        "breaking down","let's talk about",
        "reviewing","live watch"
    ]

    emotional_patterns = [
        "??","!!","what","crazy",
        "insane","shocked","unexpected","wild"
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
    "official music video","lyrics","cover","remix",
    "live performance","concert","music video",
    "trailer","teaser","song"
]


# -------------------------------------------------
# Load API
# -------------------------------------------------
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

youtube = build("youtube", "v3", developerKey=API_KEY)

region = input("Enter region code (IN, US, GB, etc.): ").upper()

dataset = []

# Target sample sizes (balanced dataset).
TRENDING_TARGET = 260
NON_TRENDING_TARGET = 260


# -------------------------------------------------
# 1️⃣ Collect TRENDING Videos (label=1)
# -------------------------------------------------
print("\nCollecting Trending Videos...\n")

next_page_token = None

while len(dataset) < TRENDING_TARGET:

    request = youtube.videos().list(
        part="snippet,statistics,contentDetails",
        chart="mostPopular",
        regionCode=region,
        maxResults=50,
        pageToken=next_page_token
    )

    response = request.execute()

    if "items" not in response:
        break

    for item in response["items"]:

        title = item["snippet"]["title"]
        duration = item["contentDetails"]["duration"]

        if "M" not in duration and "H" not in duration:
            continue

        if any(word in title.lower() for word in blacklist_keywords):
            continue

        dataset.append({
            "title": title,
            "viral": 1
        })

    next_page_token = response.get("nextPageToken")
    if not next_page_token:
        break


# -------------------------------------------------
# 2️⃣ Collect NON-TRENDING Videos From Small Channels
# -------------------------------------------------
print("\nCollecting Non-Trending Videos...\n")

existing_titles = set([x["title"] for x in dataset])
non_trending_count = 0
recent_after = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat().replace("+00:00", "Z")
seed_queries = ["vlog", "tutorial", "review", "challenge", "gameplay", "podcast"]

for query in seed_queries:
    next_page_token = None

    while non_trending_count < NON_TRENDING_TARGET:

        request = youtube.search().list(
            part="snippet",
            q=query,
            regionCode=region,
            order="date",
            type="video",
            publishedAfter=recent_after,
            maxResults=50,
            pageToken=next_page_token
        )

        response = request.execute()

        if "items" not in response:
            break

        for item in response["items"]:

            video_id = item["id"]["videoId"]

            # Fetch video stats
            video_request = youtube.videos().list(
                part="statistics",
                id=video_id
            )

            video_response = video_request.execute()

            if "items" not in video_response or not video_response["items"]:
                continue

            stats = video_response["items"][0].get("statistics", {})

            views = int(stats.get("viewCount", 0))

            # Keep only low-view videos as non-trending samples.
            if views > 10000:
                continue

            title = item["snippet"]["title"]

            if title in existing_titles:
                continue

            if any(word in title.lower() for word in blacklist_keywords):
                continue

            dataset.append({
                "title": title,
                "viral": 0
            })

            existing_titles.add(title)
            non_trending_count += 1

            if non_trending_count >= NON_TRENDING_TARGET:
                break

        next_page_token = response.get("nextPageToken")

        if not next_page_token:
            break

    if non_trending_count >= NON_TRENDING_TARGET:
        break



print("Trending samples:", len([x for x in dataset if x["viral"] == 1]))
print("Non-Trending samples:", len([x for x in dataset if x["viral"] == 0]))


# -------------------------------------------------
# Save Dataset
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(BASE_DIR, "data")
os.makedirs(data_dir, exist_ok=True)

csv_path = os.path.join(data_dir, "trendpulse_dataset.csv")

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["title", "viral"])

    for row in dataset:
        writer.writerow([row["title"], row["viral"]])

print("\nBalanced dataset saved at:", csv_path)
