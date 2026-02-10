import os
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
API_KEY=os.getenv("YOUTUBE_API_KEY")

# Create YouTube API client
youtube=build("youtube","v3",developerKey=API_KEY)

# Search for videos on a topic
request=youtube.search().list(
    part="snippet",
    q="technology",
    type="video",
    maxResults=5
)

response=request.execute()

# Print results
for item in response["items"]:
    title=item["snippet"]["title"]
    video_id=item["id"]["videoId"]
    print(title,"->",video_id)
