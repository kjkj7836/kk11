from __future__ import annotations

import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


def upload_video(api_key: str, video_path: str, title: str, description: str) -> str:
    youtube = build("youtube", "v3", developerKey=api_key)

    request_body = {
        "snippet": {"title": title, "description": description},
        "status": {"privacyStatus": "public"},
    }

    media = MediaFileUpload(video_path)
    request = youtube.videos().insert(part="snippet,status", body=request_body, media_body=media)
    response = request.execute()
    return f"https://www.youtube.com/watch?v={response['id']}"
