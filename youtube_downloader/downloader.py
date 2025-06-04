from pathlib import Path
from typing import List

from pytube import Search


def download_korean_shorts(output_dir: str, max_results: int = 5) -> List[str]:
    """Download Korean YouTube Shorts with at least 100k views.

    Args:
        output_dir: Directory to save videos.
        max_results: Maximum number of videos to download.

    Returns:
        List of video URLs downloaded.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    search = Search("shorts 한국")
    urls = []
    for video in search.results:
        try:
            if video.length <= 60 and video.views and video.views >= 100000:
                stream = (
                    video.streams.filter(file_extension="mp4", progressive=True)
                    .order_by("resolution")
                    .desc()
                    .first()
                )
                if stream:
                    stream.download(output_path=output_dir)
                    urls.append(video.watch_url)
                    if len(urls) >= max_results:
                        break
        except Exception:
            continue
    return urls
