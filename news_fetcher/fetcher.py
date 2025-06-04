from __future__ import annotations

import requests
from typing import List, Dict

EXCLUDED_KEYWORDS = {"politics", "war", "violence"}


def get_recent_news(api_key: str, query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Fetch recent news articles excluding certain keywords."""
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "pageSize": max_results * 2,
        "apiKey": api_key,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    articles = []
    for item in resp.json().get("articles", []):
        title = item.get("title", "")
        if any(word.lower() in title.lower() for word in EXCLUDED_KEYWORDS):
            continue
        articles.append({"title": title, "url": item.get("url")})
        if len(articles) >= max_results:
            break
    return articles
