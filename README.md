# kk11 Automation Project

This project provides tools to automatically download YouTube Shorts, train a model
on existing media, gather news, generate new videos and upload them back to
YouTube. All features are combined into a single script `app.py` with a simple
Tkinter GUI.

## Features

* **YouTube Downloader** – Downloads Korean Shorts with at least 100k views using
  `pytube`.
* **Model Training** – Simple framework based on PyTorch or TensorFlow with GPU
  support when available. Training can resume from saved checkpoints.
* **News Fetcher** – Retrieves recent news while filtering out politics, war and
  violence related stories via the NewsAPI service.
* **Video Generator** – Creates a short video from text using `moviepy`.
* **YouTube Uploader** – Uploads generated videos through the YouTube Data API.
* **GUI** – Tkinter based interface to drive the whole workflow.

## Installation

1. Clone this repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```
python app.py
```

The application will prompt for a YouTube API key. From there you can
- download shorts
- train the model
- fetch news
- generate a video
- upload the result back to YouTube.

## Dependencies

- pytube
- moviepy
- google-api-python-client
- requests
- torch *(optional)*
- tensorflow *(optional)*

Install these manually or via `requirements.txt`.
