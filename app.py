import os
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from typing import List, Optional, Dict

import requests
from pytube import Search
from moviepy.editor import TextClip, concatenate_videoclips
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

try:
    import torch
    from torch import nn, optim
except Exception:  # noqa: BLE001
    torch = None  # type: ignore
    nn = None  # type: ignore
    optim = None  # type: ignore

try:
    import tensorflow as tf
except Exception:  # noqa: BLE001
    tf = None


# 유튜브 쇼츠 다운로드 함수
def download_korean_shorts(output_dir: str, max_results: int = 5) -> List[str]:
    """한국어 쇼츠 중 조회수 100k 이상 영상을 다운로드한다."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    search = Search("shorts 한국")
    urls: List[str] = []
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


# 딥러닝 학습기 클래스
class Trainer:
    """PyTorch 또는 TensorFlow로 간단한 딥러닝 모델을 학습한다."""

    def __init__(self, checkpoint: Optional[str] = None, use_gpu: bool = True) -> None:
        self.checkpoint = checkpoint
        self.use_gpu = use_gpu and torch and torch.cuda.is_available()
        self.model = None
        if checkpoint:
            self.load(checkpoint)
        else:
            self._init_model()

    def _generate_dummy_data(self, samples: int = 100):
        """랜덤 데이터를 생성하여 학습에 사용한다."""
        if torch:
            x = torch.randn(samples, 10)
            y = x.sum(dim=1, keepdim=True)
            return list(zip(x, y))
        elif tf:
            x = tf.random.normal((samples, 10))
            y = tf.reduce_sum(x, axis=1, keepdims=True)
            return list(zip(x, y))
        return []

    def _init_model(self) -> None:
        if torch:
            self.model = nn.Sequential(
                nn.Linear(10, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            if self.use_gpu:
                self.model.cuda()
            self.optimizer = optim.Adam(self.model.parameters())
            self.loss_fn = nn.MSELoss()
        elif tf:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation="relu", input_shape=(10,)),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(1),
            ])
            self.model.compile(optimizer="adam", loss="mse")
        else:
            raise ImportError("PyTorch나 TensorFlow가 설치되어 있지 않습니다")

    def load(self, path: str) -> None:
        if torch and Path(path).with_suffix(".pt").exists():
            self.model = torch.load(path)
        elif tf and Path(path).with_suffix(".h5").exists():
            self.model = tf.keras.models.load_model(path)
        else:
            self._init_model()

    def train(self, data=None, epochs: int = 5) -> None:
        """주어진 데이터로 모델을 학습한다."""
        if data is None:
            data = self._generate_dummy_data()

        if torch and isinstance(self.model, nn.Module):
            self.model.train()
            for _ in range(epochs):
                for x, y in data:
                    if self.use_gpu:
                        x = x.cuda()
                        y = y.cuda()
                    self.optimizer.zero_grad()
                    pred = self.model(x)
                    loss = self.loss_fn(pred, y)
                    loss.backward()
                    self.optimizer.step()
        elif tf:
            x, y = zip(*data)
            self.model.fit(tf.stack(x), tf.stack(y), epochs=epochs)
        else:
            raise RuntimeError("학습할 수 있는 백엔드가 없습니다")

    def save(self, path: str) -> None:
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        if torch and isinstance(self.model, nn.Module):
            torch.save(self.model, path if path.endswith(".pt") else f"{path}.pt")
        elif tf:
            self.model.save(path if path.endswith(".h5") else f"{path}.h5")
        else:
            raise RuntimeError("학습할 수 있는 백엔드가 없습니다")


EXCLUDED_KEYWORDS = {"politics", "war", "violence"}


# 뉴스 기사 가져오기
def get_recent_news(api_key: str, query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """정치/전쟁/폭력 관련 키워드를 제외한 최근 뉴스를 가져온다."""
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "pageSize": max_results * 2,
        "apiKey": api_key,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    articles: List[Dict[str, str]] = []
    for item in resp.json().get("articles", []):
        title = item.get("title", "")
        if any(word.lower() in title.lower() for word in EXCLUDED_KEYWORDS):
            continue
        articles.append({"title": title, "url": item.get("url")})
        if len(articles) >= max_results:
            break
    return articles


# 간단한 텍스트 영상 생성기
class VideoGenerator:
    """학습된 모델로부터 텍스트 영상을 만든다."""

    def __init__(self, model) -> None:
        self.model = model

    def generate(self, texts: List[str], output_path: str) -> str:
        clips = [TextClip(txt, fontsize=24, color="white").set_duration(3) for txt in texts]
        video = concatenate_videoclips(clips)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        video.write_videofile(output_path)
        return output_path


# 유튜브 업로드 함수
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


# Tkinter 기반 GUI 애플리케이션
class Application(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("유튜브 자동화")
        self.geometry("400x400")
        self.api_key_var = tk.StringVar()

        tk.Label(self, text="유튜브 API 키:").pack()
        tk.Entry(self, textvariable=self.api_key_var, width=50).pack()
        tk.Button(self, text="쇼츠 다운로드", command=self.download_shorts).pack(fill=tk.X)
        tk.Button(self, text="모델 학습", command=self.train_model).pack(fill=tk.X)
        tk.Button(self, text="뉴스 가져오기", command=self.fetch_news).pack(fill=tk.X)
        tk.Button(self, text="영상 생성", command=self.generate_video).pack(fill=tk.X)
        tk.Button(self, text="영상 업로드", command=self.upload_video_ui).pack(fill=tk.X)

        self.trainer: Optional[Trainer] = None
        self.news_texts: Optional[List[str]] = None
        self.generated_video: Optional[str] = None

    def download_shorts(self) -> None:
        directory = filedialog.askdirectory()
        if directory:
            urls = download_korean_shorts(directory)
            messagebox.showinfo("다운로드 완료", f"{len(urls)}개 영상 다운로드 완료")

    def train_model(self) -> None:
        self.trainer = Trainer()
        self.trainer.train()
        file = filedialog.asksaveasfilename(defaultextension=".pt")
        if file:
            self.trainer.save(file)
            messagebox.showinfo("모델", "모델 학습 및 저장 완료")
        else:
            messagebox.showinfo("모델", "모델 학습 완료")

    def fetch_news(self) -> None:
        api_key = self.api_key_var.get()
        if not api_key:
            messagebox.showerror("오류", "API 키를 입력하세요")
            return
        self.news_texts = [n["title"] for n in get_recent_news(api_key, "technology")]
        messagebox.showinfo("뉴스", f"{len(self.news_texts)}개 기사 가져옴")

    def generate_video(self) -> None:
        if not self.trainer or not self.news_texts:
            messagebox.showerror("오류", "모델 학습과 뉴스 가져오기를 먼저 수행하세요")
            return
        gen = VideoGenerator(self.trainer.model)
        file = filedialog.asksaveasfilename(defaultextension=".mp4")
        if file:
            self.generated_video = gen.generate(self.news_texts, file)
            messagebox.showinfo("영상", "영상 생성 완료")

    def upload_video_ui(self) -> None:
        api_key = self.api_key_var.get()
        if not api_key or not self.generated_video:
            messagebox.showerror("오류", "API 키 또는 생성된 영상이 없습니다")
            return
        url = upload_video(api_key, self.generated_video, "생성된 영상", "자동 업로드")
        messagebox.showinfo("업로드 완료", f"영상 URL: {url}")


def main() -> None:
    app = Application()
    app.mainloop()


if __name__ == "__main__":
    main()
