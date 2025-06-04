import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional

from youtube_downloader import download_korean_shorts
from model_training import Trainer
from news_fetcher import get_recent_news
from video_generator import VideoGenerator
from youtube_uploader import upload_video


class Application(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("YouTube Automation")
        self.geometry("400x400")
        self.api_key_var = tk.StringVar()

        tk.Label(self, text="YouTube API Key:").pack()
        tk.Entry(self, textvariable=self.api_key_var, width=50).pack()
        tk.Button(self, text="Download Shorts", command=self.download_shorts).pack(fill=tk.X)
        tk.Button(self, text="Train Model", command=self.train_model).pack(fill=tk.X)
        tk.Button(self, text="Fetch News", command=self.fetch_news).pack(fill=tk.X)
        tk.Button(self, text="Generate Video", command=self.generate_video).pack(fill=tk.X)
        tk.Button(self, text="Upload Video", command=self.upload_video).pack(fill=tk.X)

        self.trainer: Optional[Trainer] = None
        self.news_texts: Optional[list[str]] = None
        self.generated_video: Optional[str] = None

    def download_shorts(self) -> None:
        directory = filedialog.askdirectory()
        if directory:
            urls = download_korean_shorts(directory)
            messagebox.showinfo("Downloaded", f"Downloaded {len(urls)} videos")

    def train_model(self) -> None:
        self.trainer = Trainer()
        # placeholder for real data loading
        data = []
        self.trainer.train(data)
        file = filedialog.asksaveasfilename(defaultextension=".pt")
        if file:
            self.trainer.save(file)
            messagebox.showinfo("Model", "Model saved")

    def fetch_news(self) -> None:
        api_key = self.api_key_var.get()
        if not api_key:
            messagebox.showerror("Error", "Enter API key")
            return
        self.news_texts = [n["title"] for n in get_recent_news(api_key, "technology")]
        messagebox.showinfo("News", f"Fetched {len(self.news_texts)} items")

    def generate_video(self) -> None:
        if not self.trainer or not self.news_texts:
            messagebox.showerror("Error", "Train model and fetch news first")
            return
        gen = VideoGenerator(self.trainer.model)
        file = filedialog.asksaveasfilename(defaultextension=".mp4")
        if file:
            self.generated_video = gen.generate(self.news_texts, file)
            messagebox.showinfo("Video", "Video generated")

    def upload_video(self) -> None:
        api_key = self.api_key_var.get()
        if not api_key or not self.generated_video:
            messagebox.showerror("Error", "API key or generated video missing")
            return
        url = upload_video(api_key, self.generated_video, "Generated Video", "Auto uploaded")
        messagebox.showinfo("Uploaded", f"Video URL: {url}")


if __name__ == "__main__":
    app = Application()
    app.mainloop()
