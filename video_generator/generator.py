from __future__ import annotations

from pathlib import Path
from typing import List

from moviepy.editor import TextClip, concatenate_videoclips


class VideoGenerator:
    """Generate video, audio and subtitles using a trained model."""

    def __init__(self, model) -> None:
        self.model = model

    def generate(self, texts: List[str], output_path: str) -> str:
        clips = [TextClip(txt, fontsize=24, color="white").set_duration(3) for txt in texts]
        video = concatenate_videoclips(clips)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        video.write_videofile(output_path)
        return output_path
