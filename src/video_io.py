from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Frame:
    index: int
    timestamp_sec: float
    image: np.ndarray


@dataclass
class Clip:
    frames: list[Frame] = field(default_factory=list)

    @property
    def start_sec(self) -> float:
        return self.frames[0].timestamp_sec if self.frames else 0.0

    @property
    def end_sec(self) -> float:
        return self.frames[-1].timestamp_sec if self.frames else 0.0


def sample_frames(video_path: str | Path, sample_fps: float = 1.0) -> list[Frame]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(native_fps / sample_fps))
    frames = []
    idx = 0

    while True:
        ret, img = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            timestamp = idx / native_fps
            frames.append(Frame(index=idx, timestamp_sec=timestamp, image=img))
        idx += 1

    cap.release()
    return frames


def group_into_clips(
    frames: list[Frame], clip_duration_sec: float, clip_stride_sec: float
) -> list[Clip]:
    if not frames:
        return []

    clips = []
    start = frames[0].timestamp_sec
    end_time = frames[-1].timestamp_sec

    while start <= end_time:
        window_end = start + clip_duration_sec
        clip_frames = [f for f in frames if start <= f.timestamp_sec < window_end]
        if clip_frames:
            clips.append(Clip(frames=clip_frames))
        start += clip_stride_sec

    return clips


def write_highlight_reel(
    source_video: str | Path,
    segments: list[tuple[float, float]],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    cap = cv2.VideoCapture(str(source_video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for start_sec, end_sec in sorted(segments):
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)

    writer.release()
    cap.release()
    return output_path
