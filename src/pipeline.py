from pathlib import Path

from .video_io import sample_frames, group_into_clips
from .phase1_detection import ObjectDetector
from .phase2_frame_caption import FrameCaptioner
from .phase3_clip_caption import ClipCaptioner
from .phase4_synthesis import Synthesizer


class HighlightPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.detector = ObjectDetector(config.get("phase1_detection", {}))
        self.frame_captioner = FrameCaptioner(config.get("phase2_frame_caption", {}))
        self.clip_captioner = ClipCaptioner(config.get("phase3_clip_caption", {}))
        self.synthesizer = Synthesizer(config.get("phase4_synthesis", {}))

    def run(self, video_path: str | Path, output_dir: str | Path) -> tuple[Path, str]:
        video_cfg = self.config.get("video", {})

        print(f"[Phase 0] Sampling frames from {video_path}")
        frames = sample_frames(video_path, video_cfg.get("sample_fps", 1.0))
        print(f"  Sampled {len(frames)} frames")

        print("[Phase 1] Running object detection")
        frame_detections = self.detector.detect_batch(frames)
        det_count = sum(len(fd.detections) for fd in frame_detections)
        print(f"  {det_count} total detections across {len(frame_detections)} frames")

        print("[Phase 2] Generating frame-level captions")
        frame_captions = self.frame_captioner.caption_batch(frame_detections)

        print("[Phase 3] Generating clip-level captions")
        clips = group_into_clips(
            frames,
            video_cfg.get("clip_duration_sec", 10),
            video_cfg.get("clip_stride_sec", 5),
        )
        captioned_clips = self.clip_captioner.caption_all_clips(clips, frame_captions)
        print(f"  {len(captioned_clips)} clips captioned")

        print("[Phase 4] Synthesizing report and highlight reel")
        reel_path, report = self.synthesizer.build_highlight_reel(
            video_path, captioned_clips, output_dir
        )
        print(f"  Highlight reel: {reel_path}")
        print(f"  Report written to {Path(output_dir) / 'report.md'}")

        return reel_path, report
