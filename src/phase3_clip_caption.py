from dataclasses import dataclass

from .phase2_frame_caption import CaptionedFrame
from .video_io import Clip


@dataclass
class CaptionedClip:
    clip: Clip
    frame_captions: list[CaptionedFrame]
    clip_caption: str
    importance_score: float


class ClipCaptioner:
    def __init__(self, config: dict):
        self.llm_name = config.get("llm_name", "gpt-4o-mini")
        self.prompt_template = config.get(
            "prompt_template",
            "Summarize these consecutive frame descriptions into one coherent surgical action.",
        )
        self.client = self._init_client()

    def _init_client(self):
        try:
            import openai
            return openai.OpenAI()
        except (ImportError, Exception):
            return None

    def caption_clip(
        self, clip: Clip, frame_captions: list[CaptionedFrame]
    ) -> CaptionedClip:
        captions_text = "\n".join(
            f"- [{fc.timestamp_sec:.1f}s] {fc.caption}" for fc in frame_captions
        )

        if self.client is not None:
            response = self.client.chat.completions.create(
                model=self.llm_name,
                messages=[
                    {"role": "system", "content": self.prompt_template},
                    {"role": "user", "content": captions_text},
                ],
                max_tokens=150,
            )
            clip_caption = response.choices[0].message.content.strip()
        else:
            clip_caption = f"Clip {clip.start_sec:.0f}-{clip.end_sec:.0f}s: {len(frame_captions)} frames"

        importance = self._score_importance(frame_captions)
        return CaptionedClip(
            clip=clip,
            frame_captions=frame_captions,
            clip_caption=clip_caption,
            importance_score=importance,
        )

    def _score_importance(self, frame_captions: list[CaptionedFrame]) -> float:
        if not frame_captions:
            return 0.0

        unique_objects = set()
        for fc in frame_captions:
            unique_objects.update(fc.frame_detections.detected_classes)

        object_diversity = min(len(unique_objects) / 5.0, 1.0)
        detection_density = min(
            sum(len(fc.frame_detections.detections) for fc in frame_captions)
            / (len(frame_captions) * 3.0),
            1.0,
        )
        return 0.5 * object_diversity + 0.5 * detection_density

    def caption_all_clips(
        self,
        clips: list[Clip],
        all_frame_captions: list[CaptionedFrame],
    ) -> list[CaptionedClip]:
        results = []
        caption_by_ts = {fc.timestamp_sec: fc for fc in all_frame_captions}

        for clip in clips:
            matched = [
                caption_by_ts[f.timestamp_sec]
                for f in clip.frames
                if f.timestamp_sec in caption_by_ts
            ]
            results.append(self.caption_clip(clip, matched))
        return results
