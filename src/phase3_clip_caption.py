from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .phase2_frame_caption import CaptionedFrame
from .phase_recognition import PhaseLabeledFrame, SurgicalPhase
from .video_io import Clip


@dataclass
class CaptionedClip:
    clip: Clip
    frame_captions: list[CaptionedFrame]
    clip_caption: str
    importance_score: float
    surgical_phase: SurgicalPhase | None = None
    phase_confidence: float = 0.0


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
        self,
        clip: Clip,
        frame_captions: list[CaptionedFrame],
        phase_labels: list[PhaseLabeledFrame] | None = None,
    ) -> CaptionedClip:
        phase, phase_conf = self._resolve_clip_phase(phase_labels)
        phase_context = f" [Phase: {phase.display_name}]" if phase else ""

        captions_text = "\n".join(
            f"- [{fc.timestamp_sec:.1f}s] {fc.caption}" for fc in frame_captions
        )
        if phase:
            captions_text = f"Surgical phase: {phase.display_name}\n\n{captions_text}"

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
            clip_caption = (
                f"Clip {clip.start_sec:.0f}-{clip.end_sec:.0f}s{phase_context}: "
                f"{len(frame_captions)} frames"
            )

        importance = self._score_importance(frame_captions, phase)
        return CaptionedClip(
            clip=clip,
            frame_captions=frame_captions,
            clip_caption=clip_caption,
            importance_score=importance,
            surgical_phase=phase,
            phase_confidence=phase_conf,
        )

    def _resolve_clip_phase(
        self, phase_labels: list[PhaseLabeledFrame] | None
    ) -> tuple[SurgicalPhase | None, float]:
        if not phase_labels:
            return None, 0.0
        counts = Counter(pl.phase_label.phase for pl in phase_labels)
        dominant_phase = counts.most_common(1)[0][0]
        avg_conf = sum(
            pl.phase_label.confidence
            for pl in phase_labels
            if pl.phase_label.phase == dominant_phase
        ) / max(counts[dominant_phase], 1)
        return dominant_phase, avg_conf

    def _score_importance(
        self,
        frame_captions: list[CaptionedFrame],
        phase: SurgicalPhase | None = None,
    ) -> float:
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

        base_score = 0.4 * object_diversity + 0.4 * detection_density

        phase_weight = phase.highlight_weight if phase else 0.5
        return base_score + 0.2 * phase_weight

    def caption_all_clips(
        self,
        clips: list[Clip],
        all_frame_captions: list[CaptionedFrame],
        all_phase_labels: list[PhaseLabeledFrame] | None = None,
    ) -> list[CaptionedClip]:
        results = []
        caption_by_ts = {fc.timestamp_sec: fc for fc in all_frame_captions}
        phase_by_ts = (
            {pl.timestamp_sec: pl for pl in all_phase_labels}
            if all_phase_labels
            else {}
        )

        for clip in clips:
            matched = [
                caption_by_ts[f.timestamp_sec]
                for f in clip.frames
                if f.timestamp_sec in caption_by_ts
            ]
            clip_phases = [
                phase_by_ts[f.timestamp_sec]
                for f in clip.frames
                if f.timestamp_sec in phase_by_ts
            ] or None
            results.append(self.caption_clip(clip, matched, clip_phases))
        return results
