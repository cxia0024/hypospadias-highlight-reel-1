from dataclasses import dataclass
from pathlib import Path

from .phase3_clip_caption import CaptionedClip
from .video_io import write_highlight_reel


@dataclass
class SurgicalReport:
    summary: str
    phase_timeline: list[dict]
    key_moments: list[dict]
    instrument_usage: dict[str, float]
    highlight_segments: list[tuple[float, float]]


class Synthesizer:
    def __init__(self, config: dict):
        self.max_duration = config.get("highlight_reel", {}).get("max_duration_sec", 120)
        self.min_score = config.get("highlight_reel", {}).get("min_clip_score", 0.6)
        self.strategy = config.get("highlight_reel", {}).get(
            "selection_strategy", "importance_score"
        )
        self.report_format = config.get("report", {}).get("format", "markdown")

    def select_highlights(
        self, captioned_clips: list[CaptionedClip]
    ) -> list[CaptionedClip]:
        eligible = [c for c in captioned_clips if c.importance_score >= self.min_score]

        if self.strategy == "importance_score":
            eligible.sort(key=lambda c: c.importance_score, reverse=True)
        elif self.strategy == "phase_balanced":
            eligible.sort(key=lambda c: c.clip.start_sec)

        selected = []
        total_duration = 0.0
        for clip in eligible:
            clip_dur = clip.clip.end_sec - clip.clip.start_sec
            if total_duration + clip_dur > self.max_duration:
                continue
            selected.append(clip)
            total_duration += clip_dur

        selected.sort(key=lambda c: c.clip.start_sec)
        return selected

    def generate_report(self, captioned_clips: list[CaptionedClip]) -> SurgicalReport:
        highlights = self.select_highlights(captioned_clips)

        instrument_counts: dict[str, int] = {}
        for clip in captioned_clips:
            for fc in clip.frame_captions:
                for det in fc.frame_detections.detections:
                    instrument_counts[det.class_name] = (
                        instrument_counts.get(det.class_name, 0) + 1
                    )

        total = max(sum(instrument_counts.values()), 1)
        instrument_usage = {k: v / total for k, v in instrument_counts.items()}

        total_video_sec = (
            captioned_clips[-1].clip.end_sec if captioned_clips else 0.0
        )
        summary = (
            f"Hypospadias repair procedure spanning {total_video_sec:.0f} seconds. "
            f"Identified {len(captioned_clips)} clips, selected {len(highlights)} "
            f"for the highlight reel ({sum(h.clip.end_sec - h.clip.start_sec for h in highlights):.0f}s)."
        )

        timeline = [
            {
                "start": c.clip.start_sec,
                "end": c.clip.end_sec,
                "caption": c.clip_caption,
                "score": c.importance_score,
            }
            for c in captioned_clips
        ]

        key_moments = [
            {
                "time": h.clip.start_sec,
                "caption": h.clip_caption,
                "score": h.importance_score,
            }
            for h in highlights
        ]

        segments = [(h.clip.start_sec, h.clip.end_sec) for h in highlights]

        return SurgicalReport(
            summary=summary,
            phase_timeline=timeline,
            key_moments=key_moments,
            instrument_usage=instrument_usage,
            highlight_segments=segments,
        )

    def render_report_markdown(self, report: SurgicalReport) -> str:
        lines = ["# Surgical Highlight Report\n"]
        lines.append(f"## Summary\n\n{report.summary}\n")

        lines.append("## Key Moments\n")
        for m in report.key_moments:
            lines.append(f"- **{m['time']:.0f}s** (score {m['score']:.2f}): {m['caption']}")
        lines.append("")

        lines.append("## Instrument Usage\n")
        for instr, pct in sorted(report.instrument_usage.items(), key=lambda x: -x[1]):
            lines.append(f"- {instr}: {pct:.1%}")
        lines.append("")

        lines.append("## Full Timeline\n")
        for entry in report.phase_timeline:
            lines.append(
                f"| {entry['start']:.0f}-{entry['end']:.0f}s | {entry['caption']} | {entry['score']:.2f} |"
            )
        return "\n".join(lines)

    def build_highlight_reel(
        self,
        source_video: str | Path,
        captioned_clips: list[CaptionedClip],
        output_dir: str | Path,
    ) -> tuple[Path, str]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report = self.generate_report(captioned_clips)

        reel_path = write_highlight_reel(
            source_video, report.highlight_segments, output_dir / "highlight_reel.mp4"
        )

        report_text = self.render_report_markdown(report)
        report_path = output_dir / "report.md"
        report_path.write_text(report_text)

        return reel_path, report_text
