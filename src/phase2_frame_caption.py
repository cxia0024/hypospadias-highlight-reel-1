from __future__ import annotations

import base64
from dataclasses import dataclass

import cv2

from .phase1_detection import FrameDetections


@dataclass
class CaptionedFrame:
    frame_detections: FrameDetections
    caption: str
    timestamp_sec: float


class FrameCaptioner:
    def __init__(self, config: dict):
        self.prompt_template = config.get(
            "prompt_template",
            "Describe the surgical scene showing {detected_objects}.",
        )
        self.max_tokens = config.get("max_tokens", 80)
        self.model_name = config.get("model_name", "claude-sonnet-4-20250514")
        self.client = self._init_client()

    def _init_client(self):
        try:
            import anthropic
            return anthropic.Anthropic()
        except (ImportError, Exception):
            return None

    def _encode_frame(self, image) -> str:
        _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.standard_b64encode(buf.tobytes()).decode("utf-8")

    def caption(self, frame_det: FrameDetections) -> CaptionedFrame:
        objects_str = ", ".join(frame_det.detected_classes) or "surgical field"
        prompt = self.prompt_template.format(detected_objects=objects_str)

        if self.client is not None:
            image_b64 = self._encode_frame(frame_det.frame.image)
            response = self.client.messages.create(
                model=self.model_name,
                system="You are a surgical video analysis assistant. Provide concise, medically accurate frame descriptions.",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_b64,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=self.max_tokens,
            )
            caption = response.content[0].text.strip()
        else:
            caption = f"[{frame_det.frame.timestamp_sec:.1f}s] {prompt}"

        return CaptionedFrame(
            frame_detections=frame_det,
            caption=caption,
            timestamp_sec=frame_det.frame.timestamp_sec,
        )

    def caption_batch(self, detections: list[FrameDetections]) -> list[CaptionedFrame]:
        return [self.caption(d) for d in detections]
