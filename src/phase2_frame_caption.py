from dataclasses import dataclass

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
        self.max_length = config.get("max_caption_length", 80)
        self.model = self._load_model(config)

    def _load_model(self, config: dict):
        model_name = config.get("model", "blip2")
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            import torch

            processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b"
            )
            return {"processor": processor, "model": model}
        except (ImportError, Exception):
            return None

    def caption(self, frame_det: FrameDetections) -> CaptionedFrame:
        objects_str = ", ".join(frame_det.detected_classes) or "surgical field"
        prompt = self.prompt_template.format(detected_objects=objects_str)

        if self.model is not None:
            import torch
            from PIL import Image

            image = Image.fromarray(frame_det.frame.image[:, :, ::-1])
            inputs = self.model["processor"](image, text=prompt, return_tensors="pt")
            with torch.no_grad():
                out = self.model["model"].generate(**inputs, max_new_tokens=self.max_length)
            caption = self.model["processor"].decode(out[0], skip_special_tokens=True)
        else:
            caption = f"[{frame_det.frame.timestamp_sec:.1f}s] {prompt}"

        return CaptionedFrame(
            frame_detections=frame_det,
            caption=caption,
            timestamp_sec=frame_det.frame.timestamp_sec,
        )

    def caption_batch(self, detections: list[FrameDetections]) -> list[CaptionedFrame]:
        return [self.caption(d) for d in detections]
