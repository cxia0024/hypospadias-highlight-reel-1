from dataclasses import dataclass

import numpy as np

from .video_io import Frame


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class FrameDetections:
    frame: Frame
    detections: list[Detection]

    @property
    def detected_classes(self) -> list[str]:
        return list({d.class_name for d in self.detections})


class ObjectDetector:
    def __init__(self, config: dict):
        self.confidence_threshold = config.get("confidence_threshold", 0.4)
        self.model = self._load_model(config)

    def _load_model(self, config: dict):
        weights = config.get("weights", "")
        model_type = config.get("model", "yolov8")
        try:
            from ultralytics import YOLO
            return YOLO(weights)
        except (ImportError, Exception):
            return None

    def detect(self, frame: Frame) -> FrameDetections:
        if self.model is None:
            return FrameDetections(frame=frame, detections=[])

        results = self.model(frame.image, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue
                cls_id = int(box.cls[0])
                cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        class_name=cls_name,
                        confidence=conf,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                    )
                )
        return FrameDetections(frame=frame, detections=detections)

    def detect_batch(self, frames: list[Frame]) -> list[FrameDetections]:
        return [self.detect(f) for f in frames]
