from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .phase1_detection import FrameDetections


class SurgicalPhase(Enum):
    PREPARATION = "preparation"
    DEGLOVING = "degloving"
    URETHRAL_PLATE_INCISION = "urethral_plate_incision"
    TUBULARIZATION = "tubularization"
    WATERPROOFING = "waterproofing"
    GLANSPLASTY = "glansplasty"
    SKIN_CLOSURE = "skin_closure"
    DRESSING = "dressing"

    @property
    def display_name(self) -> str:
        return self.value.replace("_", " ").title()

    @property
    def highlight_weight(self) -> float:
        weights = {
            SurgicalPhase.PREPARATION: 0.2,
            SurgicalPhase.DEGLOVING: 0.5,
            SurgicalPhase.URETHRAL_PLATE_INCISION: 0.8,
            SurgicalPhase.TUBULARIZATION: 1.0,
            SurgicalPhase.WATERPROOFING: 0.7,
            SurgicalPhase.GLANSPLASTY: 0.9,
            SurgicalPhase.SKIN_CLOSURE: 0.6,
            SurgicalPhase.DRESSING: 0.1,
        }
        return weights[self]


PHASE_ORDER = list(SurgicalPhase)


@dataclass
class PhaseLabel:
    phase: SurgicalPhase
    confidence: float
    probabilities: dict[SurgicalPhase, float]


@dataclass
class PhaseLabeledFrame:
    frame_detections: FrameDetections
    phase_label: PhaseLabel

    @property
    def timestamp_sec(self) -> float:
        return self.frame_detections.frame.timestamp_sec


PHASE_INSTRUMENT_PRIORS: dict[SurgicalPhase, dict[str, float]] = {
    SurgicalPhase.PREPARATION: {
        "catheter": 0.6, "forceps": 0.3,
    },
    SurgicalPhase.DEGLOVING: {
        "scissors": 0.7, "forceps": 0.6, "skin_flap": 0.4,
    },
    SurgicalPhase.URETHRAL_PLATE_INCISION: {
        "scissors": 0.5, "urethral_plate": 0.9, "forceps": 0.4,
    },
    SurgicalPhase.TUBULARIZATION: {
        "needle_driver": 0.9, "suture": 0.9, "urethral_plate": 0.7, "catheter": 0.5,
    },
    SurgicalPhase.WATERPROOFING: {
        "needle_driver": 0.7, "suture": 0.7, "dartos_flap": 0.9, "forceps": 0.4,
    },
    SurgicalPhase.GLANSPLASTY: {
        "needle_driver": 0.8, "suture": 0.8, "glans": 0.9, "forceps": 0.5,
    },
    SurgicalPhase.SKIN_CLOSURE: {
        "needle_driver": 0.7, "suture": 0.8, "skin_flap": 0.8, "scissors": 0.3,
    },
    SurgicalPhase.DRESSING: {
        "catheter": 0.5,
    },
}


class PhaseRecognizer:
    def __init__(self, config: dict):
        self.model_type = config.get("model", "temporal_cnn")
        self.temporal_window = config.get("temporal_smoothing_window", 15)
        self.transition_penalty = config.get("transition_penalty", 0.3)
        self.model = self._load_model(config)

    def _load_model(self, config: dict):
        weights = config.get("weights", "")
        try:
            import torch
            import torch.nn as nn

            class TemporalPhaseCNN(nn.Module):
                def __init__(self, num_phases: int = len(SurgicalPhase)):
                    super().__init__()
                    self.backbone = self._build_backbone()
                    self.temporal = nn.GRU(
                        input_size=512, hidden_size=256,
                        num_layers=2, batch_first=True, bidirectional=True,
                    )
                    self.classifier = nn.Sequential(
                        nn.Linear(512, 128),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(128, num_phases),
                    )

                def _build_backbone(self):
                    from torchvision.models import resnet18
                    backbone = resnet18(weights=None)
                    backbone.fc = nn.Linear(backbone.fc.in_features, 512)
                    return backbone

                def forward(self, x):
                    b, t, c, h, w = x.shape
                    x = x.view(b * t, c, h, w)
                    features = self.backbone(x)
                    features = features.view(b, t, -1)
                    temporal_out, _ = self.temporal(features)
                    logits = self.classifier(temporal_out)
                    return logits

            model = TemporalPhaseCNN()
            if weights:
                state = torch.load(weights, map_location="cpu", weights_only=True)
                model.load_state_dict(state)
            model.eval()
            return model
        except (ImportError, FileNotFoundError, Exception):
            return None

    def recognize_frames(
        self, frame_detections: list[FrameDetections]
    ) -> list[PhaseLabeledFrame]:
        if self.model is not None:
            return self._recognize_with_model(frame_detections)
        return self._recognize_heuristic(frame_detections)

    def _recognize_with_model(
        self, frame_detections: list[FrameDetections]
    ) -> list[PhaseLabeledFrame]:
        import torch
        from PIL import Image
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        tensors = []
        for fd in frame_detections:
            img = Image.fromarray(fd.frame.image[:, :, ::-1])
            tensors.append(transform(img))

        sequence = torch.stack(tensors).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(sequence)
            probs = torch.softmax(logits.squeeze(0), dim=-1).numpy()

        probs = self._temporal_smooth(probs)
        probs = self._apply_monotonic_prior(probs)

        results = []
        for i, fd in enumerate(frame_detections):
            phase_probs = {
                phase: float(probs[i, j]) for j, phase in enumerate(PHASE_ORDER)
            }
            best_idx = int(np.argmax(probs[i]))
            best_phase = PHASE_ORDER[best_idx]
            label = PhaseLabel(
                phase=best_phase,
                confidence=float(probs[i, best_idx]),
                probabilities=phase_probs,
            )
            results.append(PhaseLabeledFrame(frame_detections=fd, phase_label=label))
        return results

    def _recognize_heuristic(
        self, frame_detections: list[FrameDetections]
    ) -> list[PhaseLabeledFrame]:
        n = len(frame_detections)
        if n == 0:
            return []

        raw_probs = np.zeros((n, len(PHASE_ORDER)))

        for i, fd in enumerate(frame_detections):
            detected = set(d.class_name for d in fd.detections)
            position_ratio = i / max(n - 1, 1)

            for j, phase in enumerate(PHASE_ORDER):
                expected_position = j / (len(PHASE_ORDER) - 1)
                position_score = max(0, 1.0 - 2.0 * abs(position_ratio - expected_position))

                instrument_score = 0.0
                priors = PHASE_INSTRUMENT_PRIORS.get(phase, {})
                if priors:
                    matches = sum(priors[cls] for cls in detected if cls in priors)
                    instrument_score = min(matches / max(len(priors), 1), 1.0)

                raw_probs[i, j] = 0.4 * position_score + 0.6 * instrument_score

        row_sums = raw_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        raw_probs /= row_sums

        smoothed = self._temporal_smooth(raw_probs)
        smoothed = self._apply_monotonic_prior(smoothed)

        results = []
        for i, fd in enumerate(frame_detections):
            phase_probs = {
                phase: float(smoothed[i, j]) for j, phase in enumerate(PHASE_ORDER)
            }
            best_idx = int(np.argmax(smoothed[i]))
            best_phase = PHASE_ORDER[best_idx]
            label = PhaseLabel(
                phase=best_phase,
                confidence=float(smoothed[i, best_idx]),
                probabilities=phase_probs,
            )
            results.append(PhaseLabeledFrame(frame_detections=fd, phase_label=label))
        return results

    def _temporal_smooth(self, probs: np.ndarray) -> np.ndarray:
        n = probs.shape[0]
        if n <= 1:
            return probs

        smoothed = np.copy(probs)
        half_w = self.temporal_window // 2
        for i in range(n):
            start = max(0, i - half_w)
            end = min(n, i + half_w + 1)
            smoothed[i] = probs[start:end].mean(axis=0)

        row_sums = smoothed.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        smoothed /= row_sums
        return smoothed

    def _apply_monotonic_prior(self, probs: np.ndarray) -> np.ndarray:
        n, num_phases = probs.shape
        if n <= 1:
            return probs

        adjusted = np.copy(probs)
        penalty = self.transition_penalty

        for i in range(1, n):
            prev_best = int(np.argmax(adjusted[i - 1]))
            for j in range(num_phases):
                if j < prev_best:
                    adjusted[i, j] *= (1.0 - penalty)

        row_sums = adjusted.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        adjusted /= row_sums
        return adjusted
