from __future__ import annotations

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


DEFAULT_CLASSES = [
    "scissors", "forceps", "needle_driver", "suture", "catheter",
    "urethral_plate", "glans", "skin_flap", "dartos_flap",
]


class SurgicalViT:
    """
    ViT-based multi-label object detector for surgical frames.

    Architecture (from paper):
      1. Image A ∈ R^{n×m} → extract patches P_k ∈ R^{a×b}
         P_k = A[i:i+a-1, j:j+b-1], k = i × (m/b) + j
      2. Flatten patches → P̃_k, prepend positional index
         I = [[1, P̃_1], [2, P̃_2], ..., [N, P̃_N]] ∈ R^{1×(16×16+1)×196}
      3. ViT embedding → softmax → probability vector P per class
         P = softmax(ViT(I))
      4. Threshold → detected objects O = {object_i | P_i > threshold}
      5. Weighted BCE loss for class imbalance:
         w_i = 1 / f_i (inverse class frequency)
    """

    def __init__(self, config: dict):
        self.patch_size = config.get("patch_size", 16)
        self.image_size = config.get("image_size", 224)
        self.num_classes = len(config.get("classes", DEFAULT_CLASSES))
        self.class_names = config.get("classes", DEFAULT_CLASSES)
        self.threshold = config.get("confidence_threshold", 0.5)
        self.model = self._build_model(config)

    def _build_model(self, config: dict):
        weights_path = config.get("weights", "")
        try:
            import torch
            import torch.nn as nn

            model = ViTDetector(
                image_size=self.image_size,
                patch_size=self.patch_size,
                num_classes=self.num_classes,
                embed_dim=config.get("embed_dim", 768),
                num_heads=config.get("num_heads", 12),
                num_layers=config.get("num_layers", 12),
                mlp_dim=config.get("mlp_dim", 3072),
                dropout=config.get("dropout", 0.1),
            )
            if weights_path:
                state = torch.load(weights_path, map_location="cpu", weights_only=True)
                model.load_state_dict(state)
            model.eval()
            return model
        except (ImportError, FileNotFoundError, Exception):
            return None

    def predict(self, image: np.ndarray) -> list[tuple[str, float]]:
        if self.model is None:
            return []

        import torch
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        img_rgb = image[:, :, ::-1].copy()
        tensor = transform(img_rgb).unsqueeze(0)

        with torch.no_grad():
            probs = self.model(tensor).squeeze(0)

        results = []
        for i, prob in enumerate(probs.tolist()):
            if prob >= self.threshold and i < len(self.class_names):
                results.append((self.class_names[i], prob))
        return results


class ViTDetector:
    """PyTorch ViT module — only instantiated when torch is available."""

    def __new__(cls, **kwargs):
        import torch.nn as nn

        class _ViTDetector(nn.Module):
            def __init__(
                self,
                image_size: int = 224,
                patch_size: int = 16,
                num_classes: int = 9,
                embed_dim: int = 768,
                num_heads: int = 12,
                num_layers: int = 12,
                mlp_dim: int = 3072,
                dropout: float = 0.1,
            ):
                super().__init__()
                self.patch_size = patch_size
                num_patches = (image_size // patch_size) ** 2  # N = (224/16)^2 = 196
                patch_dim = 3 * patch_size * patch_size  # flattened P̃_k dimension

                # Eq 2: patch embedding — flatten P̃_k and project to embed_dim
                self.patch_embed = nn.Linear(patch_dim, embed_dim)

                # Eq 2: positional encoding [k, P̃_k] — learnable
                self.pos_embed = nn.Parameter(
                    torch.randn(1, num_patches, embed_dim) * 0.02
                )

                self.dropout = nn.Dropout(dropout)

                # ViT encoder blocks
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=mlp_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer, num_layers=num_layers
                )

                self.norm = nn.LayerNorm(embed_dim)

                # Eq 3-4: classification head — multi-label (one logit per class)
                self.classifier = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim // 2, num_classes),
                )

            def extract_patches(self, x):
                """
                Eq 1: Extract patches P_k = A[i:i+a-1, j:j+b-1]
                where k = i × (m/b) + j

                Input:  x of shape (B, 3, H, W)
                Output: patches of shape (B, N, patch_dim)
                """
                B, C, H, W = x.shape
                p = self.patch_size
                # unfold extracts sliding windows; step=p gives non-overlapping patches
                patches = x.unfold(2, p, p).unfold(3, p, p)  # (B, C, H/p, W/p, p, p)
                patches = patches.contiguous().view(B, C, -1, p, p)  # (B, C, N, p, p)
                patches = patches.permute(0, 2, 1, 3, 4)  # (B, N, C, p, p)
                patches = patches.contiguous().view(B, -1, C * p * p)  # (B, N, patch_dim)
                return patches

            def forward(self, x):
                """
                Full forward pass implementing Equations 1-4:
                  1. Extract patches P_k
                  2. Embed + add positional encoding → I
                  3. ViT encoder → embedding
                  4. Classifier → sigmoid → P (probabilities per class)
                """
                import torch

                # Eq 1: patch extraction
                patches = self.extract_patches(x)  # (B, N, patch_dim)

                # Eq 2: linear projection + positional encoding
                embeddings = self.patch_embed(patches) + self.pos_embed
                embeddings = self.dropout(embeddings)

                # ViT(I): transformer encoder
                encoded = self.encoder(embeddings)  # (B, N, embed_dim)
                encoded = self.norm(encoded)

                # Global average pool over patch tokens
                pooled = encoded.mean(dim=1)  # (B, embed_dim)

                # Eq 3-4: P = sigmoid(logits), then threshold externally
                logits = self.classifier(pooled)  # (B, num_classes)
                probs = torch.sigmoid(logits)
                return probs

        return _ViTDetector(**kwargs)


class WeightedBCELoss:
    """
    Eq 5: Weighted Binary Cross-Entropy for class-imbalanced surgical data.

    L = -Σ w_i [y_i log σ(ŷ_i) + (1 - y_i) log(1 - σ(ŷ_i))]

    where w_i = 1 / f_i (inverse of class frequency).
    """

    def __init__(self, class_frequencies: list[float]):
        import torch
        weights = [1.0 / max(f, 1e-6) for f in class_frequencies]
        total = sum(weights)
        self.weights = torch.tensor([w / total * len(weights) for w in weights])

    def __call__(self, predictions, targets):
        import torch.nn.functional as F
        self.weights = self.weights.to(predictions.device)
        return F.binary_cross_entropy(
            predictions, targets, weight=self.weights.unsqueeze(0), reduction="mean"
        )


class ObjectDetector:
    def __init__(self, config: dict):
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.classes = config.get("classes", DEFAULT_CLASSES)
        self.model_type = config.get("model", "vit")
        self.vit = SurgicalViT(config) if self.model_type == "vit" else None
        self.yolo = self._load_yolo(config) if self.model_type == "yolov8" else None

    def _load_yolo(self, config: dict):
        try:
            from ultralytics import YOLO
            return YOLO(config.get("weights", ""))
        except (ImportError, Exception):
            return None

    def detect(self, frame: Frame) -> FrameDetections:
        if self.model_type == "vit":
            return self._detect_vit(frame)
        if self.model_type == "yolov8":
            return self._detect_yolo(frame)
        return FrameDetections(frame=frame, detections=[])

    def _detect_vit(self, frame: Frame) -> FrameDetections:
        predictions = self.vit.predict(frame.image) if self.vit else []
        h, w = frame.image.shape[:2]
        detections = [
            Detection(
                class_name=cls_name,
                confidence=conf,
                bbox=(0, 0, w, h),
            )
            for cls_name, conf in predictions
        ]
        return FrameDetections(frame=frame, detections=detections)

    def _detect_yolo(self, frame: Frame) -> FrameDetections:
        if self.yolo is None:
            return FrameDetections(frame=frame, detections=[])
        results = self.yolo(frame.image, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue
                cls_id = int(box.cls[0])
                cls_name = self.yolo.names.get(cls_id, f"class_{cls_id}")
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
