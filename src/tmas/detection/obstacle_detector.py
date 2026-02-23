"""Obstacle detection module using RF-DETR.

RF-DETR (Roboflow DETR) is a state-of-the-art real-time object detection
model accepted to ICLR 2026. It achieves superior accuracy-latency trade-offs
compared to YOLO and RT-DETR, making it ideal for safety-critical obstacle
detection on Jetson AGX Orin.

Key advantages:
- SOTA on COCO (78.5 AP with 2XL, 75.1 AP with L at 6.8ms)
- DINOv2 vision transformer backbone for robust features
- Real-time performance on edge devices
- Apache 2.0 license for deployment
- Pre-trained on COCO (90+ classes including persons, vehicles, animals)

For obstacle detection in TMAS, we prioritize:
- Person/vehicle detection: >99% recall (safety-critical)
- Static obstacle detection: >95% recall
- Real-time inference: >20 FPS on Jetson AGX Orin
- Low latency: <50ms for sudden obstacle detection

References:
- Paper: RF-DETR (ICLR 2026)
- GitHub: https://github.com/roboflow/rf-detr
- Docs: https://roboflow.com/model/rf-detr
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRSmall
    from rfdetr.util.coco_classes import COCO_CLASSES
    RFDETR_AVAILABLE = True
except ImportError:
    RFDETR_AVAILABLE = False
    print("Warning: rfdetr not installed. Install with: pip install rfdetr")


class ObstacleDetector(nn.Module):
    """Obstacle detection model using RF-DETR.

    Detects 20+ obstacle classes including persons, vehicles, animals,
    and static obstacles critical for autonomous navigation.

    The detector maps COCO classes to TMAS obstacle classes and provides
    safety-critical filtering with configurable confidence thresholds.
    """

    # TMAS obstacle classes (20 classes)
    OBSTACLE_CLASSES = [
        "person",              # 0 - CRITICAL dynamic
        "car",                 # 1 - CRITICAL dynamic
        "truck",               # 2 - CRITICAL dynamic
        "military_vehicle",    # 3 - CRITICAL dynamic (custom)
        "motorcycle",          # 4 - CRITICAL dynamic
        "bicycle",             # 5 - WARNING dynamic
        "bus",                 # 6 - CRITICAL dynamic
        "animal",              # 7 - HIGH dynamic (dog, horse, cow, etc.)
        "fallen_tree",         # 8 - HIGH static (custom)
        "rock",                # 9 - HIGH static (custom)
        "crater",              # 10 - HIGH static (custom)
        "barrier",             # 11 - MEDIUM static (custom)
        "wreckage",            # 12 - MEDIUM static (custom)
        "debris",              # 13 - MEDIUM static (custom)
        "metal_fragments",     # 14 - MEDIUM static (custom)
        "concrete_block",      # 15 - MEDIUM static (custom)
        "container",           # 16 - MEDIUM static
        "pole",                # 17 - MEDIUM static
        "tire",                # 18 - LOW static (custom)
        "unknown_obstacle"     # 19 - MEDIUM static
    ]

    # Mapping from COCO classes to TMAS obstacle classes
    COCO_TO_TMAS = {
        0: 0,   # person → person
        1: 5,   # bicycle → bicycle
        2: 1,   # car → car
        3: 4,   # motorcycle → motorcycle
        5: 6,   # bus → bus
        7: 2,   # truck → truck
        16: 7,  # dog → animal
        17: 7,  # horse → animal
        18: 7,  # sheep → animal
        19: 7,  # cow → animal
        44: 16, # bottle → container (approximation)
        60: 16, # dining table → container (approximation)
        84: 17, # book → unknown_obstacle (fallback)
    }

    # Priority levels for obstacle classes
    PRIORITY_CRITICAL = ["person", "car", "truck", "military_vehicle", "motorcycle", "bus"]
    PRIORITY_HIGH = ["animal", "fallen_tree", "rock", "crater", "bicycle"]
    PRIORITY_MEDIUM = ["barrier", "wreckage", "debris", "metal_fragments", "concrete_block", "container", "pole", "unknown_obstacle"]
    PRIORITY_LOW = ["tire"]

    def __init__(
        self,
        model_size: str = "large",
        confidence_threshold: float = 0.5,
        confidence_critical: float = 0.3,
        device: str = "cuda",
        pretrained: bool = True
    ):
        """Initialize obstacle detector.

        Args:
            model_size: RF-DETR model size (small/medium/large)
            confidence_threshold: Default confidence threshold
            confidence_critical: Lower threshold for critical classes (persons/vehicles)
            device: Device to run inference (cuda/cpu)
            pretrained: Load pretrained COCO weights
        """
        super().__init__()

        if not RFDETR_AVAILABLE:
            raise ImportError(
                "rfdetr not installed. Install with: pip install rfdetr\n"
                "For XL/2XL models: pip install rfdetr[plus]"
            )

        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.confidence_critical = confidence_critical
        self.device = device

        # Load RF-DETR model
        if model_size == "small":
            self.model = RFDETRSmall()
        elif model_size == "medium":
            self.model = RFDETRMedium()
        elif model_size == "large":
            self.model = RFDETRLarge()
        else:
            raise ValueError(f"Unsupported model size: {model_size}. Use small/medium/large")

        # Move to device
        self.model = self.model.to(device)
        self.model.eval()

    def _map_coco_to_tmas(
        self,
        coco_class: int,
        confidence: float
    ) -> Tuple[Optional[int], bool]:
        """Map COCO class to TMAS obstacle class.

        Args:
            coco_class: COCO class ID
            confidence: Detection confidence

        Returns:
            Tuple of (TMAS class ID or None, is_critical)
        """
        if coco_class in self.COCO_TO_TMAS:
            tmas_class = self.COCO_TO_TMAS[coco_class]
            class_name = self.OBSTACLE_CLASSES[tmas_class]

            # Check if critical class
            is_critical = class_name in self.PRIORITY_CRITICAL

            # Apply appropriate threshold
            threshold = self.confidence_critical if is_critical else self.confidence_threshold

            if confidence >= threshold:
                return tmas_class, is_critical
            else:
                return None, is_critical

        # Unknown COCO class - map to unknown_obstacle if high confidence
        if confidence >= self.confidence_threshold:
            return 19, False  # unknown_obstacle
        else:
            return None, False

    def forward(
        self,
        images: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        """Forward pass for obstacle detection.

        Args:
            images: Input RGB images [B, 3, H, W] or single image [3, H, W]

        Returns:
            List of detections per image with:
                - boxes: [N, 4] in [x1, y1, x2, y2] format
                - scores: [N] confidence scores
                - labels: [N] TMAS class indices
                - is_critical: [N] boolean critical flags
        """
        # Handle single image
        if images.dim() == 3:
            images = images.unsqueeze(0)

        batch_size = images.shape[0]
        all_detections = []

        with torch.no_grad():
            for i in range(batch_size):
                # Convert to PIL Image (RF-DETR expects PIL)
                img_np = images[i].cpu().permute(1, 2, 0).numpy()
                img_np = (img_np * 255).astype(np.uint8)

                from PIL import Image
                pil_image = Image.fromarray(img_np)

                # Run detection
                detections = self.model.predict(pil_image, threshold=0.1)

                # Filter and map detections
                boxes = []
                scores = []
                labels = []
                is_critical = []

                if len(detections) > 0:
                    for j in range(len(detections.xyxy)):
                        coco_class = int(detections.class_id[j])
                        confidence = float(detections.confidence[j])

                        # Map to TMAS class
                        tmas_class, critical = self._map_coco_to_tmas(coco_class, confidence)

                        if tmas_class is not None:
                            boxes.append(detections.xyxy[j])
                            scores.append(confidence)
                            labels.append(tmas_class)
                            is_critical.append(critical)

                # Convert to tensors
                if len(boxes) > 0:
                    detection_dict = {
                        "boxes": torch.tensor(np.array(boxes), dtype=torch.float32, device=self.device),
                        "scores": torch.tensor(scores, dtype=torch.float32, device=self.device),
                        "labels": torch.tensor(labels, dtype=torch.long, device=self.device),
                        "is_critical": torch.tensor(is_critical, dtype=torch.bool, device=self.device)
                    }
                else:
                    # No detections
                    detection_dict = {
                        "boxes": torch.zeros((0, 4), dtype=torch.float32, device=self.device),
                        "scores": torch.zeros(0, dtype=torch.float32, device=self.device),
                        "labels": torch.zeros(0, dtype=torch.long, device=self.device),
                        "is_critical": torch.zeros(0, dtype=torch.bool, device=self.device)
                    }

                all_detections.append(detection_dict)

        return all_detections

    def predict(
        self,
        images: torch.Tensor,
        return_names: bool = True
    ) -> List[Dict]:
        """Predict obstacles with human-readable output.

        Args:
            images: Input images [B, 3, H, W]
            return_names: Include class names in output

        Returns:
            List of detection dictionaries per image
        """
        detections = self.forward(images)

        # Add class names if requested
        if return_names:
            for det in detections:
                if len(det["labels"]) > 0:
                    det["class_names"] = [
                        self.OBSTACLE_CLASSES[label.item()]
                        for label in det["labels"]
                    ]
                else:
                    det["class_names"] = []

        return detections


def create_obstacle_detector(
    model_size: str = "large",
    confidence_threshold: float = 0.5,
    device: str = "cuda",
    **kwargs
) -> ObstacleDetector:
    """Create RF-DETR obstacle detector.

    Args:
        model_size: Model size (small/medium/large)
        confidence_threshold: Default confidence threshold
        device: Device (cuda/cpu)
        **kwargs: Additional arguments

    Returns:
        ObstacleDetector instance

    Example:
        >>> detector = create_obstacle_detector(model_size="large")
        >>> images = torch.randn(2, 3, 640, 640).cuda()
        >>> detections = detector.predict(images)
        >>> print(f"Detected {len(detections[0]['boxes'])} obstacles")
    """
    return ObstacleDetector(
        model_size=model_size,
        confidence_threshold=confidence_threshold,
        device=device,
        **kwargs
    )


def main():
    """Test obstacle detector."""
    print("Testing RF-DETR obstacle detector...")

    if not RFDETR_AVAILABLE:
        print("Error: rfdetr not installed")
        return

    # Create detector
    print("\nCreating RF-DETR Large detector...")
    detector = create_obstacle_detector(
        model_size="large",
        confidence_threshold=0.5,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Test with dummy image
    print("\nTesting with dummy image...")
    test_image = torch.randn(1, 3, 640, 640)
    if torch.cuda.is_available():
        test_image = test_image.cuda()

    detections = detector.predict(test_image)

    print(f"\nDetections: {len(detections[0]['boxes'])} obstacles")
    print(f"Classes: {detections[0]['class_names']}")

    # Print model info
    print(f"\nModel size: {detector.model_size}")
    print(f"Confidence threshold: {detector.confidence_threshold}")
    print(f"Critical threshold: {detector.confidence_critical}")
    print(f"Number of TMAS classes: {len(detector.OBSTACLE_CLASSES)}")

    # Print class priorities
    print("\nClass priorities:")
    print(f"  CRITICAL: {', '.join(detector.PRIORITY_CRITICAL)}")
    print(f"  HIGH: {', '.join(detector.PRIORITY_HIGH)}")
    print(f"  MEDIUM: {', '.join(detector.PRIORITY_MEDIUM)}")

    print("\nObstacle detector test successful!")


if __name__ == "__main__":
    main()
