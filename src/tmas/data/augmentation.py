"""Data augmentation pipelines for TMAS training.

This module provides safety-critical augmentation strategies designed to
maximize model recall while maintaining realism. Special focus on preserving
small object features (mines) and testing robustness to environmental conditions.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from typing import Optional, Tuple


class TMASAugmentation:
    """Augmentation pipeline factory for TMAS training."""

    @staticmethod
    def get_terrain_segmentation_train(
        image_size: Tuple[int, int] = (720, 1280),
        normalize: bool = True
    ) -> A.Compose:
        """Get training augmentations for terrain segmentation.

        Args:
            image_size: Target image size (height, width)
            normalize: Apply ImageNet normalization

        Returns:
            Albumentations composition
        """
        transforms = [
            # Geometric transforms
            A.RandomResizedCrop(
                height=image_size[0],
                width=image_size[1],
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                p=1.0
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT,
                p=0.5
            ),

            # Color transforms
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.5
            ),

            # Weather and environmental effects
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.15),
            A.RandomRain(
                slant_lower=-10,
                slant_upper=10,
                drop_length=10,
                drop_width=1,
                p=0.1
            ),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                angle_lower=0,
                angle_upper=1,
                p=0.1
            ),

            # Noise and blur (simulates camera/environmental conditions)
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
            ], p=0.3),

            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.2),

            # Normalization
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0 if normalize else 0.0
            ),

            ToTensorV2()
        ]

        return A.Compose(transforms)

    @staticmethod
    def get_terrain_segmentation_val(
        image_size: Tuple[int, int] = (720, 1280),
        normalize: bool = True
    ) -> A.Compose:
        """Get validation augmentations for terrain segmentation.

        Args:
            image_size: Target image size
            normalize: Apply normalization

        Returns:
            Albumentations composition
        """
        transforms = [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0 if normalize else 0.0
            ),
            ToTensorV2()
        ]

        return A.Compose(transforms)

    @staticmethod
    def get_mine_detection_train(
        image_size: Tuple[int, int] = (720, 1280),
        normalize: bool = True
    ) -> A.Compose:
        """Get training augmentations for mine detection.

        CRITICAL: Designed to preserve small object features while
        maximizing robustness to environmental conditions.

        Args:
            image_size: Target image size
            normalize: Apply normalization

        Returns:
            Albumentations composition with bbox support
        """
        transforms = [
            # Geometric transforms (bbox-aware)
            A.RandomResizedCrop(
                height=image_size[0],
                width=image_size[1],
                scale=(0.9, 1.0),  # Conservative to avoid losing small mines
                ratio=(0.95, 1.05),
                p=0.8
            ),
            A.HorizontalFlip(p=0.5),

            # Small rotation only (mines are typically horizontal)
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=10,  # Limited rotation
                border_mode=cv2.BORDER_REFLECT,
                p=0.5
            ),

            # Perspective transform (simulates camera angle variation)
            A.Perspective(scale=(0.05, 0.1), p=0.3),

            # Color transforms (important for camouflaged mines)
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.6
            ),
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=30,
                val_shift_limit=30,
                p=0.6
            ),

            # Color temperature variation
            A.OneOf([
                A.ToGray(p=1.0),
                A.ToSepia(p=1.0),
            ], p=0.1),

            # Weather effects (critical for real-world robustness)
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.4, p=0.2),
            A.RandomRain(
                slant_lower=-10,
                slant_upper=10,
                drop_length=15,
                p=0.15
            ),
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                p=0.2
            ),

            # Noise (sensor noise, dust, etc.)
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(
                    color_shift=(0.01, 0.05),
                    intensity=(0.1, 0.5),
                    p=1.0
                ),
            ], p=0.4),

            # Slight blur (camera focus, motion)
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.2),

            # Occlusion (grass, debris partially covering mines)
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.2
            ),

            # Normalization
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0 if normalize else 0.0
            ),

            ToTensorV2()
        ]

        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='pascal_voc',  # [x_min, y_min, x_max, y_max]
                label_fields=['labels'],
                min_visibility=0.3,  # Keep bboxes with >30% visible
                min_area=25.0  # Minimum 5x5 pixels
            )
        )

    @staticmethod
    def get_mine_detection_val(
        image_size: Tuple[int, int] = (720, 1280),
        normalize: bool = True
    ) -> A.Compose:
        """Get validation augmentations for mine detection.

        Args:
            image_size: Target image size
            normalize: Apply normalization

        Returns:
            Albumentations composition
        """
        transforms = [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0 if normalize else 0.0
            ),
            ToTensorV2()
        ]

        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels']
            )
        )

    @staticmethod
    def get_multimodal_train(
        image_size: Tuple[int, int] = (720, 1280),
        normalize_rgb: bool = True,
        normalize_thermal: bool = False
    ) -> A.Compose:
        """Get training augmentations for RGB + Thermal fusion.

        Applies synchronized transforms to both modalities.

        Args:
            image_size: Target image size
            normalize_rgb: Normalize RGB with ImageNet stats
            normalize_thermal: Normalize thermal (usually done separately)

        Returns:
            Albumentations composition for 4-channel input
        """
        transforms = [
            # Geometric (affects both RGB and thermal identically)
            A.RandomResizedCrop(
                height=image_size[0],
                width=image_size[1],
                scale=(0.85, 1.0),
                ratio=(0.9, 1.1),
                p=0.8
            ),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=10,
                border_mode=cv2.BORDER_REFLECT,
                p=0.5
            ),

            # RGB-only augmentations (affects first 3 channels)
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.5
            ),

            # Weather (affects both modalities differently)
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.15),

            # Noise
            A.GaussNoise(var_limit=(10.0, 40.0), p=0.3),

            ToTensorV2()
        ]

        return A.Compose(transforms)

    @staticmethod
    def get_test_time_augmentation() -> A.Compose:
        """Get test-time augmentation for inference ensembling.

        Returns:
            Minimal augmentation for TTA
        """
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3
            ),
            ToTensorV2()
        ])


def get_augmentation(
    task: str,
    split: str = "train",
    image_size: Tuple[int, int] = (720, 1280),
    **kwargs
) -> A.Compose:
    """Get augmentation pipeline for specific task and split.

    Args:
        task: Task name (terrain_seg/mine_det/multimodal)
        split: Dataset split (train/val/test)
        image_size: Target image size
        **kwargs: Additional arguments

    Returns:
        Albumentations composition

    Example:
        >>> aug = get_augmentation('terrain_seg', 'train')
        >>> transformed = aug(image=image, mask=mask)
    """
    if task == "terrain_seg":
        if split == "train":
            return TMASAugmentation.get_terrain_segmentation_train(image_size, **kwargs)
        else:
            return TMASAugmentation.get_terrain_segmentation_val(image_size, **kwargs)

    elif task == "mine_det":
        if split == "train":
            return TMASAugmentation.get_mine_detection_train(image_size, **kwargs)
        else:
            return TMASAugmentation.get_mine_detection_val(image_size, **kwargs)

    elif task == "multimodal":
        return TMASAugmentation.get_multimodal_train(image_size, **kwargs)

    elif task == "tta":
        return TMASAugmentation.get_test_time_augmentation()

    else:
        raise ValueError(f"Unknown task: {task}")


def main():
    """Test augmentation pipelines."""
    import numpy as np
    import matplotlib.pyplot as plt

    # Create dummy image and mask
    image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    mask = np.random.randint(0, 19, (720, 1280), dtype=np.uint8)

    # Test terrain segmentation augmentation
    print("Testing terrain segmentation augmentation...")
    aug = get_augmentation("terrain_seg", "train")

    for i in range(5):
        transformed = aug(image=image, mask=mask)
        print(f"  Sample {i+1}: image shape = {transformed['image'].shape}, "
              f"mask shape = {transformed['mask'].shape}")

    # Test mine detection augmentation
    print("\nTesting mine detection augmentation...")
    boxes = np.array([
        [100, 100, 200, 150],
        [300, 400, 380, 460],
        [500, 200, 600, 280]
    ], dtype=np.float32)
    labels = np.array([1, 3, 5], dtype=np.int64)

    aug = get_augmentation("mine_det", "train")

    for i in range(5):
        transformed = aug(image=image, bboxes=boxes, labels=labels)
        print(f"  Sample {i+1}: image shape = {transformed['image'].shape}, "
              f"num boxes = {len(transformed['bboxes'])}")

    print("\nAugmentation tests successful!")


if __name__ == "__main__":
    main()
