# TMAS Implementation Plan

**Document Version:** 1.0
**Date:** February 2026
**Status:** Ready for Implementation

---

## Overview

This document provides a comprehensive, step-by-step implementation plan for the TMAS (Traversability & Mine Analysis System). Each step is an **atomic task** that cannot be subdivided further and represents a single, cohesive unit of work.

### Implementation Strategy

- **Development Platform**: Standard GPU workstation (NVIDIA RTX 3090/4090 or similar)
- **Target Platform**: NVIDIA Jetson AGX Orin 64GB (deployment)
- **Framework**: Python-first approach, then ROS 2 integration
- **Scope**: Full end-to-end system (all 3 modules)
- **Data**: Public datasets + synthetic data generation

### Critical Success Metrics

| Metric | Target Value |
|--------|--------------|
| Mine/IED Recall (AT) | > 99.5% |
| Mine/IED Recall (AP) | > 99.0% |
| Person/Vehicle Recall | > 99.0% |
| Multi-sensor Fusion Latency | < 25 ms |
| Frame Rate | ≥ 20 FPS |
| Terrain Segmentation mIoU | > 75% |

---

## Phase 1: Project Setup & Infrastructure (Week 1-2)

### Step 1.1: Initialize Project Repository Structure
**Deliverable**: Complete project directory structure with all necessary folders

Create the following directory structure:
```
TMAS-Traversability-Mine-Analysis-System/
├── configs/
│   ├── models/
│   ├── sensors/
│   ├── training/
│   └── default.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── synthetic/
│   ├── real/
│   └── annotations/
├── models/
│   ├── checkpoints/
│   └── exports/
├── src/
│   └── tmas/
│       ├── __init__.py
│       ├── core/
│       ├── models/
│       ├── fusion/
│       ├── detection/
│       ├── segmentation/
│       ├── tracking/
│       ├── bev/
│       ├── utils/
│       └── data/
├── scripts/
│   ├── data_preparation/
│   ├── training/
│   ├── evaluation/
│   └── deployment/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── performance/
├── docs/
│   ├── api/
│   ├── tutorials/
│   └── deployment/
├── notebooks/
│   ├── 01_data_exploration/
│   ├── 02_model_development/
│   └── 03_evaluation/
└── ros2_ws/
    └── src/
```

**Verification**: All directories exist and are properly structured.

---

### Step 1.2: Create Python Package Configuration
**Deliverable**: Working Python package with proper dependencies

Files to create:
- `pyproject.toml` - Modern Python packaging configuration
- `setup.py` - Legacy compatibility
- `requirements.txt` - Core dependencies
- `requirements-dev.txt` - Development dependencies
- `src/tmas/__init__.py` - Package initialization

Core dependencies to include:
```
torch>=2.2.0
torchvision>=0.17.0
opencv-python>=4.9.0
numpy>=1.26.0
pillow>=10.2.0
albumentations>=1.3.1
timm>=0.9.16
transformers>=4.37.0
onnx>=1.15.0
onnxruntime-gpu>=1.17.0
tensorrt>=8.6.0
scipy>=1.12.0
scikit-learn>=1.4.0
matplotlib>=3.8.2
seaborn>=0.13.2
wandb>=0.16.3
tensorboard>=2.16.0
pytest>=8.0.0
black>=24.1.0
ruff>=0.2.0
mypy>=1.8.0
```

**Verification**: `pip install -e .` works without errors.

---

### Step 1.3: Setup Development Environment and Tooling
**Deliverable**: Configured development environment with linting, formatting, and type checking

Create configuration files:
- `.pre-commit-config.yaml` - Pre-commit hooks
- `pyproject.toml` - Black, Ruff, MyPy configuration
- `.vscode/settings.json` - VS Code settings (optional)
- `Makefile` - Common development commands

Setup pre-commit hooks:
```yaml
repos:
  - repo: https://github.com/psf/black
    hooks:
      - id: black
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff
  - repo: https://github.com/pre-commit/mirrors-mypy
    hooks:
      - id: mypy
```

**Verification**: Pre-commit hooks run successfully on sample files.

---

### Step 1.4: Setup Experiment Tracking Infrastructure
**Deliverable**: Working W&B project and TensorBoard logging

- Create W&B account and project: `tmas-development`
- Configure W&B API key in environment
- Create logging utilities in `src/tmas/utils/logging.py`:
  - W&B logger wrapper
  - TensorBoard logger wrapper
  - Unified logging interface
  - Checkpoint management utilities

**Verification**: Sample training run logs successfully to both W&B and TensorBoard.

---

### Step 1.5: Create Base Configuration System
**Deliverable**: YAML-based configuration system with validation

Implement in `src/tmas/core/config.py`:
- Configuration dataclasses using `dataclasses` or `pydantic`
- YAML loading/saving utilities
- Configuration validation
- Default configurations for each module

Create default configs:
- `configs/default.yaml` - System-wide defaults
- `configs/models/terrain_segmentation.yaml`
- `configs/models/mine_detection.yaml`
- `configs/models/obstacle_detection.yaml`
- `configs/training/default_training.yaml`

**Verification**: Configurations load correctly and validate schema.

---

### Step 1.6: Setup Testing Framework
**Deliverable**: Working pytest setup with example tests

Create:
- `tests/conftest.py` - Pytest fixtures
- `tests/unit/test_config.py` - Configuration tests
- `tests/unit/test_utils.py` - Utility function tests
- `pytest.ini` - Pytest configuration
- `.coveragerc` - Coverage configuration

Implement fixtures for:
- Dummy RGB images (640×480, 1280×720)
- Dummy thermal images (640×512)
- Sample detection annotations
- Sample segmentation masks

**Verification**: `pytest tests/` runs successfully with 100% pass rate.

---

## Phase 2: Data Acquisition & Preparation (Week 2-4)

### Step 2.1: Download and Setup RELLIS-3D Dataset
**Deliverable**: RELLIS-3D dataset downloaded and organized

Tasks:
- Download RELLIS-3D from official source (https://github.com/unmannedlab/RELLIS-3D)
- Extract to `data/raw/rellis3d/`
- Create dataset statistics script
- Verify 13,556 frames with RGB + annotations

Dataset structure:
```
data/raw/rellis3d/
├── sequences/
│   ├── 00000/
│   ├── 00001/
│   └── ...
└── annotations/
```

**Verification**: Dataset integrity check passes, all 13k+ frames accessible.

---

### Step 2.2: Download and Setup TartanDrive Dataset
**Deliverable**: TartanDrive dataset downloaded and organized

Tasks:
- Download TartanDrive from official source
- Extract to `data/raw/tartandrive/`
- Verify 200k+ frames with RGB + stereo
- Create data manifest JSON file

**Verification**: Dataset accessible, manifest file created with frame counts.

---

### Step 2.3: Request Access to GICHD Mine Database
**Deliverable**: Access request submitted or alternative dataset identified

Tasks:
- Research GICHD Mine Action Database access procedure
- Submit formal access request to GICHD
- Document process in `docs/data_acquisition.md`
- **Alternative**: Search for publicly available mine detection datasets:
  - Check papers with code (https://paperswithcode.com/)
  - Check Kaggle for landmine datasets
  - Check academic repositories

**Note**: This may take weeks to approve. Proceed with synthetic data in parallel.

**Verification**: Request submitted or alternative dataset identified and downloaded.

---

### Step 2.4: Search and Download Public Mine/IED Datasets
**Deliverable**: At least one public mine detection dataset downloaded

Search sources:
- Kaggle: "landmine", "mine detection", "UXO"
- Papers with Code: Mine detection papers
- IEEE DataPort
- Humanitarian demining research groups

Minimum requirement: 1000+ annotated images of mines/IED-like objects

**Verification**: Dataset downloaded with annotations in COCO or YOLO format.

---

### Step 2.5: Search and Download Thermal Imaging Datasets
**Deliverable**: Thermal imaging dataset for training thermal backbone

Search sources:
- FLIR Thermal Dataset (free dataset from FLIR)
- KAIST Multispectral Dataset
- CVC-14 Thermal Dataset
- Any RGB-Thermal paired dataset

Target: 5000+ thermal images with annotations

**Verification**: Thermal dataset downloaded and accessible.

---

### Step 2.6: Create Dataset Registry and Metadata System
**Deliverable**: Centralized dataset registry with metadata

Implement in `src/tmas/data/registry.py`:
- Dataset registry class
- Metadata tracking (size, split, statistics)
- Dataset iterator interface
- Data loading utilities

Create `data/registry.json`:
```json
{
  "datasets": {
    "rellis3d": {
      "path": "data/raw/rellis3d",
      "type": "terrain_segmentation",
      "num_samples": 13556,
      "splits": {"train": 0.8, "val": 0.1, "test": 0.1}
    },
    "tartandrive": {...},
    "mines_public": {...}
  }
}
```

**Verification**: Registry loads all datasets and reports statistics correctly.

---

### Step 2.7: Implement RELLIS-3D Data Loader
**Deliverable**: PyTorch Dataset class for RELLIS-3D

Implement in `src/tmas/data/rellis3d.py`:
- `RELLIS3DDataset(torch.utils.data.Dataset)`
- Load RGB images
- Load segmentation annotations (14 terrain classes)
- Support for train/val/test splits
- Return format: `{"image": Tensor, "mask": Tensor, "metadata": dict}`

Include:
- Image normalization (ImageNet stats)
- Basic augmentations (flip, crop, color jitter)
- Configurable input resolution

**Verification**: DataLoader works, can iterate through all samples.

---

### Step 2.8: Implement TartanDrive Data Loader
**Deliverable**: PyTorch Dataset class for TartanDrive

Implement in `src/tmas/data/tartandrive.py`:
- `TartanDriveDataset(torch.utils.data.Dataset)`
- Load RGB images
- Load stereo depth maps
- Support for different environments
- Return format: `{"image": Tensor, "depth": Tensor, "metadata": dict}`

**Verification**: DataLoader works for depth estimation tasks.

---

### Step 2.9: Implement Mine Detection Data Loader
**Deliverable**: PyTorch Dataset class for mine detection data

Implement in `src/tmas/data/mines.py`:
- `MineDetectionDataset(torch.utils.data.Dataset)`
- Support COCO and YOLO annotation formats
- Load RGB images + bounding boxes
- Class mapping: AT mine, AP mine, IED, UXO, anomaly
- Augmentations for small object detection

Return format:
```python
{
  "image": Tensor,
  "boxes": Tensor,  # [N, 4] xyxy format
  "labels": Tensor,  # [N] class indices
  "metadata": dict
}
```

**Verification**: DataLoader works with both annotation formats.

---

### Step 2.10: Implement Thermal Data Loader
**Deliverable**: PyTorch Dataset class for thermal imaging data

Implement in `src/tmas/data/thermal.py`:
- `ThermalDataset(torch.utils.data.Dataset)`
- Load thermal images (single channel or 3-channel grayscale)
- Normalize thermal values appropriately
- Support for RGB-Thermal paired data
- Temperature range normalization

**Verification**: Thermal images load correctly with proper normalization.

---

### Step 2.11: Create Data Augmentation Pipeline
**Deliverable**: Comprehensive augmentation pipeline using Albumentations

Implement in `src/tmas/data/augmentation.py`:
- Terrain segmentation augmentations
- Object detection augmentations
- Multi-modal augmentations (RGB + Thermal consistency)
- Safety-critical augmentations (high recall focus)

Augmentation strategies:
```python
- Geometric: RandomRotate90, HorizontalFlip, ShiftScaleRotate
- Color: ColorJitter, RandomBrightnessContrast, HueSaturationValue
- Noise: GaussNoise, MultiplicativeNoise
- Weather: RandomFog, RandomRain, RandomSunFlare
- Occlusion: CoarseDropout, GridDropout
```

**Verification**: Augmentations work on sample images without errors.

---

### Step 2.12: Create Train/Val/Test Splits for All Datasets
**Deliverable**: Standardized data splits with JSON manifests

Create split files in `data/splits/`:
- `rellis3d_train.json`, `rellis3d_val.json`, `rellis3d_test.json`
- `mines_train.json`, `mines_val.json`, `mines_test.json`
- `thermal_train.json`, `thermal_val.json`, `thermal_test.json`

Split ratios:
- Train: 80%
- Val: 10%
- Test: 10%

Ensure:
- No data leakage between splits
- Stratified splits for imbalanced classes
- Reproducible (fixed random seed)

**Verification**: All splits created, total samples match original dataset size.

---

## Phase 3: Synthetic Data Generation (Week 4-6)

### Step 3.1: Setup Blender for Synthetic Data Generation
**Deliverable**: Blender installed with Python scripting configured

Tasks:
- Install Blender 4.0+ with command-line support
- Install Blender Python dependencies (numpy, opencv)
- Create `scripts/data_preparation/synthetic/setup_blender.sh`
- Test Blender headless rendering

**Verification**: Blender can render a test scene from command line.

---

### Step 3.2: Acquire or Create 3D Mine Models
**Deliverable**: 3D models for common mine types

Acquire models for:
- AT mines: TM-62M, TM-46, M15
- AP mines: PMN, PMN-2, PFM-1, M14
- IED containers: Various sizes

Sources:
- Free 3D model sites (TurboSquid, CGTrader, Sketchfab)
- Create simple geometric proxies (cylinders, discs)
- GICHD reference images for accurate proportions

Store in: `data/synthetic/3d_models/mines/`

**Verification**: At least 10 mine type 3D models available in .blend or .obj format.

---

### Step 3.3: Create Procedural Terrain Generator
**Deliverable**: Blender Python script for procedural terrain generation

Implement in `scripts/data_preparation/synthetic/generate_terrain.py`:
- Procedural terrain mesh generation (noise-based)
- Multiple terrain types:
  - Paved road
  - Gravel
  - Grass (short and tall)
  - Sand
  - Rocky terrain
  - Wetland
- Texture mapping for each terrain type
- Vegetation scattering (grass, bushes)

**Verification**: Script generates diverse terrain meshes successfully.

---

### Step 3.4: Implement Mine Placement and Burial Simulation
**Deliverable**: Script to place mines in terrain with realistic burial

Implement in `scripts/data_preparation/synthetic/place_mines.py`:
- Random mine placement on terrain
- Burial depth simulation (0-15cm for AT, 0-10cm for AP)
- Partial visibility (edge showing, completely buried, surface-level)
- Realistic orientation (tilted, rotated)
- Soil displacement simulation (fresh digging)

Placement rules:
- Avoid clustering (minimum distance between mines)
- Realistic scenarios (roadside IEDs, field patterns)
- 0-5 mines per scene (sparse)

**Verification**: Mines placed realistically with varying burial depths.

---

### Step 3.5: Create Lighting and Weather Randomization
**Deliverable**: Dynamic lighting and weather conditions in Blender

Implement in `scripts/data_preparation/synthetic/randomize_environment.py`:
- Sun position randomization (time of day)
- Cloud coverage (clear, overcast, cloudy)
- Fog density variation
- Rain/wet surface effects
- Dust/haze simulation
- Shadow intensity variation

**Verification**: Same scene renders with dramatically different lighting conditions.

---

### Step 3.6: Implement Thermal Camera Simulation
**Deliverable**: Thermal rendering pipeline in Blender

Implement thermal simulation based on:
- Material thermal properties (metal vs plastic vs soil)
- Time-of-day thermal effects (day vs night)
- Buried object thermal signature
- Surface temperature gradients

Use Blender's compositor to:
- Generate temperature-based color mapping
- Simulate thermal blur (lower resolution)
- Add thermal noise

Output: 640×512 thermal images (FLIR Boson 640 resolution)

**Verification**: Thermal renders show realistic temperature differences.

---

### Step 3.7: Create Automatic Annotation Generation Pipeline
**Deliverable**: Script to automatically generate annotations from Blender scene

Implement in `scripts/data_preparation/synthetic/export_annotations.py`:
- Extract 2D bounding boxes for all mines from 3D scene
- Generate segmentation masks for terrain classes
- Export depth maps from camera
- Export metadata (mine type, burial depth, GPS coordinates)

Annotation formats:
- COCO JSON for object detection
- PNG masks for segmentation
- YAML metadata files

**Verification**: Annotations match rendered images perfectly.

---

### Step 3.8: Implement Domain Randomization Pipeline
**Deliverable**: Randomization script for textures, colors, and materials

Implement in `scripts/data_preparation/synthetic/domain_randomization.py`:
- Random texture swapping for terrain
- Color jitter for all materials
- Mine material randomization (metal, plastic, painted)
- Weathering effects (rust, dirt, erosion)
- Random camera parameters (focal length, exposure)

Goal: Maximize diversity to improve sim-to-real transfer.

**Verification**: Same scene with 100 different randomizations looks highly diverse.

---

### Step 3.9: Create Batch Rendering Pipeline
**Deliverable**: Parallel rendering system for large-scale data generation

Implement in `scripts/data_preparation/synthetic/batch_render.py`:
- Multi-process rendering (utilize all CPU cores)
- Render queue management
- Error handling and retry logic
- Progress tracking
- Render farm support (optional, for cluster)

Target: Generate 1000 scenes/hour on 16-core workstation

Command:
```bash
python scripts/data_preparation/synthetic/batch_render.py \
  --num_scenes 100000 \
  --output_dir data/synthetic/rendered \
  --workers 16
```

**Verification**: 100 scenes render successfully in parallel.

---

### Step 3.10: Generate Initial Synthetic Dataset (100k Images)
**Deliverable**: 100,000 synthetic RGB + Thermal + Annotations

Execute batch rendering:
- 100,000 unique scenes
- RGB images (1280×720)
- Thermal images (640×512)
- Segmentation masks (14 terrain classes)
- Mine bounding boxes (0-5 per image)
- Metadata (camera params, GPS, conditions)

Distribution:
- 60% with mines (various types and burial)
- 40% without mines (negative samples)

Storage: ~500GB (compressed)

**Verification**: Dataset statistics match requirements, no corrupted images.

---

### Step 3.11: Create Synthetic Dataset Loader
**Deliverable**: PyTorch Dataset class for synthetic data

Implement in `src/tmas/data/synthetic.py`:
- `SyntheticDataset(torch.utils.data.Dataset)`
- Load RGB + Thermal + Annotations
- Support for multi-task learning (segmentation + detection)
- Metadata filtering (by weather, time of day, burial depth)

**Verification**: DataLoader works for synthetic data.

---

### Step 3.12: Validate Synthetic Data Quality
**Deliverable**: Validation report on synthetic data realism

Create validation notebook: `notebooks/01_data_exploration/synthetic_validation.ipynb`

Analysis:
- Visual inspection of 100 random samples
- Annotation accuracy check
- Distribution analysis (mine types, terrain types, conditions)
- Comparison with real data (if available)
- Domain gap assessment

**Verification**: Synthetic data passes quality checks.

---

## Phase 4: Model Development - Terrain Segmentation (Week 6-8)

### Step 4.1: Implement EfficientViT-L2 Backbone
**Deliverable**: EfficientViT-L2 model loaded with pretrained weights

Implement in `src/tmas/models/backbones/efficientvit.py`:
- Load EfficientViT-L2 from `timm` or official repo
- Configure for feature extraction
- Multi-scale feature outputs (1/4, 1/8, 1/16, 1/32)
- Freeze/unfreeze utilities

```python
class EfficientViTBackbone(nn.Module):
    def __init__(self, variant='l2', pretrained=True):
        # Implementation

    def forward(self, x):
        # Returns multi-scale features
        return {
            'stride4': feat_4,
            'stride8': feat_8,
            'stride16': feat_16,
            'stride32': feat_32
        }
```

**Verification**: Forward pass works, output shapes correct.

---

### Step 4.2: Implement Semantic Segmentation Decoder
**Deliverable**: Decoder head for terrain segmentation

Implement in `src/tmas/models/segmentation/decoder.py`:
- Feature Pyramid Network (FPN) style decoder
- Upsample and fuse multi-scale features
- Final 1×1 conv to 14 classes (terrain types)
- Bilinear upsampling to input resolution

Architecture:
```
EfficientViT features → FPN decoder → 14-class logits → Upsample
```

**Verification**: Segmentation map output shape = input shape.

---

### Step 4.3: Implement Terrain Segmentation Model
**Deliverable**: Complete terrain segmentation model

Implement in `src/tmas/models/segmentation/terrain_model.py`:
- `TerrainSegmentationModel(nn.Module)`
- Combine EfficientViT backbone + decoder
- Forward pass returns logits (B, 14, H, W)
- Auxiliary outputs for deep supervision (optional)

**Verification**: Model forward pass works end-to-end.

---

### Step 4.4: Implement Segmentation Loss Functions
**Deliverable**: Loss functions for terrain segmentation

Implement in `src/tmas/models/losses/segmentation_loss.py`:
- Cross-Entropy Loss (class-weighted)
- Focal Loss (for class imbalance)
- Dice Loss (soft IoU)
- Combined loss (CE + Dice)

Class weights based on RELLIS-3D statistics:
- Road: 1.0
- Grass: 2.0
- Rubble: 5.0 (rare class)

**Verification**: Loss computation works on sample batch.

---

### Step 4.5: Create Terrain Segmentation Training Script
**Deliverable**: Training script for terrain segmentation

Implement in `scripts/training/train_terrain_segmentation.py`:
- Load RELLIS-3D + Synthetic data
- Create DataLoaders
- Initialize model, optimizer, scheduler
- Training loop with:
  - Forward pass
  - Loss computation
  - Backward pass
  - Gradient clipping
  - Logging to W&B/TensorBoard
- Validation loop
- Checkpoint saving (best model, last model)

Hyperparameters:
- Optimizer: AdamW
- Learning rate: 1e-4
- Batch size: 16
- Epochs: 50
- Scheduler: CosineAnnealingLR

**Verification**: Training script runs for 1 epoch without errors.

---

### Step 4.6: Train Terrain Segmentation Model on Synthetic Data
**Deliverable**: Trained model checkpoint achieving >70% mIoU on synthetic val

Execute:
```bash
python scripts/training/train_terrain_segmentation.py \
  --config configs/models/terrain_segmentation.yaml \
  --data synthetic \
  --epochs 50
```

Target metrics:
- mIoU > 70% on synthetic validation
- Per-class IoU > 50% for all classes

**Verification**: Model checkpoint saved, metrics logged.

---

### Step 4.7: Fine-tune Terrain Segmentation on RELLIS-3D
**Deliverable**: Fine-tuned model achieving >75% mIoU on real data

Execute:
```bash
python scripts/training/train_terrain_segmentation.py \
  --config configs/models/terrain_segmentation.yaml \
  --data rellis3d \
  --checkpoint models/checkpoints/terrain_synthetic_best.pth \
  --epochs 30
```

Target metrics:
- mIoU > 75% on RELLIS-3D validation
- Road class IoU > 90%
- Grass class IoU > 80%

**Verification**: Fine-tuned model outperforms synthetic-only baseline.

---

### Step 4.8: Implement Terrain Segmentation Inference Pipeline
**Deliverable**: Fast inference pipeline with post-processing

Implement in `src/tmas/segmentation/inference.py`:
- `TerrainSegmentationInference` class
- Model loading from checkpoint
- Preprocessing (resize, normalize)
- Inference (with TTA optional)
- Post-processing (CRF, morphological ops)
- Confidence map generation

**Verification**: Inference runs at >30 FPS on GPU.

---

### Step 4.9: Create Terrain Segmentation Evaluation Script
**Deliverable**: Comprehensive evaluation on test set

Implement in `scripts/evaluation/eval_terrain_segmentation.py`:
- Load test dataset
- Run inference on all samples
- Compute metrics:
  - mIoU (mean Intersection over Union)
  - Per-class IoU
  - Pixel accuracy
  - Confusion matrix
- Generate visualizations
- Save results to JSON

**Verification**: Evaluation runs successfully, generates report.

---

### Step 4.10: Create Terrain Segmentation Visualization Notebook
**Deliverable**: Jupyter notebook for result analysis

Create `notebooks/03_evaluation/terrain_segmentation_results.ipynb`:
- Load evaluation results
- Plot mIoU over epochs
- Visualize predictions on test samples
- Analyze failure cases
- Per-class performance breakdown
- Confusion matrix heatmap

**Verification**: Notebook runs and generates all visualizations.

---

## Phase 5: Model Development - Mine Detection (Week 8-12)

### Step 5.1: Implement ResNet-18 Thermal Backbone
**Deliverable**: ResNet-18 for thermal branch

Implement in `src/tmas/models/backbones/resnet_thermal.py`:
- ResNet-18 with single-channel input (thermal)
- Pretrained on ImageNet, first conv adapted
- Multi-scale feature extraction
- Lightweight for real-time performance

**Verification**: Forward pass works on thermal images.

---

### Step 5.2: Implement Cross-Attention Fusion Module
**Deliverable**: Cross-attention transformer for RGB-Thermal fusion

Implement in `src/tmas/fusion/cross_attention.py`:
- Multi-head cross-attention
- RGB features attend to thermal features (and vice versa)
- Learnable fusion weights
- Efficient implementation (Flash Attention if available)

Architecture:
```
RGB_feat, Thermal_feat → Cross-Attention → Fused_feat
```

**Verification**: Fusion module works, output shape correct.

---

### Step 5.3: Implement RT-DETR Detection Head
**Deliverable**: RT-DETR real-time detection head

Implement in `src/tmas/models/detection/rtdetr.py`:
- Load RT-DETR-L architecture
- Adapt for 8 classes (mine types)
- Configurable number of queries
- NMS-free detection (DETR benefit)

Classes:
1. AT mine (confirmed)
2. AP mine (confirmed)
3. IED roadside
4. IED buried
5. UXO
6. Wire/trigger
7. Soil anomaly
8. False positive (background)

**Verification**: Detection head outputs bounding boxes and classes.

---

### Step 5.4: Implement Complete Mine Detection Model
**Deliverable**: End-to-end mine detection model with fusion

Implement in `src/tmas/models/detection/mine_detector.py`:
- `MineDetectionModel(nn.Module)`
- RGB backbone (EfficientViT-L2, shared with segmentation)
- Thermal backbone (ResNet-18)
- Cross-attention fusion
- RT-DETR head
- Multi-scale detection

**Verification**: Full model forward pass works.

---

### Step 5.5: Implement Evidential Deep Learning for Uncertainty
**Deliverable**: EDL-based uncertainty estimation

Implement in `src/tmas/models/uncertainty/evidential.py`:
- Evidential loss function
- Dirichlet distribution modeling
- Epistemic + Aleatoric uncertainty
- Confidence calibration

Replace standard classification head with EDL output:
```python
# Standard: logits → softmax → probability
# EDL: logits → evidence → Dirichlet → probability + uncertainty
```

**Verification**: Uncertainty scores computed for each detection.

---

### Step 5.6: Implement Detection Loss Functions
**Deliverable**: Loss functions for mine detection

Implement in `src/tmas/models/losses/detection_loss.py`:
- Focal Loss (for extreme class imbalance)
- GIoU Loss (for bounding boxes)
- L1 Loss (box regression)
- Evidential Loss (for uncertainty)
- Combined loss with weights

Class weighting:
- Confirmed mines: weight = 10.0 (high recall priority)
- IED: weight = 8.0
- Anomaly: weight = 5.0
- False positive: weight = 1.0

**Verification**: Loss computation works on sample batch.

---

### Step 5.7: Create Mine Detection Training Script
**Deliverable**: Training script for mine detection

Implement in `scripts/training/train_mine_detection.py`:
- Load synthetic + real mine datasets
- DataLoader with detection augmentations
- Multi-modal data loading (RGB + Thermal)
- Training loop with:
  - RGB-Thermal fusion
  - Detection loss
  - Uncertainty calibration
  - Recall monitoring (>99.5% target)
- Checkpoint saving (best recall model)

Hyperparameters:
- Optimizer: AdamW
- Learning rate: 5e-5
- Batch size: 8 (large images)
- Epochs: 100
- Recall threshold: 0.3 (very low for high sensitivity)

**Verification**: Training script runs for 1 epoch.

---

### Step 5.8: Train Mine Detection Model on Synthetic Data
**Deliverable**: Trained mine detector achieving >95% recall on synthetic

Execute:
```bash
python scripts/training/train_mine_detection.py \
  --config configs/models/mine_detection.yaml \
  --data synthetic \
  --epochs 100
```

Target metrics:
- Recall (AT mines) > 95%
- Recall (AP mines) > 92%
- Recall (IED) > 90%
- Precision > 40% (acceptable with high recall)

**Verification**: Model achieves target recall on synthetic validation.

---

### Step 5.9: Fine-tune Mine Detection on Real Data (if available)
**Deliverable**: Fine-tuned model on real mine images

If real mine data available:
```bash
python scripts/training/train_mine_detection.py \
  --config configs/models/mine_detection.yaml \
  --data mines_real \
  --checkpoint models/checkpoints/mine_synthetic_best.pth \
  --epochs 50
```

If not available: Skip and rely on synthetic data.

**Verification**: Model performance improves on real validation data (if available).

---

### Step 5.10: Implement ByteTrack for Temporal Consistency
**Deliverable**: Object tracking for mines across frames

Implement in `src/tmas/tracking/bytetrack.py`:
- ByteTrack algorithm implementation
- Track initialization and association
- Kalman filter for state prediction
- Confidence accumulation over time
- Track ID management

Benefits:
- Reduce false positives (require detection in N frames)
- Increase confidence through temporal voting
- Smooth detection jitter

**Verification**: Tracking works on video sequence, maintains IDs.

---

### Step 5.11: Implement Mine Detection Inference Pipeline
**Deliverable**: Real-time inference with tracking and uncertainty

Implement in `src/tmas/detection/inference.py`:
- `MineDetectionInference` class
- RGB + Thermal input processing
- Model inference
- Uncertainty thresholding
- Temporal tracking (ByteTrack)
- GPS coordinate mapping (if available)

Output format:
```python
{
  'detections': [
    {
      'bbox': [x1, y1, x2, y2],
      'class': 'AT_mine',
      'confidence': 0.95,
      'uncertainty': 0.05,
      'track_id': 42,
      'gps': (lat, lon)
    }
  ]
}
```

**Verification**: Inference runs at >20 FPS with tracking.

---

### Step 5.12: Create Mine Detection Evaluation Script
**Deliverable**: Comprehensive evaluation with recall focus

Implement in `scripts/evaluation/eval_mine_detection.py`:
- Load test dataset
- Run inference with various confidence thresholds
- Compute metrics:
  - Recall @ IoU 0.5 (primary metric)
  - Precision @ IoU 0.5
  - F1 score
  - Average Precision (AP)
  - Per-class recall
- Generate precision-recall curves
- Analyze missed detections (false negatives)

**Verification**: Evaluation shows >99% recall achievable at some threshold.

---

### Step 5.13: Optimize Detection Threshold for Maximum Recall
**Deliverable**: Optimal threshold configuration for >99.5% recall

Create optimization script: `scripts/evaluation/optimize_threshold.py`

Tasks:
- Grid search over confidence thresholds [0.1, 0.15, 0.2, ..., 0.5]
- Find threshold that maximizes recall while keeping FPR < 0.15
- Validate on test set
- Save optimal threshold to config

**Verification**: Threshold found that achieves >99.5% recall.

---

### Step 5.14: Create Mine Detection Visualization Notebook
**Deliverable**: Jupyter notebook for mine detection analysis

Create `notebooks/03_evaluation/mine_detection_results.ipynb`:
- Visualize detections with bounding boxes
- Show RGB + Thermal + Fused predictions
- Uncertainty heatmaps
- False positive analysis
- False negative analysis (critical!)
- Precision-recall curves
- Confusion matrix

**Verification**: Notebook generates comprehensive visualizations.

---

## Phase 6: Model Development - Obstacle Detection (Week 12-14)

### Step 6.1: Prepare Obstacle Detection Dataset
**Deliverable**: Dataset with 20+ obstacle classes

Collect/download datasets:
- COCO dataset (persons, vehicles, animals)
- Open Images (debris, barriers)
- Synthetic obstacles in Blender scenes

Classes:
1. Person
2. Vehicle (car, truck, military)
3. Animal (large)
4. Tree fallen
5. Rock/boulder
6. Crater/hole
7. Wreckage
8. Debris
9. Barrier
10+ (additional classes)

**Verification**: Dataset with 10k+ annotated obstacle images.

---

### Step 6.2: Implement Obstacle Detection Model (YOLOv8/RT-DETR)
**Deliverable**: Real-time obstacle detection model

Implement in `src/tmas/models/detection/obstacle_detector.py`:
- Choose YOLOv8-Large or RT-DETR-L
- Adapt for 20+ obstacle classes
- Pretrained on COCO, fine-tune on obstacle dataset
- Multi-scale detection (small to large obstacles)

**Verification**: Model detects obstacles in test images.

---

### Step 6.3: Implement Monocular Depth Estimation
**Deliverable**: Depth estimation for obstacle distance

Options:
- DPT (Dense Prediction Transformer)
- MiDaS v3
- ZoeDepth

Implement in `src/tmas/models/depth/monocular_depth.py`:
- Load pretrained depth model
- Inference on RGB images
- Depth map to metric distance (calibration required)
- Integration with obstacle detection

**Verification**: Depth maps generated for test images.

---

### Step 6.4: Implement Trajectory Prediction Module
**Deliverable**: Trajectory prediction for moving obstacles

Implement in `src/tmas/tracking/trajectory_prediction.py`:
- Kalman filter for linear motion
- Constant velocity model
- Path extrapolation (1-3 seconds ahead)
- Collision detection with vehicle path

**Verification**: Trajectories predicted for moving objects in video.

---

### Step 6.5: Implement Time-to-Collision (TTC) Estimation
**Deliverable**: TTC calculation for collision warning

Implement in `src/tmas/detection/ttc.py`:
- Compute TTC from depth and velocity
- Account for vehicle ego-motion
- Safety zones:
  - Critical: 0-10m (TTC < 1s)
  - Warning: 10-20m (TTC < 3s)
  - Observation: 20-50m

**Verification**: TTC computed accurately (± 0.3s target).

---

### Step 6.6: Implement Sudden Obstacle Detection
**Deliverable**: Frame differencing for sudden appearance

Implement in `src/tmas/detection/sudden_obstacle.py`:
- Frame differencing (current - previous)
- Motion saliency map
- Edge-triggered alerts for new objects
- Latency < 50ms (1-2 frames @ 20 FPS)

**Verification**: Sudden obstacles trigger immediate alerts.

---

### Step 6.7: Create Obstacle Detection Training Script
**Deliverable**: Training script for obstacle detection

Implement in `scripts/training/train_obstacle_detection.py`:
- Load obstacle dataset
- Train YOLOv8/RT-DETR
- Target: Recall > 99% for persons/vehicles

**Verification**: Training script runs successfully.

---

### Step 6.8: Train Obstacle Detection Model
**Deliverable**: Trained model achieving >99% recall on critical classes

Execute:
```bash
python scripts/training/train_obstacle_detection.py \
  --config configs/models/obstacle_detection.yaml \
  --epochs 50
```

Target metrics:
- Person recall > 99.5%
- Vehicle recall > 99%
- Static obstacle recall > 95%

**Verification**: Model achieves target recall.

---

### Step 6.9: Implement Obstacle Detection Inference Pipeline
**Deliverable**: Real-time obstacle detection with TTC

Implement in `src/tmas/detection/obstacle_inference.py`:
- Detection + depth + tracking + TTC
- Alert generation (critical, warning, observation)
- Emergency brake recommendation logic

**Verification**: Inference runs at >20 FPS.

---

### Step 6.10: Create Obstacle Detection Evaluation Script
**Deliverable**: Evaluation focusing on recall and TTC accuracy

Implement in `scripts/evaluation/eval_obstacle_detection.py`:
- Recall/precision metrics
- TTC accuracy (± 0.3s)
- Latency measurement
- Sudden obstacle detection latency

**Verification**: Evaluation confirms >99% recall and TTC accuracy.

---

## Phase 7: BEV Transformation & Fusion (Week 14-16)

### Step 7.1: Implement BEV Transformation Module
**Deliverable**: Transform perspective view to Bird's Eye View

Implement in `src/tmas/bev/transform.py`:
- Inverse perspective mapping (IPM)
- Camera calibration (intrinsics/extrinsics)
- Ground plane assumption
- Transform RGB features to BEV grid (400×400, 5cm/pixel, 20m×20m)

**Verification**: BEV grid generated from camera view.

---

### Step 7.2: Implement BEV Terrain Cost Map
**Deliverable**: Terrain cost map in BEV

Implement in `src/tmas/bev/cost_map.py`:
- Project terrain segmentation to BEV
- Assign terrain costs (from spec):
  - Paved road: 0.0
  - Gravel: 0.1
  - Sand: 0.4
  - Rubble: 0.8
- Add geometry cost (slope, roughness)
- Output: 400×400 float32 grid

**Verification**: Cost map generated from segmentation.

---

### Step 7.3: Implement BEV Threat Map
**Deliverable**: Mine and obstacle threat map in BEV

Implement in `src/tmas/bev/threat_map.py`:
- Project mine detections to BEV
- Project obstacle detections to BEV
- Threat cost assignment:
  - Confirmed mine: ∞ (impassable)
  - Suspected anomaly: 0.95
  - Person/vehicle: ∞
  - Obstacle: 0.7-0.9
- Output: 400×400 uint8 grid

**Verification**: Threat map highlights detected hazards.

---

### Step 7.4: Implement Final BEV Fusion
**Deliverable**: Unified traversability map

Implement in `src/tmas/bev/fusion.py`:
- Fuse terrain cost + threat cost
- Formula: `Cost_final = max(Cost_terrain + Cost_geometry, Cost_threat)`
- If threat cost = ∞, cell is blocked
- Color-coded visualization (green=safe, red=blocked)

**Verification**: Final BEV map combines all information.

---

### Step 7.5: Create BEV Visualization Module
**Deliverable**: Real-time BEV visualization

Implement in `src/tmas/visualization/bev_viz.py`:
- Render BEV cost map
- Overlay detections (mines, obstacles)
- Vehicle position marker
- Safe/unsafe zones color-coded
- Matplotlib/OpenCV rendering for HMI

**Verification**: BEV visualized in real-time.

---

## Phase 8: System Integration (Week 16-18)

### Step 8.1: Implement Multi-Sensor Data Synchronization
**Deliverable**: Time-synchronized RGB + Thermal input

Implement in `src/tmas/core/synchronization.py`:
- Timestamp alignment
- Buffer management
- Interpolation for missing frames
- Software-based sync (hardware trigger simulation)

**Verification**: RGB and thermal frames synchronized correctly.

---

### Step 8.2: Create Unified TMAS Inference Pipeline
**Deliverable**: End-to-end system integration

Implement in `src/tmas/core/tmas_system.py`:
- `TMASSystem` class
- Load all models (terrain, mine, obstacle)
- Process RGB + Thermal inputs
- Run all modules in parallel:
  - Terrain segmentation
  - Mine detection
  - Obstacle detection
- Generate BEV maps
- Output alerts

**Verification**: Full pipeline runs end-to-end.

---

### Step 8.3: Implement Alert System
**Deliverable**: Real-time alert generation

Implement in `src/tmas/core/alerts.py`:
- Alert types: CRITICAL, WARNING, INFO
- Alert triggers:
  - Mine detected → CRITICAL
  - Person detected → CRITICAL
  - TTC < 2s → WARNING
  - Anomaly detected → INFO
- Audio alert generation (beep patterns)
- Visual alert overlays

**Verification**: Alerts trigger correctly for test scenarios.

---

### Step 8.4: Create Output Data Structures
**Deliverable**: Standardized output format

Implement in `src/tmas/core/output.py`:
- Dataclasses for:
  - Mine detection output
  - Obstacle detection output
  - BEV cost map
  - Alert stream
- JSON serialization
- MCAP logging format (ROS 2 compatible)

**Verification**: Outputs serialize correctly.

---

### Step 8.5: Implement MCAP Logging
**Deliverable**: Full session recording for post-analysis

Implement in `src/tmas/utils/mcap_logger.py`:
- Record all inputs (RGB, thermal, GPS)
- Record all outputs (detections, BEV, alerts)
- Timestamped messages
- Playback utilities

**Verification**: Session recorded and playable.

---

### Step 8.6: Create Operator HMI (Head-Mounted Display Interface)
**Deliverable**: Qt6-based operator interface

Implement in `src/tmas/hmi/main_window.py`:
- Qt6 application
- Layout:
  - RGB camera view (left)
  - BEV map (right)
  - Alert panel (bottom)
  - Status bar (FPS, latency, objects detected)
- Real-time updates (>20 FPS)

**Verification**: HMI displays all information in real-time.

---

### Step 8.7: Implement Performance Monitoring
**Deliverable**: Latency and FPS tracking

Implement in `src/tmas/utils/performance.py`:
- Latency measurement for each module
- End-to-end latency tracking
- FPS counter
- GPU memory monitoring
- Performance logging

Target:
- Total latency < 50ms (20 FPS)
- Each module < 25ms

**Verification**: Performance metrics logged correctly.

---

### Step 8.8: Create End-to-End Integration Tests
**Deliverable**: Integration test suite

Implement in `tests/integration/test_full_system.py`:
- Load sample RGB + Thermal video
- Run full TMAS pipeline
- Verify all outputs generated
- Check latency < 50ms
- Verify alerts trigger correctly

**Verification**: All integration tests pass.

---

### Step 8.9: Create Demo Video Processing Script
**Deliverable**: Script to process video and generate output

Implement in `scripts/demo/process_video.py`:
- Load video file (RGB + Thermal)
- Run TMAS pipeline on each frame
- Generate output video with:
  - Detections overlaid
  - BEV map side-by-side
  - Alerts displayed
- Save to MP4

**Verification**: Demo video generated successfully.

---

### Step 8.10: Optimize Pipeline for Real-Time Performance
**Deliverable**: Optimized pipeline achieving >20 FPS

Optimizations:
- Model quantization (FP16)
- TensorRT compilation (defer to Phase 9)
- Multi-threading (parallel module execution)
- GPU stream optimization
- Reduce unnecessary data copies

**Verification**: Pipeline runs at >20 FPS on GPU workstation.

---

## Phase 9: Model Optimization & Export (Week 18-20)

### Step 9.1: Export Models to ONNX Format
**Deliverable**: ONNX models for all modules

Implement in `scripts/deployment/export_to_onnx.py`:
- Export terrain segmentation model
- Export mine detection model
- Export obstacle detection model
- Verify ONNX models with sample inputs

**Verification**: ONNX models run successfully with ONNX Runtime.

---

### Step 9.2: Optimize ONNX Models
**Deliverable**: Optimized ONNX graphs

Use ONNX optimizer:
- Constant folding
- Dead code elimination
- Operator fusion
- Graph simplification

**Verification**: Optimized ONNX models are smaller and faster.

---

### Step 9.3: Convert ONNX to TensorRT (FP16)
**Deliverable**: TensorRT engines for faster inference

Implement in `scripts/deployment/convert_to_tensorrt.py`:
- Convert ONNX to TensorRT
- FP16 precision (2x speedup)
- Optimize for batch size = 1 (real-time)
- Profile on target GPU

**Verification**: TensorRT models run at >40 FPS.

---

### Step 9.4: Implement INT8 Calibration for TensorRT
**Deliverable**: INT8 quantized models (if latency still high)

Implement calibration:
- Create calibration dataset (1000 samples)
- Run TensorRT INT8 calibration
- Validate accuracy (ensure recall still >99%)

**Verification**: INT8 models run at >60 FPS with minimal accuracy loss.

---

### Step 9.5: Benchmark Models on Target Hardware (if available)
**Deliverable**: Performance report on Jetson AGX Orin

If Jetson available:
- Deploy TensorRT models to Jetson
- Measure latency and FPS
- Measure power consumption
- Thermal testing

If not available: Benchmark on desktop GPU and estimate Jetson performance.

**Verification**: Models meet <25ms latency target.

---

### Step 9.6: Create Model Deployment Package
**Deliverable**: Deployable model artifacts

Package structure:
```
models/deploy/
├── terrain_segmentation.trt
├── mine_detection.trt
├── obstacle_detection.trt
├── depth_estimation.trt
├── configs/
│   ├── terrain_config.yaml
│   ├── mine_config.yaml
│   └── obstacle_config.yaml
└── README.md
```

**Verification**: Models loadable with deployment script.

---

## Phase 10: ROS 2 Integration (Week 20-24)

### Step 10.1: Create ROS 2 Workspace Structure
**Deliverable**: ROS 2 workspace with TMAS package

Create:
```
ros2_ws/
└── src/
    └── tmas_ros2/
        ├── package.xml
        ├── setup.py
        ├── tmas_ros2/
        │   ├── __init__.py
        │   ├── nodes/
        │   ├── launch/
        │   └── config/
        └── resource/
```

**Verification**: ROS 2 package builds with `colcon build`.

---

### Step 10.2: Create Camera Driver Nodes (Simulated)
**Deliverable**: ROS 2 nodes for RGB and thermal cameras

Implement:
- `ros2_ws/src/tmas_ros2/tmas_ros2/nodes/rgb_camera_node.py`
- `ros2_ws/src/tmas_ros2/tmas_ros2/nodes/thermal_camera_node.py`

For now, simulate cameras by:
- Reading from video files
- Publishing to `/tmas/camera/rgb/image_raw`
- Publishing to `/tmas/camera/thermal/image_raw`

**Verification**: Camera topics publish at 20 Hz.

---

### Step 10.3: Create Terrain Segmentation ROS 2 Node
**Deliverable**: ROS 2 node for terrain segmentation

Implement in `nodes/terrain_segmentation_node.py`:
- Subscribe to `/tmas/camera/rgb/image_raw`
- Run terrain segmentation model
- Publish to `/tmas/segmentation/terrain_map`
- Publish to `/tmas/bev/terrain_cost_map`

**Verification**: Node publishes segmentation results.

---

### Step 10.4: Create Mine Detection ROS 2 Node
**Deliverable**: ROS 2 node for mine detection

Implement in `nodes/mine_detection_node.py`:
- Subscribe to RGB and thermal image topics
- Run mine detection model
- Publish to `/tmas/detections/mines`
- Publish to `/tmas/alerts` (if mine detected)

**Verification**: Node publishes mine detections.

---

### Step 10.5: Create Obstacle Detection ROS 2 Node
**Deliverable**: ROS 2 node for obstacle detection

Implement in `nodes/obstacle_detection_node.py`:
- Subscribe to `/tmas/camera/rgb/image_raw`
- Run obstacle detection + depth + TTC
- Publish to `/tmas/detections/obstacles`
- Publish to `/tmas/alerts` (if critical obstacle)

**Verification**: Node publishes obstacle detections with TTC.

---

### Step 10.6: Create BEV Fusion ROS 2 Node
**Deliverable**: ROS 2 node for BEV map generation

Implement in `nodes/bev_fusion_node.py`:
- Subscribe to terrain cost map
- Subscribe to mine detections
- Subscribe to obstacle detections
- Generate unified BEV cost map
- Publish to `/tmas/bev/cost_map`
- Publish to `/tmas/bev/threat_map`

**Verification**: BEV maps published in real-time.

---

### Step 10.7: Create Visualization ROS 2 Node
**Deliverable**: ROS 2 node for visualization

Implement in `nodes/visualization_node.py`:
- Subscribe to all detection topics
- Subscribe to BEV maps
- Generate visualization overlays
- Publish to `/tmas/visualization/image`

**Verification**: Visualization publishes annotated images.

---

### Step 10.8: Create Alert Manager ROS 2 Node
**Deliverable**: Central alert aggregation node

Implement in `nodes/alert_manager_node.py`:
- Subscribe to all alert sources
- Aggregate and prioritize alerts
- Publish unified alert stream to `/tmas/alerts`
- Optional: Audio alert (beep/buzzer)

**Verification**: Alerts aggregated correctly with priorities.

---

### Step 10.9: Create Launch Files for TMAS System
**Deliverable**: ROS 2 launch files for full system

Create `launch/tmas_full.launch.py`:
- Launch all nodes (cameras, detection, BEV, viz)
- Configure parameters from YAML
- Set up topic remappings

Create `launch/tmas_simulation.launch.py`:
- Launch with simulated cameras (video files)

**Verification**: Full system launches with single command.

---

### Step 10.10: Create RViz2 Configuration for Visualization
**Deliverable**: RViz2 config for TMAS visualization

Create `config/tmas_rviz.rviz`:
- Camera view display
- BEV map display
- Detection markers (mines, obstacles)
- Alert panel

**Verification**: RViz2 shows all TMAS outputs.

---

### Step 10.11: Implement ROS 2 Bag Recording
**Deliverable**: Record TMAS sessions to ROS 2 bag

Create script: `scripts/ros2/record_session.sh`
- Record all topics to bag file
- MCAP format (ROS 2 default)

**Verification**: Session recorded and playable.

---

### Step 10.12: Create ROS 2 Integration Tests
**Deliverable**: Integration tests for ROS 2 nodes

Implement in `tests/integration/test_ros2_integration.py`:
- Launch all nodes
- Publish test data to camera topics
- Verify all output topics publish
- Verify latency < 50ms

**Verification**: All ROS 2 integration tests pass.

---

## Phase 11: Testing & Validation (Week 24-28)

### Step 11.1: Create Unit Test Suite for All Modules
**Deliverable**: Comprehensive unit tests (>80% coverage)

Implement tests for:
- Data loaders
- Model architectures
- Loss functions
- BEV transformations
- Alert logic
- Synchronization

Target: >80% code coverage

**Verification**: `pytest tests/unit/` shows >80% coverage.

---

### Step 11.2: Create Performance Benchmark Suite
**Deliverable**: Automated performance benchmarking

Implement in `tests/performance/benchmark_latency.py`:
- Measure latency for each module
- Measure end-to-end latency
- Measure FPS
- Measure GPU memory usage
- Generate performance report

**Verification**: Benchmark report shows <25ms latency.

---

### Step 11.3: Create Recall Validation Test Suite
**Deliverable**: Tests to ensure recall targets are met

Implement in `tests/validation/test_recall.py`:
- Test mine detection recall >99.5%
- Test obstacle detection recall >99%
- Test on held-out test sets
- Generate recall report

**Verification**: All recall targets met on test set.

---

### Step 11.4: Create Failure Mode Testing Suite
**Deliverable**: Tests for sensor failures and degradation

Implement in `tests/integration/test_failure_modes.py`:
- Simulate RGB camera failure
- Simulate thermal camera failure
- Simulate GPS loss
- Verify graceful degradation
- Verify alerts triggered

**Verification**: System degrades gracefully, no crashes.

---

### Step 11.5: Create Stress Testing Suite
**Deliverable**: Test system under extreme conditions

Implement in `tests/stress/test_extreme_conditions.py`:
- High mine density (10+ per frame)
- High obstacle count (20+ per frame)
- Poor lighting conditions
- Thermal noise
- Measure performance degradation

**Verification**: System remains stable under stress.

---

### Step 11.6: Validate on Real-World Video (if available)
**Deliverable**: Validation on real terrain/obstacle videos

If real videos available:
- Run TMAS on real video sequences
- Manually verify detections
- Measure false positive/negative rates

If not available: Use realistic synthetic videos.

**Verification**: Qualitative validation shows good performance.

---

### Step 11.7: Create Safety Validation Report
**Deliverable**: Document showing all safety requirements met

Create `docs/safety_validation_report.md`:
- List all safety requirements (SR-1 to SR-12)
- Provide test results for each
- Show recall >99.5% for mines
- Show latency <100ms for alerts
- Show graceful degradation for failures

**Verification**: All safety requirements documented and met.

---

### Step 11.8: Create Performance Validation Report
**Deliverable**: Document showing all performance targets met

Create `docs/performance_validation_report.md`:
- List all performance metrics
- Provide benchmark results
- Show latency <25ms for fusion
- Show FPS >20
- Show mIoU >75% for terrain

**Verification**: All performance targets documented and met.

---

### Step 11.9: Create User Acceptance Test Plan
**Deliverable**: Test plan for end-user validation

Create `docs/user_acceptance_test_plan.md`:
- Define test scenarios (convoy, patrol, demining)
- Define acceptance criteria
- Define test procedures
- Create test data/videos

**Note**: Actual execution requires field testing (pending external coordination).

**Verification**: Test plan documented and reviewed.

---

## Phase 12: Documentation & Deployment Preparation (Week 28-32)

### Step 12.1: Write API Documentation
**Deliverable**: Complete API reference documentation

Generate API docs using Sphinx:
- Document all public classes and functions
- Include usage examples
- Generate HTML documentation
- Host on Read the Docs or GitHub Pages

**Verification**: API docs accessible and complete.

---

### Step 12.2: Create Training Tutorial
**Deliverable**: Step-by-step training guide

Create `docs/tutorials/01_training_models.md`:
- How to prepare data
- How to train each module
- How to monitor training
- How to evaluate models
- Troubleshooting guide

**Verification**: Tutorial can be followed by new users.

---

### Step 12.3: Create Deployment Tutorial for Jetson
**Deliverable**: Jetson deployment guide

Create `docs/tutorials/02_jetson_deployment.md`:
- JetPack installation
- TensorRT model deployment
- ROS 2 setup on Jetson
- Performance tuning
- Troubleshooting

**Verification**: Guide is clear and actionable.

---

### Step 12.4: Create Operator Manual
**Deliverable**: User manual for TMAS operators

Create `docs/operator_manual.md`:
- System overview
- How to start/stop system
- How to interpret BEV map
- How to respond to alerts
- Emergency procedures
- Maintenance procedures

**Verification**: Manual is clear for non-technical operators.

---

### Step 12.5: Create System Architecture Documentation
**Deliverable**: Technical architecture document

Create `docs/architecture.md`:
- System block diagram
- Data flow diagram
- Module interfaces
- Performance characteristics
- Design decisions and rationale

**Verification**: Architecture clearly documented.

---

### Step 12.6: Create Dataset Preparation Guide
**Deliverable**: Guide for creating custom datasets

Create `docs/tutorials/03_dataset_preparation.md`:
- How to collect data
- How to annotate data
- Annotation tools and formats
- Quality control procedures

**Verification**: Guide enables users to create their own datasets.

---

### Step 12.7: Create Model Fine-Tuning Guide
**Deliverable**: Guide for fine-tuning on custom data

Create `docs/tutorials/04_fine_tuning.md`:
- When to fine-tune vs train from scratch
- How to prepare custom dataset
- How to fine-tune each module
- How to evaluate fine-tuned models

**Verification**: Guide is actionable for advanced users.

---

### Step 12.8: Create Troubleshooting Guide
**Deliverable**: Common issues and solutions

Create `docs/troubleshooting.md`:
- Installation issues
- Performance issues
- Accuracy issues
- Hardware issues
- ROS 2 issues
- FAQ

**Verification**: Common issues covered with solutions.

---

### Step 12.9: Create Release Checklist
**Deliverable**: Checklist for software releases

Create `docs/release_checklist.md`:
- Code quality checks (linting, tests)
- Performance benchmarks
- Documentation updates
- Version tagging
- Changelog updates
- Binary releases (Docker, models)

**Verification**: Checklist is comprehensive.

---

### Step 12.10: Create Docker Deployment Package
**Deliverable**: Docker containers for easy deployment

Create:
- `Dockerfile` for development
- `Dockerfile.jetson` for Jetson deployment
- `docker-compose.yml` for full system
- Include all dependencies and models

**Verification**: Docker containers build and run successfully.

---

### Step 12.11: Create Model Zoo and Download Scripts
**Deliverable**: Pre-trained model repository

Create `scripts/download_models.sh`:
- Download pretrained models from cloud storage
- Verify checksums
- Place in correct directories

Host models on:
- Hugging Face Model Hub
- Google Drive
- GitHub Releases

**Verification**: Models download successfully.

---

### Step 12.12: Create Continuous Integration Pipeline
**Deliverable**: CI/CD for automated testing

Create `.github/workflows/ci.yml`:
- Run tests on every commit
- Check code formatting
- Run linting
- Measure code coverage
- Build Docker images

**Verification**: CI pipeline runs successfully.

---

## Phase 13: Field Testing Preparation (Week 32-36) [PENDING EXTERNAL COORDINATION]

**Note**: The following steps require access to military/engineering test facilities and personnel. These are placeholders for future coordination.

### Step 13.1: Prepare Field Testing Equipment List
**Deliverable**: BOM for field testing

Document required equipment:
- NVIDIA Jetson AGX Orin + accessories
- FLIR Blackfly S camera
- FLIR Boson 640 thermal camera
- GPS/INS unit (Emlid Reach RS2+)
- Vehicle mounting hardware
- Power supply (12V DC)
- Cables and enclosures

**Verification**: Equipment list complete and reviewed.

---

### Step 13.2: Prepare Test Range Setup Procedures
**Deliverable**: Procedures for setting up test range

Create `docs/field_testing/test_range_setup.md`:
- Mine replica placement procedures
- Safety protocols
- Test scenario definitions
- Data collection procedures

**Verification**: Procedures documented.

---

### Step 13.3: Create Field Data Collection Protocol
**Deliverable**: Protocol for collecting field data

Create `docs/field_testing/data_collection_protocol.md`:
- Video recording procedures
- GPS tagging procedures
- Ground truth annotation procedures
- Quality control checks

**Verification**: Protocol documented.

---

### Step 13.4: Prepare Safety Protocols for Field Testing
**Deliverable**: Safety procedures for test personnel

Create `docs/field_testing/safety_protocols.md`:
- Personnel protective equipment
- Test range safety zones
- Emergency procedures
- Communication protocols

**Verification**: Safety protocols reviewed by experts.

---

### Step 13.5: Create Field Testing Data Analysis Pipeline
**Deliverable**: Scripts for analyzing field test data

Implement in `scripts/field_testing/analyze_field_data.py`:
- Load field data (video + GPS + ground truth)
- Run TMAS inference
- Compare with ground truth
- Generate performance report
- Identify failure cases

**Verification**: Analysis pipeline works on sample field data.

---

## Phase 14: Advanced Features (Week 36-40) [OPTIONAL ENHANCEMENTS]

### Step 14.1: Implement Multi-Frame Temporal Fusion
**Deliverable**: Temporal aggregation for improved recall

Implement temporal voting:
- Aggregate detections over last N frames (N=5)
- Increase confidence for persistent detections
- Reduce false positives from transient noise

**Verification**: Recall improves by 1-2% with temporal fusion.

---

### Step 14.2: Implement Ensemble Model Strategy
**Deliverable**: Ensemble of multiple detection models

Train multiple mine detection models:
- Different architectures (RT-DETR, YOLOv8, Faster R-CNN)
- Different backbones
- Ensemble voting (majority vote or weighted average)

**Verification**: Ensemble achieves higher recall than single model.

---

### Step 14.3: Implement Active Learning Pipeline
**Deliverable**: Active learning for continuous improvement

Implement in `scripts/active_learning/select_samples.py`:
- Identify low-confidence detections
- Select hard examples for annotation
- Retrain model with new annotations
- Iterative improvement loop

**Verification**: Active learning improves model on hard cases.

---

### Step 14.4: Implement GPR Signal Processing (if GPR available)
**Deliverable**: GPR integration for buried mine detection

Implement in `src/tmas/sensors/gpr.py`:
- GPR signal preprocessing
- Anomaly detection in GPR signals
- Fusion with RGB/Thermal detections
- 3D subsurface mapping

**Verification**: GPR detects buried mines (if hardware available).

---

### Step 14.5: Implement Multi-Mission Configuration System
**Deliverable**: Switchable profiles for different missions

Create mission profiles:
- `convoy_mode.yaml`: Speed priority, 30m range
- `demining_mode.yaml`: Maximum recall, slower acceptable
- `patrol_mode.yaml`: Balanced performance

**Verification**: System behavior changes per mission profile.

---

### Step 14.6: Implement Cloud-Based Model Updates
**Deliverable**: Over-the-air model updates

Implement in `src/tmas/utils/model_updater.py`:
- Check for model updates on cloud server
- Download new models
- Validate checksums
- Hot-swap models without restart

**Verification**: Models update successfully from cloud.

---

### Step 14.7: Implement Federated Learning Framework
**Deliverable**: Privacy-preserving collaborative learning

Implement federated learning:
- Local model training on vehicle
- Gradient aggregation (no raw data sharing)
- Distributed model improvement
- Privacy-preserving protocols

**Verification**: Federated training improves model without sharing data.

---

### Step 14.8: Implement Explainability Visualizations
**Deliverable**: Grad-CAM visualizations for detections

Implement in `src/tmas/utils/explainability.py`:
- Grad-CAM for mine detections
- Saliency maps for segmentation
- Attention visualizations for fusion
- Helps operators understand system decisions

**Verification**: Visualizations highlight decision-making regions.

---

### Step 14.9: Implement Mission Replay and Analysis Tool
**Deliverable**: Post-mission analysis application

Implement Qt6 application:
- Load MCAP mission recordings
- Replay mission with all detections
- Annotate missed detections
- Generate improvement reports

**Verification**: Replay tool works on sample missions.

---

### Step 14.10: Implement Multi-Vehicle Coordination
**Deliverable**: Share threat maps between multiple vehicles

Implement ROS 2 multi-vehicle nodes:
- Broadcast detected threats to other vehicles
- Merge threat maps from multiple sources
- Collaborative mapping

**Verification**: Multiple vehicles share detections successfully.

---

## Milestone Summary

| Milestone | Week | Deliverable |
|-----------|------|-------------|
| M1: Project Setup Complete | Week 2 | All infrastructure in place |
| M2: Data Pipeline Ready | Week 4 | All datasets loaded and accessible |
| M3: Synthetic Data Generated | Week 6 | 100k synthetic images created |
| M4: Terrain Segmentation Trained | Week 8 | mIoU >75% achieved |
| M5: Mine Detection Trained | Week 12 | Recall >95% on synthetic |
| M6: Obstacle Detection Trained | Week 14 | Recall >99% for critical classes |
| M7: BEV System Integrated | Week 16 | Full BEV maps generated |
| M8: System Integration Complete | Week 18 | End-to-end pipeline working |
| M9: Models Optimized | Week 20 | TensorRT models <25ms latency |
| M10: ROS 2 Integration Complete | Week 24 | Full ROS 2 system operational |
| M11: Testing Complete | Week 28 | All validation tests pass |
| M12: Documentation Complete | Week 32 | All docs and tutorials ready |
| M13: Field Testing Prep | Week 36 | Ready for external testing |
| M14: Advanced Features | Week 40 | Optional enhancements complete |

---

## Critical Success Factors

1. **High Recall Priority**: All design decisions prioritize recall over precision for safety
2. **Real-Time Performance**: System must meet <25ms latency and >20 FPS targets
3. **Multi-Modal Fusion**: RGB + Thermal fusion is critical for robust detection
4. **Graceful Degradation**: System must handle sensor failures safely
5. **Uncertainty Quantification**: Operators need to trust system confidence scores
6. **Field Validation**: Synthetic data must transfer to real-world scenarios

---

## Risk Mitigation Strategies

| Risk | Mitigation |
|------|------------|
| Insufficient real mine data | Heavy reliance on synthetic data + domain randomization |
| Poor sim-to-real transfer | Domain adaptation, continual learning, extensive validation |
| Latency exceeds target | TensorRT INT8 quantization, model distillation, hardware upgrade |
| False negatives (missed mines) | Ensemble models, temporal fusion, conservative thresholds |
| Sensor synchronization issues | Software buffering, timestamp alignment, fallback to single sensor |
| Field testing delays | Thorough simulation and synthetic validation first |

---

## Next Steps After Implementation

1. **Hardware Procurement**: Order Jetson AGX Orin + cameras + GPS
2. **Field Testing Coordination**: Partner with military/engineering units
3. **Real Data Collection**: Collect 500+ scenarios with mine replicas
4. **Fine-Tuning on Real Data**: Domain adaptation to real-world conditions
5. **Safety Certification**: MIL-STD compliance testing
6. **Pilot Deployment**: Operational testing with trained personnel
7. **Continuous Improvement**: Active learning from field data

---

**Document End**

*This implementation plan provides atomic, indivisible tasks for building the complete TMAS system. Each step is designed to be independently verifiable and contributes directly to the final system.*
