# Synthetic Data Generation Pipeline

Complete pipeline for generating synthetic mine detection training data using Blender.

## Overview

This pipeline generates photorealistic synthetic images of mines and IEDs in various terrains, lighting conditions, and burial states. The synthetic data addresses the critical shortage of real mine detection datasets while providing perfect ground truth annotations.

---

## Setup

### 1. Install Blender

Run the setup script to install Blender 4.0+ with Python support:

```bash
bash scripts/data_preparation/synthetic/setup_blender.sh
```

This will:
- Download and install Blender 4.0.2 to `~/.local/blender/`
- Install required Python packages (numpy, opencv-python) in Blender's Python
- Test headless rendering capability
- Create a symlink at `~/.local/blender/blender`

After installation, add Blender to your PATH:

```bash
export PATH="${HOME}/.local/blender:$PATH"
```

Add this line to your `~/.bashrc` or `~/.zshrc` for permanent access.

### 2. Verify Installation

```bash
blender --version
blender --background --python-expr "import numpy; import cv2; print('Dependencies OK')"
```

Expected output:
```
Blender 4.0.2
Dependencies OK
```

---

## Pipeline Components

### Core Modules

1. **terrain_generator.py** - Procedural terrain generation
   - 4 terrain types: desert, grassland, rocky, forest
   - PBR materials with noise/bump mapping
   - Scatter objects (rocks, grass)
   - Size: 20m × 20m grid

2. **create_placeholder_mines.py** - Mine model generation
   - 9 mine types (AP blast, AT blast, UXO, IED)
   - Realistic dimensions based on actual munitions
   - Automatic material assignment

3. **mine_placement.py** - Mine placement and burial simulation
   - Random placement with collision detection
   - Burial depths: 0-15cm
   - Weathering effects
   - Orientation randomization

4. **lighting_weather.py** - Lighting and weather simulation
   - 4 times of day: dawn, day, dusk, night
   - 4 weather conditions: clear, overcast, fog, rain
   - Sun position and color temperature
   - Volumetric fog and rain particles

5. **thermal_simulation.py** - Thermal camera simulation
   - Material-based temperature emission
   - LWIR (8-14μm) wavelength
   - Resolution: 640×512
   - Temperature range: -20°C to 80°C

6. **auto_annotate.py** - Automatic COCO annotation generation
   - 3D to 2D bounding box projection
   - 8 mine classes
   - Burial depth metadata
   - Position and weathering info

7. **domain_randomization.py** - Domain randomization
   - Color/texture variation (HSV space)
   - Material property randomization
   - Camera position/FOV variation
   - Lighting randomization

8. **batch_render.py** - Multi-threaded batch rendering
   - Parallel rendering (4+ workers)
   - Progress tracking
   - Automatic annotation merging
   - Resume capability

9. **visualize_annotations.py** - Annotation verification
   - RGB + thermal visualization
   - Class distribution statistics
   - Example grid generation

---

## Quick Start: Generate 100 Images

### Step 1: Generate Placeholder Mine Models

```bash
cd scripts/data_preparation/synthetic

# Generate 9 mine model types
blender --background --python create_placeholder_mines.py
```

This creates models in `data/synthetic/mine_models/`:
- `ap_blast/` - Anti-personnel blast mines (PMN-2, M14, Type-72)
- `at_blast/` - Anti-tank blast mines (TM-62M, M15)
- `uxo/` - Unexploded ordnance (mortars)
- `ied/` - Improvised explosive devices

### Step 2: Run Batch Rendering

```bash
python batch_render.py \
  --num-scenes 100 \
  --workers 4 \
  --output-dir data/synthetic/mines \
  --blender blender
```

Parameters:
- `--num-scenes`: Number of scenes to generate (100)
- `--workers`: Parallel rendering processes (4)
- `--output-dir`: Output directory
- `--blender`: Path to Blender executable

Expected time: ~2-3 hours for 100 scenes (30-40 images/hour on 4 cores)

### Step 3: Verify Annotations

```bash
python visualize_annotations.py \
  --data-dir data/synthetic/mines \
  --random-samples 10 \
  --output-dir data/synthetic/mines/visualizations \
  --class-grid
```

This generates:
- 10 random sample visualizations
- Class example grid showing each mine type
- Statistics summary

---

## Full Dataset Generation (10,000 Images)

For the full 10k training dataset:

```bash
# Generate 10,000 scenes (estimated 5-7 days on 4 cores)
python batch_render.py \
  --num-scenes 10000 \
  --workers 4 \
  --output-dir data/synthetic/mines_10k

# Create train/val/test splits
python -c "
from src.tmas.data.synthetic import create_split_files
create_split_files('data/synthetic/mines_10k', train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
"

# Verify dataset
python visualize_annotations.py \
  --data-dir data/synthetic/mines_10k \
  --random-samples 20
```

---

## Output Structure

After generation, the output directory contains:

```
data/synthetic/mines/
├── rgb/                      # RGB images (1280×720)
│   ├── scene_000000_rgb.png
│   ├── scene_000001_rgb.png
│   └── ...
├── thermal/                  # Thermal images (640×512)
│   ├── scene_000000_thermal.png
│   ├── scene_000001_thermal.png
│   └── ...
├── annotations.json          # COCO format annotations
├── train.json               # Training split IDs
├── val.json                 # Validation split IDs
├── test.json                # Test split IDs
├── progress.json            # Generation progress log
└── generation_stats.json    # Statistics
```

### COCO Annotation Format

```json
{
  "info": {...},
  "images": [
    {
      "id": 1,
      "file_name": "scene_000000_rgb.png",
      "thermal_file_name": "scene_000000_thermal.png",
      "width": 1280,
      "height": 720
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 3,
      "bbox": [450, 320, 85, 65],
      "area": 5525,
      "iscrowd": 0,
      "burial_depth": 0.05,
      "weathering": 0.6,
      "position_3d": [2.5, 1.2, 0.0]
    }
  ],
  "categories": [
    {"id": 1, "name": "ap_blast", "supercategory": "explosive_threat"},
    {"id": 2, "name": "ap_fragmentation", "supercategory": "explosive_threat"},
    {"id": 3, "name": "at_blast", "supercategory": "explosive_threat"},
    ...
  ]
}
```

---

## Loading Synthetic Data

### Using PyTorch DataLoader

```python
from src.tmas.data.synthetic import SyntheticMineDataset
from torch.utils.data import DataLoader

# Load training set
dataset = SyntheticMineDataset(
    data_dir="data/synthetic/mines",
    split="train",
    load_thermal=True
)

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)

# Iterate
for batch in dataloader:
    images = batch["images"]        # [B, 3, H, W]
    thermals = batch["thermals"]    # [B, 1, H, W]
    targets = batch["targets"]      # List of dicts with boxes, labels
```

---

## Customization

### Custom Scene Configuration

Edit `batch_render.py::generate_scene_config()`:

```python
config = {
    "terrain_type": "desert",           # desert, grassland, rocky, forest
    "time_of_day": "day",               # dawn, day, dusk, night
    "weather": "clear",                 # clear, overcast, fog, rain
    "num_mines": 5,                     # Number of mines per scene
    "camera_height": 5.0,               # Camera height in meters
    "camera_distance": 15.0,            # Distance from center
    "camera_angle": 0.0,                # Tilt angle
}
```

### Custom Domain Randomization

Modify `domain_randomization.py::DomainRandomizer`:

```python
randomizer = DomainRandomizer(seed=42)

# Color variation
randomizer.randomize_colors(material, hue_range=0.2)

# Camera randomization
randomizer.randomize_camera_position(camera, position_variation=(3.0, 3.0, 1.5))
randomizer.randomize_camera_fov(camera, fov_range=(30, 60))

# Lighting
randomizer.randomize_sun_strength(sun, strength_range=(0.4, 2.5))
```

---

## Performance Optimization

### Rendering Speed

- **CPU cores**: More workers = faster (4-8 recommended)
- **GPU**: Cycles renderer can use GPU (CUDA/OptiX)
- **Samples**: Lower samples = faster but noisier (128 recommended)
- **Resolution**: Lower resolution = faster (test with 640×480)

### Disk Space

- RGB: ~200KB per image
- Thermal: ~50KB per image
- 10,000 images ≈ 2.5GB
- 100,000 images ≈ 25GB

---

## Troubleshooting

### Blender Not Found

```bash
# Check Blender path
which blender

# Or specify full path
python batch_render.py --blender /usr/local/bin/blender
```

### Out of Memory

```bash
# Reduce workers
python batch_render.py --workers 2

# Or render one at a time
python batch_render.py --workers 1
```

### Slow Rendering

```bash
# Enable GPU rendering (requires CUDA)
export CYCLES_DEVICE=CUDA

# Reduce samples in thermal_simulation.py
scene.cycles.samples = 64  # Default 128
```

### Resume Failed Generation

```bash
# Resume from last completed scene
python batch_render.py --num-scenes 10000 --resume
```

---

## Requirements

- **System**: Linux x64 (Ubuntu 22.04+ recommended)
- **Blender**: 4.0+ (installed via setup script)
- **Disk**: 4GB for Blender + 25GB per 100k images
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Optional but recommended (NVIDIA CUDA for faster rendering)

---

## Next Steps

After generating synthetic data:

1. **Train baseline detector** using synthetic data only
2. **Evaluate** on validation split
3. **Fine-tune** with real data when available
4. **Domain adaptation** to bridge synthetic-to-real gap

See `docs/PROGRESS_ANALYSIS.md` for full implementation roadmap.
