# Synthetic Data Generation

This directory contains scripts for generating synthetic mine detection training data using Blender.

## Setup

### Install Blender

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

### Verify Installation

```bash
blender --version
blender --background --python-expr "import numpy; import cv2; print('Dependencies OK')"
```

## Usage

The synthetic data generation pipeline consists of:

1. **Terrain Generation**: Procedural terrain with realistic textures
2. **Mine Placement**: Random placement with burial depth simulation
3. **Lighting**: Day/night cycles, weather conditions
4. **Thermal Simulation**: Temperature-based thermal rendering
5. **Annotation**: Automatic COCO format annotation generation

## Directory Structure

```
synthetic/
├── setup_blender.sh          # Blender installation script
├── README.md                 # This file
├── terrain_generator.py      # Procedural terrain generation (TBD)
├── mine_models/              # 3D mine models (TBD)
├── render_scene.py           # Main rendering script (TBD)
└── generate_dataset.py       # Batch generation pipeline (TBD)
```

## Requirements

- Linux x64 system
- 4GB+ free disk space for Blender
- 100GB+ free space for synthetic dataset
- GPU with CUDA support (recommended for faster rendering)

## Target Output

- 100,000 RGB images (1280x720)
- 100,000 thermal images (640x512)
- COCO format annotations
- Multiple terrain types: desert, forest, grassland, rocky
- Burial depths: 0-15cm
- Weather conditions: clear, overcast, rain, fog
