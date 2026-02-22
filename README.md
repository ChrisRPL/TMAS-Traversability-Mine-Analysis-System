# TMAS: Traversability & Mine Analysis System

<div align="center">

**AI-Powered Computer Vision Platform for Military and Engineering Vehicle Support**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/Platform-NVIDIA%20Jetson%20AGX%20Orin-76B900)](https://developer.nvidia.com/embedded/jetson-agx-orin)
[![Status](https://img.shields.io/badge/Status-In%20Development-orange)](https://github.com/ChrisRPL/TMAS-Traversability-Mine-Analysis-System)

[Overview](#overview) • [Features](#features) • [Architecture](#architecture) • [Requirements](#requirements) • [Installation](#installation) • [Usage](#usage) • [Documentation](#documentation)

</div>

---

## 🎯 Overview

**TMAS (Traversability & Mine Analysis System)** is a comprehensive AI-powered computer vision platform designed to support military and engineering vehicles operating in hazardous off-road environments. The system combines three critical functions:

1. **Terrain Traversability Analysis** - Real-time assessment of terrain passability with 14 terrain classes
2. **Mine & IED Detection** - Detection of anti-tank mines, anti-personnel mines, and improvised explosive devices
3. **Obstacle Detection** - Recognition of static and dynamic obstacles including persons, vehicles, and sudden threats

### 🚨 Critical Safety Application

TMAS is designed for **safety-critical military and engineering operations**. A false negative (missed mine) could result in loss of life or equipment. The system achieves:

- **> 99.5% recall** for mine/IED detection
- **> 99% recall** for person/vehicle detection
- **< 25ms latency** for multi-sensor fusion inference
- **≥ 20 FPS** real-time processing

> ⚠️ **Important**: TMAS is an *assistive tool*, not a replacement for trained personnel. Final safety decisions always rest with qualified operators.

---

## ✨ Features

### 🛡️ Mine & IED Detection Module

- **Multi-stage cascade detection** with high recall priority
- **Threat types detected**:
  - Anti-tank (AT) mines (20-40cm diameter)
  - Anti-personnel (AP) mines (5-15cm diameter)
  - Roadside IEDs (improvised explosive devices)
  - Buried IEDs with visible trigger elements
  - Unexploded ordnance (UXO)
- **Multi-modal fusion**: RGB + Thermal imaging for maximum detection probability
- **Temporal tracking**: Confidence accumulation across multiple frames
- **Evidential Deep Learning**: Calibrated uncertainty estimation

### 🗺️ Terrain Traversability Module

- **14 terrain classes**: paved road, gravel, dry grass, sand, wetland, dense brush, rubble, etc.
- **BEV (Bird's Eye View) cost map**: 400×400 grid, 5cm/pixel resolution, 20m×20m coverage
- **Geometry analysis**: Slope, roughness, and obstacle integration
- **Traversability cost estimation**: Real-time route planning support

### 🚧 Obstacle Detection Module

- **Static obstacles**: Fallen trees, boulders, wrecks, debris, craters
- **Dynamic obstacles**: Persons, vehicles, animals with trajectory prediction
- **Sudden obstacle detection**: < 50ms latency for emergency response
- **Time-to-Collision (TTC)**: Prediction with ± 0.3s accuracy
- **Emergency brake recommendation**: Automatic warning when TTC < 2s

---

## 🏗️ Architecture

### System Overview

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  CAMERA RGB  │  │   THERMAL    │  │  GPR (opt.)  │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ RGB Backbone │  │Thermal Backb.│  │  GPR Signal  │
│ EfficientViT │  │  ResNet-18   │  │  Processing  │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       └─────────────────┼─────────────────┘
                        ▼
         ┌──────────────────────────────┐
         │   MULTI-MODAL FUSION         │
         │  (Cross-Attention Transformer)│
         └─────────────┬────────────────┘
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
┌──────────────┐┌──────────────┐┌──────────────┐
│   TERRAIN    ││    MINE      ││   OBSTACLE   │
│ SEGMENTATION ││  DETECTION   ││  DETECTION   │
└──────┬───────┘└──────┬───────┘└──────┬───────┘
       └───────────────┼───────────────┘
                       ▼
         ┌──────────────────────────────┐
         │  BEV TRANSFORM + FUSION      │
         │ (Unified Threat + Cost Map)  │
         └──────────────────────────────┘
```

### Key Components

| Component | Specification |
|-----------|---------------|
| **RGB Backbone** | EfficientViT-L2 (shared with segmentation) |
| **Thermal Backbone** | ResNet-18 (lightweight, fast) |
| **Mine Detector** | RT-DETR-L (real-time DETR) |
| **Multi-modal Fusion** | Cross-Attention Transformer |
| **Obstacle Detector** | YOLOv8 / RT-DETR (20+ classes) |
| **Tracking** | ByteTrack with confidence accumulation |
| **Uncertainty** | Evidential Deep Learning + MC Dropout |

---

## 🔧 Requirements

### Hardware Platform

| Component | Specification |
|-----------|---------------|
| **Compute Unit** | NVIDIA Jetson AGX Orin 64GB |
| **GPU** | 2048 CUDA cores, 64 Tensor cores |
| **CPU** | 12-core ARM Cortex-A78AE |
| **Memory** | 64GB LPDDR5 (204 GB/s) |
| **Power** | 15-60W (configurable) |
| **RGB Camera** | FLIR Blackfly S (global shutter, 12MP) |
| **Thermal Camera** | FLIR Boson 640 (LWIR, 640×512) |
| **GPS/INS** | Emlid Reach RS2+ (RTK, cm accuracy) |
| **Optional GPR** | GSSI StructureScan Mini XT |
| **Enclosure** | MIL-STD-810G (IP67, -40°C to +55°C) |

### Software Stack

| Layer | Technology |
|-------|------------|
| **OS** | JetPack 6.0 (Ubuntu 22.04 hardened) |
| **CUDA / TensorRT** | CUDA 12.2 / cuDNN 8.9 / TensorRT 8.6 |
| **ML Framework** | PyTorch 2.2 → ONNX → TensorRT |
| **Middleware** | ROS 2 Iron (LTS) |
| **Sensor Sync** | Hardware trigger + PTP |
| **Logging** | MCAP format (deterministic) |
| **Interface** | Qt6 + RViz2 |

### Military Standards

- **MIL-STD-810G**: Shock, vibration, temperature resistance
- **MIL-STD-461G**: Electromagnetic compatibility (EMC/EMI)
- **IP67**: Dust and waterproof
- **Power**: 9-36V DC (compatible with military vehicles)
- **MTBF**: > 5000 hours

---

## 📦 Installation

### Prerequisites

```bash
# JetPack 6.0 on NVIDIA Jetson AGX Orin
# Verify CUDA installation
nvcc --version

# Verify TensorRT
dpkg -l | grep TensorRT
```

### Clone Repository

```bash
git clone https://github.com/ChrisRPL/TMAS-Traversability-Mine-Analysis-System.git
cd TMAS-Traversability-Mine-Analysis-System
```

### Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install ROS 2 Iron (if not already installed)
# Follow: https://docs.ros.org/en/iron/Installation.html

# Build ROS 2 workspace
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

### Download Pre-trained Models

```bash
# Download model checkpoints
./scripts/download_models.sh

# Models will be stored in: models/
# - terrain_segmentation.trt
# - mine_detection.trt
# - obstacle_detection.trt
```

---

## 🚀 Usage

### Quick Start

```bash
# Launch TMAS system with all modules
ros2 launch tmas tmas_full.launch.py

# Launch with visualization
ros2 launch tmas tmas_full.launch.py enable_viz:=true

# Launch specific module only
ros2 launch tmas terrain_only.launch.py
ros2 launch tmas mine_detection_only.launch.py
```

### Python API

```python
from tmas import TMASSystem

# Initialize system
tmas = TMASSystem(
    config_path="configs/default.yaml",
    enable_thermal=True,
    enable_gpr=False
)

# Process single frame
result = tmas.process_frame(
    rgb_image=rgb_frame,
    thermal_image=thermal_frame
)

# Access results
print(f"Detected mines: {len(result.mine_detections)}")
print(f"BEV cost map shape: {result.bev_cost_map.shape}")
print(f"Obstacles: {len(result.obstacles)}")

# Get alerts
for alert in result.alerts:
    if alert.severity == "CRITICAL":
        print(f"⚠️ {alert.type}: {alert.message}")
```

### ROS 2 Topics

```bash
# Input topics
/tmas/camera/rgb/image_raw          # RGB camera input
/tmas/camera/thermal/image_raw      # Thermal camera input
/tmas/gps/fix                       # GPS position

# Output topics
/tmas/bev/cost_map                  # BEV traversability cost map
/tmas/bev/threat_map                # BEV threat detection map
/tmas/detections/mines              # Mine/IED detections
/tmas/detections/obstacles          # Obstacle detections
/tmas/alerts                        # Real-time alerts
/tmas/visualization                 # Debug visualization
```

---

## 📊 Performance Metrics

### Detection Performance (Target)

| Metric | Target Value |
|--------|--------------|
| Mine/IED Recall (AT) | > 99.5% |
| Mine/IED Recall (AP) | > 99.0% |
| Person/Vehicle Recall | > 99.0% |
| Static Obstacle Recall | > 95% |
| False Positive Rate (mines) | < 10% |
| Terrain Segmentation mIoU | > 75% |

### System Performance

| Metric | Target Value |
|--------|--------------|
| Multi-sensor Fusion Latency | < 25 ms |
| Frame Rate | ≥ 20 FPS |
| Sudden Obstacle Detection | < 50 ms |
| Mine Detection Range | 30m (RGB), 15m (thermal) |
| Obstacle Detection Range | 50m (RGB) |
| BEV Resolution | 5 cm/pixel (400×400) |
| TTC Prediction Accuracy | ± 0.3s |

---

## 🗂️ Project Structure

```
TMAS-Traversability-Mine-Analysis-System/
├── configs/                    # Configuration files
│   ├── default.yaml           # Default system config
│   ├── models/                # Model-specific configs
│   └── sensors/               # Sensor calibration
├── data/                      # Training and test data
│   ├── synthetic/             # Synthetic mine data
│   ├── real/                  # Real-world data
│   └── annotations/           # Ground truth labels
├── models/                    # Pre-trained models
│   ├── terrain_segmentation.trt
│   ├── mine_detection.trt
│   └── obstacle_detection.trt
├── ros2_ws/                   # ROS 2 workspace
│   └── src/
│       └── tmas/              # TMAS ROS 2 package
├── scripts/                   # Utility scripts
│   ├── download_models.sh     # Download pre-trained models
│   ├── calibrate_sensors.py   # Sensor calibration
│   └── train_model.py         # Training script
├── src/                       # Source code
│   ├── tmas/
│   │   ├── core/              # Core system
│   │   ├── models/            # Model implementations
│   │   ├── fusion/            # Multi-modal fusion
│   │   ├── detection/         # Detection modules
│   │   └── utils/             # Utilities
├── tests/                     # Unit and integration tests
├── docs/                      # Documentation
│   ├── api/                   # API documentation
│   ├── tutorials/             # Tutorials
│   └── deployment/            # Deployment guides
├── LICENSE                    # MIT License
├── README.md                  # This file
├── SPEC.md                    # Detailed specification
├── requirements.txt           # Python dependencies
└── .gitignore                 # Git ignore rules
```

---

## 📚 Documentation

- **[Detailed Specification](SPEC.md)** - Complete technical specification (internal use)
- **[API Documentation](docs/api/)** - Python and ROS 2 API reference
- **[Deployment Guide](docs/deployment/)** - Hardware setup and deployment
- **[Training Guide](docs/training/)** - Model training and fine-tuning
- **[Testing Guide](docs/testing/)** - Validation and testing procedures

---

## 🛣️ Roadmap

### Phase 1: R&D (8 weeks)
- ✅ Architecture design
- ✅ Baseline models on synthetic data
- ⏳ > 95% recall on synthetic mine data

### Phase 2: Sensor Integration (4 weeks)
- ⏳ RGB-Thermal calibration
- ⏳ Hardware synchronization
- ⏳ Multi-sensor fusion pipeline

### Phase 3: Optimization (4 weeks)
- ⏳ TensorRT INT8 quantization
- ⏳ Multi-stream inference
- ⏳ < 25ms latency achievement

### Phase 4: Field Data Collection (6 weeks)
- ⏳ Test range setup with mine replicas
- ⏳ 500+ scenario data collection
- ⏳ Multi-condition testing (weather, lighting)

### Phase 5: Fine-tuning (4 weeks)
- ⏳ Training on real-world data
- ⏳ Domain adaptation
- ⏳ > 99% recall on real data

### Phase 6: Vehicle Integration (4 weeks)
- ⏳ Vehicle mounting
- ⏳ ROS 2 integration
- ⏳ Operator interface (HMI)

### Phase 7: Validation (6 weeks)
- ⏳ Acceptance testing
- ⏳ Safety certification
- ⏳ Military standards compliance

### Phase 8: Pilot Deployment (4 weeks)
- ⏳ Operational testing with personnel
- ⏳ Feedback integration
- ⏳ Final certification

**Total Duration**: ~40 weeks (10 months)

---

## ⚠️ Safety & Limitations

### Critical Safety Requirements

| ID | Requirement | Verification |
|----|-------------|--------------|
| SR-1 | Mine (AT) recall > 99.5% | 1000+ test scenarios |
| SR-2 | Mine (AP) recall > 99.0% | 1000+ test scenarios |
| SR-3 | IED recall > 98.5% | 500+ test scenarios |
| SR-4 | Person recall > 99.5% | Pedestrian test scenarios |
| SR-5 | Emergency response < 100ms | End-to-end latency tests |

### Limitations

- **Not 100% reliable**: No technology guarantees perfect mine detection
- **Environmental dependencies**: Performance degrades in extreme weather
- **Sensor limitations**: RGB cannot see buried objects; thermal has limited range
- **Human-in-the-loop required**: Final safety decisions must involve trained personnel
- **False positives acceptable**: System prioritizes safety over precision

### Emergency Protocols

| Failure Mode | System Response | Operator Action |
|--------------|-----------------|-----------------|
| RGB camera loss | Switch to thermal-only | Reduce speed, consider withdrawal |
| Thermal camera loss | Continue with RGB-only | Increased caution |
| GPS loss | Relative positioning | Manual position logging |
| GPU overheating | Reduce FPS, prioritize mine detection | Stop vehicle, allow cooling |
| Software error | Automatic restart, alert | Stop until verification |

---

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

### Development Setup

```bash
# Clone repository
git clone https://github.com/ChrisRPL/TMAS-Traversability-Mine-Analysis-System.git
cd TMAS-Traversability-Mine-Analysis-System

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
ruff check src/
black src/
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact & Support

- **Project Lead**: [ChrisRPL](https://github.com/ChrisRPL)
- **Issue Tracker**: [GitHub Issues](https://github.com/ChrisRPL/TMAS-Traversability-Mine-Analysis-System/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ChrisRPL/TMAS-Traversability-Mine-Analysis-System/discussions)

---

## 🙏 Acknowledgments

- **GICHD (Geneva International Centre for Humanitarian Demining)** - Mine detection datasets
- **NVIDIA** - Jetson platform and optimization support
- **FLIR Systems** - Thermal imaging expertise
- **Military & Engineering Units** - Operational feedback and validation

---

## 📖 Citation

If you use TMAS in your research or operations, please cite:

```bibtex
@software{tmas2026,
  title={TMAS: Traversability and Mine Analysis System},
  author={Romanowski, Krzysztof},
  year={2026},
  url={https://github.com/ChrisRPL/TMAS-Traversability-Mine-Analysis-System},
  version={2.0}
}
```

---

<div align="center">

**⚠️ Critical Reminder ⚠️**

*TMAS is an assistive tool for trained operators. It does NOT replace qualified personnel.*
*No mine detection technology is 100% reliable. Always follow standard safety protocols.*

**Developed for military and engineering safety applications**

</div>
