# TMAS Project - AI Development Guide

**Traversability & Mine Analysis System**
**Safety-Critical Military/Engineering Application**

---

## ⚠️ Critical Context

**This is a SAFETY-CRITICAL system for military and engineering operations.**

A false negative (missed mine) can result in **loss of life or equipment**. All development decisions must prioritize:

1. **Recall over Precision** - Better to have false alarms than miss threats
2. **Graceful Degradation** - System must handle failures safely
3. **Uncertainty Quantification** - Operators need trustworthy confidence scores
4. **Real-Time Performance** - <25ms latency, >20 FPS required
5. **Human-in-the-Loop** - System assists, never replaces human decision-making

---

## Project Overview

TMAS is a multi-modal AI computer vision system with three core modules:

### **Module 1: Terrain Traversability Analysis**
- 14 terrain classes (paved road → rubble)
- BEV cost map (400×400 grid, 5cm/pixel, 20m range)
- Geometry analysis (slope, roughness)
- Target: >75% mIoU

### **Module 2: Mine & IED Detection**
- AT mines, AP mines, IEDs, UXO detection
- RGB + Thermal fusion for maximum recall
- Multi-stage cascade (anomaly → classification → verification)
- Target: >99.5% recall (CRITICAL)

### **Module 3: Obstacle Detection**
- Persons, vehicles, animals (dynamic)
- Trees, rocks, craters, debris (static)
- Sudden obstacle detection (<50ms latency)
- Time-to-Collision (TTC) prediction (±0.3s)
- Target: >99% recall for persons/vehicles

### **System Integration**
- Multi-modal sensor fusion (RGB + Thermal + optional GPR)
- BEV transformation and fusion
- Real-time alert generation
- ROS 2 middleware
- MCAP logging for post-analysis

---

## Current Status

✅ **Completed:**
- Project structure and README
- Implementation plan (200+ atomic tasks)
- Specification document (SPEC.md - gitignored)

⧖ **In Progress:**
- Phase 1: Project setup & infrastructure
- Phase 2: Data acquisition

⏳ **Upcoming:**
- Synthetic data generation (Blender pipeline)
- Model training (terrain → mines → obstacles)
- System integration and ROS 2 deployment

---

## Codebase Structure

```
TMAS-Traversability-Mine-Analysis-System/
├── configs/                    # YAML configurations
│   ├── models/                # Model configs
│   ├── sensors/               # Sensor calibration (gitignored)
│   ├── training/              # Training hyperparameters
│   └── default.yaml          # System-wide defaults
├── data/                      # Datasets (gitignored)
│   ├── raw/                  # Downloaded datasets
│   ├── processed/            # Preprocessed data
│   ├── synthetic/            # Synthetic Blender data
│   ├── real/                 # Real field data
│   └── annotations/          # Ground truth labels
├── models/                    # Model artifacts (gitignored)
│   ├── checkpoints/          # Training checkpoints (.pth)
│   └── exports/              # ONNX/TensorRT exports
├── src/tmas/                  # Main source code
│   ├── core/                 # System core (config, sync, TMAS system)
│   ├── models/               # Model architectures
│   │   ├── backbones/        # EfficientViT, ResNet-18
│   │   ├── segmentation/     # Terrain segmentation
│   │   ├── detection/        # Mine & obstacle detection
│   │   ├── fusion/           # Cross-attention fusion
│   │   ├── uncertainty/      # Evidential Deep Learning
│   │   └── losses/           # Loss functions
│   ├── data/                 # Data loaders
│   ├── tracking/             # ByteTrack, trajectory prediction
│   ├── bev/                  # BEV transformation
│   ├── detection/            # Inference pipelines
│   ├── segmentation/         # Segmentation inference
│   ├── hmi/                  # Qt6 operator interface
│   ├── utils/                # Utilities (logging, metrics)
│   └── visualization/        # Visualization tools
├── scripts/                   # Utility scripts
│   ├── data_preparation/     # Dataset download, preprocessing
│   │   └── synthetic/        # Blender rendering pipeline
│   ├── training/             # Training scripts
│   ├── evaluation/           # Evaluation and benchmarking
│   ├── deployment/           # ONNX/TensorRT export
│   └── ros2/                 # ROS 2 utilities
├── tests/                     # Test suite
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   ├── performance/          # Performance benchmarks
│   └── validation/           # Recall/safety validation
├── notebooks/                 # Jupyter notebooks
│   ├── 01_data_exploration/
│   ├── 02_model_development/
│   └── 03_evaluation/
├── ros2_ws/                   # ROS 2 workspace (gitignored build/)
│   └── src/tmas_ros2/        # TMAS ROS 2 package
├── docs/                      # Documentation
│   ├── api/                  # API reference
│   ├── tutorials/            # User tutorials
│   └── deployment/           # Deployment guides
├── agent-scripts/             # Development tools (gitignored)
│   └── (cloned from steipete/agent-scripts)
├── CLAUDE.md                  # This file
├── IMPLEMENTATION_PLAN.md     # 200+ atomic tasks
├── README.md                  # Public documentation
├── SPEC.md                    # Technical spec (gitignored)
└── .gitignore
```

---

## Development Workflow

### **Git Strategy: Atomic Commits**

**CRITICAL RULE: Every change must be a separate atomic commit**

This is a **SAFETY-CRITICAL** system. Atomic commits enable:
- Easy rollback of problematic changes
- Clear audit trail for safety certification
- Parallel development by multiple agents
- Precise debugging and bisection

### **Commit Guidelines**

#### **1. One Logical Change = One Commit**

✅ **GOOD - Atomic commits:**
```bash
# Separate commits for each file/change
git commit -m "add efficientvit backbone implementation"
git commit -m "add resnet-18 thermal backbone"
git commit -m "add cross-attention fusion module"
git commit -m "add mine detection model architecture"
git commit -m "add unit tests for fusion module"
```

❌ **BAD - Batch commits:**
```bash
# Multiple unrelated changes in one commit
git commit -m "add all model architectures and tests"
```

#### **2. Commit Message Format**

**Format:**
```
<type>: <concise description in lowercase>

Optional body with more details if needed.
```

**Types:**
- `feat`: New feature or module
- `fix`: Bug fix
- `refactor`: Code refactoring (no behavior change)
- `test`: Add or update tests
- `docs`: Documentation changes
- `perf`: Performance improvements
- `chore`: Build, CI, dependencies
- `config`: Configuration changes

**Examples:**
```bash
feat: add efficientvit-l2 backbone with pretrained weights
feat: implement cross-attention rgb-thermal fusion
feat: add evidential deep learning uncertainty estimation
fix: correct masking bug in mine detection dataloader
fix: resolve tensor shape mismatch in bev transformation
refactor: simplify terrain cost calculation logic
test: add unit tests for bytetrack integration
test: add recall validation tests for mine detection
docs: add training tutorial for terrain segmentation
perf: optimize bev transformation with vectorization
config: update mine detection threshold to 0.15 for recall
chore: add tensorrt to requirements
```

#### **3. Never Mention AI Tools in Commits**

❌ **BAD:**
```
feat: implement fusion module with Claude's help
fix: bug found by AI assistant
```

✅ **GOOD:**
```
feat: implement cross-attention fusion module
fix: resolve dimension mismatch in fusion forward pass
```

#### **4. Verification Before Commit**

Before committing:
1. **Run tests**: `pytest tests/` (or relevant subset)
2. **Run linting**: `ruff check src/` and `black --check src/`
3. **Verify functionality**: Does the change work as intended?
4. **Check imports**: No broken imports or circular dependencies

### **Multi-Agent Development Strategy**

This project can be developed by multiple AI agents working in parallel on different phases of the IMPLEMENTATION_PLAN.md.

#### **Agent Roles and Task Assignment**

**Agent 1: Data Engineer**
- Phase 2: Data acquisition and preparation
- Phase 3: Synthetic data generation
- Responsibilities:
  - Download and organize datasets
  - Create data loaders
  - Build Blender rendering pipeline
  - Generate 100k synthetic images

**Agent 2: Model Architect (Terrain)**
- Phase 4: Terrain segmentation development
- Responsibilities:
  - Implement EfficientViT backbone
  - Build segmentation decoder
  - Train and evaluate terrain model
  - Achieve >75% mIoU target

**Agent 3: Model Architect (Detection)**
- Phase 5: Mine detection development
- Phase 6: Obstacle detection development
- Responsibilities:
  - Implement RT-DETR detection heads
  - Build multi-modal fusion
  - Train mine detector (>99.5% recall)
  - Train obstacle detector (>99% recall)

**Agent 4: Integration Engineer**
- Phase 7: BEV transformation
- Phase 8: System integration
- Responsibilities:
  - Implement BEV projection
  - Integrate all modules
  - Build alert system
  - Create unified inference pipeline

**Agent 5: Optimization Engineer**
- Phase 9: Model optimization
- Responsibilities:
  - Export to ONNX
  - Convert to TensorRT
  - Optimize for <25ms latency
  - Benchmark on target hardware

**Agent 6: ROS 2 Engineer**
- Phase 10: ROS 2 integration
- Responsibilities:
  - Create ROS 2 nodes for each module
  - Build launch files
  - Implement MCAP logging
  - Create RViz2 visualizations

**Agent 7: Test Engineer**
- Phase 11: Testing & validation
- Responsibilities:
  - Write comprehensive test suite
  - Validate recall requirements
  - Test failure modes
  - Safety validation report

**Agent 8: Documentation Engineer**
- Phase 12: Documentation
- Responsibilities:
  - API documentation
  - Tutorials and guides
  - Operator manual
  - Deployment guides

#### **Parallel Development Protocol**

1. **Task Assignment**: Each agent picks tasks from IMPLEMENTATION_PLAN.md
2. **Communication**: Use git commit messages and PR descriptions
3. **Branch Strategy**:
   - `main` branch: Production-ready code
   - Feature branches: `feature/terrain-segmentation`, `feature/mine-detection`, etc.
   - Agent creates branch, commits atomically, creates PR
4. **Merge Protocol**:
   - All tests must pass
   - Code review by lead agent (if applicable)
   - Squash and merge NOT allowed (preserve atomic commits)
5. **Conflict Resolution**:
   - Communicate via PR comments
   - Last merged wins for config conflicts
   - Code conflicts resolved by agent lead

#### **Atomic Task Shipping Process**

Follow this process for EVERY task in IMPLEMENTATION_PLAN.md:

**Step 1: Plan**
- Read task description and deliverables
- Identify files to create/modify
- Check dependencies (previous tasks completed?)

**Step 2: Implement**
- Create/modify ONE file at a time
- Test the change immediately
- Verify it works

**Step 3: Commit**
- Stage the file: `git add <file>`
- Commit with clear message: `git commit -m "feat: add <component>"`
- Push immediately: `git push origin <branch>`

**Step 4: Verify**
- Run verification criteria from task
- Run tests: `pytest tests/`
- Check metrics (if applicable)

**Step 5: Document**
- Update task status in IMPLEMENTATION_PLAN.md (optional)
- Add comments to code if complex
- Update README if public-facing

**Example: Implementing Step 4.1 (EfficientViT Backbone)**

```bash
# Step 1: Plan
# - Create src/tmas/models/backbones/efficientvit.py
# - Load pretrained EfficientViT-L2 from timm
# - Return multi-scale features

# Step 2: Implement
# (Write code in efficientvit.py)

# Step 3: Commit
git add src/tmas/models/backbones/efficientvit.py
git commit -m "feat: add efficientvit-l2 backbone with multi-scale features"
git push origin feature/terrain-segmentation

# Step 4: Verify
pytest tests/unit/test_efficientvit.py
# Output: Forward pass works, output shapes correct ✓

# Step 5: Document
# (Add docstrings to code, update API docs if needed)
```

---

## Development Tools

### **Agent Scripts (Development Helper)**

Clone the agent-scripts repository for useful development utilities:

```bash
cd /path/to/TMAS-Traversability-Mine-Analysis-System
git clone https://github.com/steipete/agent-scripts.git
```

**Available scripts:**
- `agent-scripts/commiter.sh` - Automated atomic commit helper
- `agent-scripts/reviewer.sh` - Code review automation
- `agent-scripts/tester.sh` - Test runner wrapper
- Other utilities for AI-assisted development

**Usage example:**
```bash
# Use commiter for atomic commits
./agent-scripts/commiter.sh src/tmas/models/backbones/efficientvit.py
# Script prompts for commit message and commits atomically

# Run tests with test runner
./agent-scripts/tester.sh tests/unit/test_efficientvit.py
```

⚠️ **These scripts are development-only and gitignored.**

### **Available Claude Tools**

- **WebSearch**: Search for papers, PyTorch docs, dataset sources
- **Bash**: Run tests, git operations, file operations, download data
- **Read/Write/Edit**: File creation and modification
- **Glob/Grep**: Code navigation and search
- **Task**: Spawn sub-agents for complex multi-step tasks
- **TodoWrite**: Track progress on multi-step implementations

### **Pre-commit Hooks**

Pre-commit hooks are configured in `.pre-commit-config.yaml`:
- Black (code formatting)
- Ruff (linting)
- MyPy (type checking)

Install hooks:
```bash
pre-commit install
```

Hooks run automatically before each commit, ensuring code quality.

---

## Code Quality Standards

### **Python Style**

- **PEP 8** compliance (enforced by Black + Ruff)
- **Type hints** for function signatures
- **Docstrings** for all public classes/functions (Google style)
- **Max line length**: 88 characters (Black default)

**Example:**
```python
def detect_mines(
    rgb_image: torch.Tensor,
    thermal_image: torch.Tensor,
    confidence_threshold: float = 0.15
) -> Dict[str, Any]:
    """Detect mines in RGB and thermal images.

    Args:
        rgb_image: RGB image tensor (B, 3, H, W)
        thermal_image: Thermal image tensor (B, 1, H, W)
        confidence_threshold: Detection confidence threshold (default: 0.15)

    Returns:
        Dictionary containing:
            - detections: List of detection dicts
            - uncertainties: Uncertainty scores
            - tracking_ids: Temporal track IDs
    """
    # Implementation
    pass
```

### **Testing Requirements**

**Test Coverage Targets:**
- Core modules: >90%
- Utility functions: >80%
- Overall: >75%

**Test Types:**
- **Unit tests**: Test individual components
- **Integration tests**: Test module interactions
- **Performance tests**: Benchmark latency/FPS
- **Validation tests**: Verify recall/precision targets

**Running tests:**
```bash
# All tests
pytest tests/

# Specific module
pytest tests/unit/test_mine_detection.py

# With coverage
pytest tests/ --cov=src/tmas --cov-report=html

# Performance benchmarks
pytest tests/performance/ -v
```

### **Logging Standards**

Use Python's `logging` module (not print statements):

```python
import logging

logger = logging.getLogger(__name__)

# Good
logger.info(f"Loaded model checkpoint: {checkpoint_path}")
logger.warning(f"Low confidence detection: {confidence:.2f}")
logger.error(f"Model inference failed: {error}")

# Bad
print(f"Loaded model: {checkpoint_path}")
```

For experiment tracking, use W&B:
```python
import wandb

wandb.log({
    "train/loss": loss.item(),
    "val/recall": recall,
    "val/miou": miou
})
```

---

## Configuration Management

### **Hydra/OmegaConf Configuration System**

All hyperparameters are managed via YAML configs:

**Structure:**
```
configs/
├── default.yaml              # System-wide defaults
├── models/
│   ├── terrain_segmentation.yaml
│   ├── mine_detection.yaml
│   └── obstacle_detection.yaml
├── training/
│   ├── terrain_training.yaml
│   └── mine_training.yaml
└── sensors/
    └── camera_calibration.yaml  # Gitignored
```

**Usage:**
```bash
# Use default config
python scripts/training/train_terrain_segmentation.py

# Override specific parameters
python scripts/training/train_terrain_segmentation.py \
  model.latent_dim=768 \
  training.batch_size=32 \
  training.learning_rate=5e-5
```

**Key Parameters:**

**Terrain Segmentation:**
- `model.backbone`: "efficientvit_l2"
- `model.num_classes`: 14
- `training.batch_size`: 16
- `training.learning_rate`: 1e-4
- `training.epochs`: 50

**Mine Detection:**
- `model.rgb_backbone`: "efficientvit_l2"
- `model.thermal_backbone`: "resnet18"
- `model.num_classes`: 8
- `model.confidence_threshold`: 0.15  # Low for high recall
- `training.batch_size`: 8
- `training.learning_rate`: 5e-5
- `training.epochs`: 100

**Obstacle Detection:**
- `model.detector`: "yolov8_large"
- `model.num_classes`: 20
- `model.confidence_threshold`: 0.25
- `training.batch_size`: 16
- `training.epochs`: 50

---

## Dataset Guidelines

### **Expected Datasets**

**Terrain Segmentation:**
- **RELLIS-3D**: 13,556 frames, RGB + LiDAR, 14 terrain classes
- **Synthetic**: 40k terrain images from Blender

**Mine Detection:**
- **GICHD Mine Database**: 10k+ RGB images (request access)
- **Thermal Mine Dataset**: 5k LWIR images
- **Synthetic**: 60k mine images from Blender (various burial depths)

**Obstacle Detection:**
- **COCO**: Persons, vehicles, animals
- **Open Images**: Debris, barriers, wrecks
- **Synthetic**: 20k obstacle scenarios

### **Data Organization**

```
data/
├── raw/
│   ├── rellis3d/
│   ├── tartandrive/
│   ├── mines_public/
│   └── thermal/
├── processed/
│   ├── rellis3d_preprocessed/
│   └── mines_preprocessed/
├── synthetic/
│   ├── rendered/
│   │   ├── rgb/
│   │   ├── thermal/
│   │   └── annotations/
│   └── 3d_models/
│       └── mines/
├── splits/
│   ├── rellis3d_train.json
│   ├── rellis3d_val.json
│   └── ...
└── registry.json  # Dataset metadata
```

### **Data Preprocessing**

**Images:**
- Resize to standard resolution (1280×720 for RGB, 640×512 for thermal)
- Normalize with ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

**Annotations:**
- Convert all to COCO JSON format
- Class mapping in `src/tmas/data/class_definitions.py`

**Augmentations:**
- See `src/tmas/data/augmentation.py` for safety-critical augmentation strategies

---

## Model Development Guidelines

### **Backbone Selection**

**RGB Backbone (Shared):**
- **EfficientViT-L2**: Efficient transformer for real-time inference
- Pretrained on ImageNet
- Multi-scale feature extraction (stride 4, 8, 16, 32)

**Thermal Backbone:**
- **ResNet-18**: Lightweight CNN for thermal images
- Modified first conv for single-channel input
- Faster than large transformers

### **Loss Function Design**

**Terrain Segmentation:**
```python
loss = 0.5 * CrossEntropyLoss(class_weighted) + 0.5 * DiceLoss()
```

**Mine Detection:**
```python
loss = (
    10.0 * FocalLoss(alpha=0.25, gamma=2.0) +  # Class imbalance
    5.0 * GIoULoss() +                         # Box regression
    1.0 * EvidentialLoss()                     # Uncertainty
)
```
- High weight on mine classes for recall
- Low confidence threshold (0.15) for high sensitivity

**Obstacle Detection:**
```python
loss = FocalLoss() + GIoULoss()
```

### **Training Best Practices**

**Hyperparameters:**
- Optimizer: AdamW (weight_decay=0.01)
- Learning rate: 1e-4 to 5e-5 (depends on model size)
- Scheduler: CosineAnnealingLR with warmup
- Gradient clipping: max_norm=1.0
- Mixed precision: FP16 for faster training

**Checkpointing:**
- Save best model (by validation metric)
- Save last model (for resuming)
- Save every N epochs for safety

**Early Stopping:**
- Monitor validation recall (for mine detection)
- Patience: 10-20 epochs

### **Evaluation Metrics**

**Terrain Segmentation:**
- **Primary**: mIoU (mean Intersection over Union) > 75%
- Per-class IoU for each terrain type
- Confusion matrix

**Mine Detection:**
- **Primary**: Recall @ IoU 0.5 > 99.5% (CRITICAL)
- Precision @ IoU 0.5
- F1 score
- Average Precision (AP)
- Uncertainty calibration (ECE)

**Obstacle Detection:**
- **Primary**: Recall > 99% for persons/vehicles
- Recall > 95% for static obstacles
- TTC accuracy (± 0.3s)
- Latency < 50ms for sudden obstacles

---

## Performance Optimization

### **Target Metrics**

| Metric | Target | Critical? |
|--------|--------|-----------|
| Total latency (end-to-end) | < 50ms | YES |
| Multi-sensor fusion latency | < 25ms | YES |
| Frame rate | ≥ 20 FPS | YES |
| Mine detection recall (AT) | > 99.5% | CRITICAL |
| Mine detection recall (AP) | > 99.0% | CRITICAL |
| Person/vehicle recall | > 99.0% | CRITICAL |
| Terrain segmentation mIoU | > 75% | Important |

### **Optimization Strategy**

**Phase 1: PyTorch Optimization**
- Mixed precision (FP16)
- Gradient checkpointing for large models
- Efficient data loading (num_workers, prefetch_factor)
- Model distillation (if needed)

**Phase 2: ONNX Export**
- Export to ONNX format
- Graph optimization (constant folding, operator fusion)
- Validate accuracy preservation

**Phase 3: TensorRT Conversion**
- Convert ONNX to TensorRT
- FP16 precision (2x speedup)
- INT8 quantization if FP16 insufficient (4x speedup)
- Calibration on representative dataset

**Phase 4: Deployment Optimization**
- Multi-stream execution (parallel module inference)
- GPU memory optimization
- CPU-GPU transfer minimization

### **Benchmarking**

Use `tests/performance/benchmark_latency.py`:
```bash
python tests/performance/benchmark_latency.py \
  --model mine_detection \
  --num_runs 1000 \
  --batch_size 1
```

Expected output:
```
Module: Mine Detection
Mean latency: 22.3ms
Std latency: 1.2ms
FPS: 44.8
GPU memory: 2.1GB
```

---

## ROS 2 Integration

### **Node Architecture**

**Nodes:**
1. `rgb_camera_node` - Publish RGB images
2. `thermal_camera_node` - Publish thermal images
3. `terrain_segmentation_node` - Terrain classification
4. `mine_detection_node` - Mine/IED detection
5. `obstacle_detection_node` - Obstacle detection + TTC
6. `bev_fusion_node` - BEV cost map generation
7. `visualization_node` - Visualization overlays
8. `alert_manager_node` - Alert aggregation

**Topics:**
```
/tmas/camera/rgb/image_raw           (sensor_msgs/Image)
/tmas/camera/thermal/image_raw       (sensor_msgs/Image)
/tmas/segmentation/terrain_map       (sensor_msgs/Image)
/tmas/bev/cost_map                   (nav_msgs/OccupancyGrid)
/tmas/bev/threat_map                 (nav_msgs/OccupancyGrid)
/tmas/detections/mines               (tmas_msgs/MineDetectionArray)
/tmas/detections/obstacles           (tmas_msgs/ObstacleDetectionArray)
/tmas/alerts                         (tmas_msgs/Alert)
/tmas/visualization/image            (sensor_msgs/Image)
```

### **Launch Files**

```bash
# Full system
ros2 launch tmas_ros2 tmas_full.launch.py

# Simulation with video files
ros2 launch tmas_ros2 tmas_simulation.launch.py \
  rgb_video:=/path/to/rgb.mp4 \
  thermal_video:=/path/to/thermal.mp4

# Individual modules
ros2 launch tmas_ros2 mine_detection_only.launch.py
```

### **Recording Sessions**

```bash
# Record all topics
ros2 bag record -a -o tmas_session_001

# Record specific topics
ros2 bag record /tmas/detections/mines /tmas/alerts

# Playback
ros2 bag play tmas_session_001
```

---

## Safety and Validation

### **Safety Requirements (from SPEC.md)**

| ID | Requirement | How to Verify |
|----|-------------|---------------|
| SR-1 | Mine (AT) recall > 99.5% | `tests/validation/test_recall.py` |
| SR-2 | Mine (AP) recall > 99.0% | `tests/validation/test_recall.py` |
| SR-3 | IED recall > 98.5% | `tests/validation/test_recall.py` |
| SR-4 | Person recall > 99.5% | `tests/validation/test_recall.py` |
| SR-5 | Emergency response < 100ms | `tests/performance/benchmark_latency.py` |
| SR-6 | Sudden obstacle latency < 50ms | `tests/performance/benchmark_latency.py` |
| SR-7 | Graceful degradation on failure | `tests/integration/test_failure_modes.py` |
| SR-8 | 100% detection logging | `tests/integration/test_mcap_logging.py` |

### **Validation Checklist**

Before deployment:
- ✅ All unit tests pass (`pytest tests/unit/`)
- ✅ All integration tests pass (`pytest tests/integration/`)
- ✅ Recall validation tests pass (`pytest tests/validation/`)
- ✅ Performance benchmarks meet targets
- ✅ Failure mode tests pass
- ✅ Field data validation (if available)
- ✅ Safety validation report generated
- ✅ Documentation complete

---

## Troubleshooting

### **Common Issues**

**Issue: Low recall on mine detection**
- Solution: Lower confidence threshold (try 0.1, 0.05)
- Solution: Increase temporal voting window
- Solution: Use ensemble of multiple models
- Solution: Check for data distribution mismatch

**Issue: High latency (>50ms)**
- Solution: Convert to TensorRT FP16
- Solution: Reduce model size (use lighter backbones)
- Solution: Enable multi-stream execution
- Solution: Profile with PyTorch profiler

**Issue: Poor thermal fusion**
- Solution: Check thermal normalization (should be [0, 1])
- Solution: Verify thermal camera calibration
- Solution: Increase thermal branch weight in fusion

**Issue: BEV projection incorrect**
- Solution: Verify camera intrinsics/extrinsics
- Solution: Check ground plane assumption
- Solution: Validate with known landmarks

**Issue: ROS 2 nodes not communicating**
- Solution: Check topic names (`ros2 topic list`)
- Solution: Verify QoS settings (reliability, durability)
- Solution: Check network configuration (if multi-machine)

### **Debugging Tools**

**PyTorch Profiler:**
```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    output = model(input_tensor)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

**TensorBoard Profiler:**
```bash
tensorboard --logdir=runs/
```

**RViz2 Visualization:**
```bash
rviz2 -d config/tmas_rviz.rviz
```

---

## Owner & Contact

**Project Lead:**
- Name: Krzysztof Romanowski
- GitHub: [ChrisRPL](https://github.com/ChrisRPL)
- Email: shepard128@gmail.com

**Repository:**
- GitHub: https://github.com/ChrisRPL/TMAS-Traversability-Mine-Analysis-System

**Related Projects:**
- HiMAC-JEPA: https://github.com/ChrisRPL/HiMAC-JEPA

---

## Quick Reference

### **Key Commands**

```bash
# Setup
pip install -e .
pre-commit install

# Development
pytest tests/                           # Run all tests
pytest tests/unit/ -v                   # Run unit tests verbose
pytest tests/ --cov=src/tmas            # Test with coverage
ruff check src/                         # Lint code
black src/                              # Format code
mypy src/                               # Type check

# Training
python scripts/training/train_terrain_segmentation.py
python scripts/training/train_mine_detection.py
python scripts/training/train_obstacle_detection.py

# Evaluation
python scripts/evaluation/eval_terrain_segmentation.py
python scripts/evaluation/eval_mine_detection.py
python scripts/evaluation/optimize_threshold.py

# Deployment
python scripts/deployment/export_to_onnx.py
python scripts/deployment/convert_to_tensorrt.py

# ROS 2
ros2 launch tmas_ros2 tmas_full.launch.py
ros2 bag record -a
ros2 topic echo /tmas/alerts
```

### **Important Files**

- **IMPLEMENTATION_PLAN.md** - 200+ atomic tasks, follow sequentially
- **SPEC.md** - Technical specification (gitignored, internal use)
- **README.md** - Public documentation
- **configs/default.yaml** - System configuration
- **pyproject.toml** - Package configuration
- **requirements.txt** - Python dependencies

### **Next Steps**

1. Start with Phase 1 (Week 1-2): Project setup
2. Follow IMPLEMENTATION_PLAN.md tasks sequentially
3. Commit atomically after each task
4. Verify each task before proceeding
5. Track progress with TodoWrite tool if needed

---

**Remember: This is a SAFETY-CRITICAL system. Every decision should prioritize recall, reliability, and operator safety.**
