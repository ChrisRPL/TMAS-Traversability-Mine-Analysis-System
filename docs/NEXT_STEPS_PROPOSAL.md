# TMAS Development: Next Steps Proposal

**Date**: February 23, 2026
**Current Phase**: Phase 5 Complete ✓
**Next Phase**: Phase 6 - Obstacle Detection

---

## Current Implementation Status

### ✓ Completed Phases

**Phase 1: Project Setup & Infrastructure**
- Complete project structure
- Development environment configured
- Git repository initialized

**Phase 2-3: Data Acquisition & Synthetic Generation**
- Synthetic mine data pipeline (Blender-based)
- RELLIS-3D terrain segmentation dataset integration
- Thermal dataset loaders (FLIR-compatible)
- Multi-modal augmentation pipeline

**Phase 4: Terrain Segmentation Model**
- EfficientViT-L2 backbone implementation
- Multi-scale segmentation head
- Combined CE + Dice loss
- Training script with W&B integration
- Target: mIoU > 75% on RELLIS-3D

**Phase 5: Mine Detection Model** ✓ JUST COMPLETED
- ResNet-18 thermal backbone (291 lines)
- Cross-attention RGB-Thermal fusion (426 lines)
- RT-DETR detection head (458 lines)
- Complete mine detector with uncertainty (363 lines)
- Evidential uncertainty estimation (404 lines)
- Detection loss functions (546 lines)
- Training script + config (550 lines)
- **Total: 3,038 lines of production code**

### 📊 Implementation Metrics

| Category | Count |
|----------|-------|
| Python modules | 28 files |
| Training scripts | 12 files |
| Git commits (Phase 5) | 8 atomic commits |
| Total code (Phase 5) | 3,038 lines |
| Components ready for training | 2/3 (Terrain ✓, Mine ✓, Obstacle ⏳) |

---

## Analysis of Remaining Work

### Critical Path Analysis

Based on SPEC.md and IMPLEMENTATION_PLAN.md, the remaining critical phases are:

1. **Phase 6: Obstacle Detection** (10 steps) ← NEXT
2. **Phase 7: BEV Transformation & Fusion** (6 steps)
3. **Phase 8: System Integration** (8 steps)
4. **Phase 9: Model Optimization & Export** (5 steps)
5. **Phase 10: ROS 2 Integration** (9 steps)

### Why Phase 6 (Obstacle Detection) is Next

**Rationale**:
1. **Completes the 3-module architecture**: Terrain ✓, Mine ✓, Obstacle ⏳
2. **Enables full pipeline testing**: All detection modules ready for integration
3. **Safety-critical requirements**: Person/vehicle recall > 99% (matches mine detection priority)
4. **Builds on existing work**: Can reuse RT-DETR architecture from mine detection
5. **Spec alignment**: Section 3.4 defines complete obstacle detection requirements

---

## Proposed Next 3 Steps

### 🎯 Step 1: Implement YOLOv8-Based Obstacle Detection Model

**Objective**: Create real-time obstacle detector for 20+ classes with >99% recall for persons/vehicles

**Deliverables**:
1. `src/tmas/detection/obstacle_detector.py` - YOLOv8-L detection model
2. `src/tmas/data/obstacle_dataset.py` - COCO + custom obstacle dataset loader
3. `configs/models/obstacle_detection.yaml` - Training configuration

**Technical Details**:

**Why YOLOv8 instead of RT-DETR for obstacles?**
- **Speed**: YOLOv8-L is faster (25-30 FPS vs 20 FPS for RT-DETR)
- **Anchor-based**: Better for large objects (vehicles, persons)
- **Proven performance**: COCO-pretrained for person/vehicle classes
- **Complementary**: RT-DETR for small objects (mines), YOLO for large (obstacles)

**Architecture**:
```
Obstacle Detection Module (YOLOv8-L):
- Backbone: CSPDarknet53
- Neck: PANet with multi-scale fusion
- Head: Anchor-free detection head
- Classes: 20+ obstacle types
- Input: RGB only (640×640)
- Output: Bounding boxes + confidence + class
```

**Dataset Strategy**:
1. **Base**: COCO dataset (person, car, truck, bicycle, motorcycle, bus)
2. **Augment**: Open Images (debris, barriers, rocks)
3. **Synthetic**: Blender-generated military obstacles (wreckage, craters, barriers)
4. **Custom classes**:
   - Military vehicles (tank, APC, humvee)
   - Fallen trees
   - Large rocks/boulders (>30cm)
   - Craters/holes
   - Debris/wreckage
   - Barricades

**Class Hierarchy** (20 classes):
```
Dynamic (Critical - Priority 1):
  1. Person
  2. Car/truck
  3. Military vehicle
  4. Motorcycle/bicycle
  5. Bus
  6. Large animal

Static (High - Priority 2):
  7. Fallen tree
  8. Large rock/boulder
  9. Crater/hole
  10. Barrier/barricade

Static (Medium - Priority 3):
  11. Vehicle wreckage
  12. Debris
  13. Metal fragments
  14. Concrete blocks
  15. Sandbags
  16. Wire coils
  17. Tires
  18. Containers
  19. Poles/posts
  20. Unknown obstacle
```

**Implementation Files**:

**File 1**: `src/tmas/detection/obstacle_detector.py` (~400 lines)
- YOLOv8ObstacleDetector class
- Load pretrained COCO weights
- Adapt to 20 obstacle classes
- Confidence filtering (0.5 for dynamic, 0.4 for static)
- NMS with class-aware thresholds

**File 2**: `src/tmas/data/obstacle_dataset.py` (~350 lines)
- COCOObstacleDataset (COCO + custom annotations)
- ObstacleDataModule (train/val/test loaders)
- Augmentation pipeline (flips, crops, color jitter)
- Class remapping from COCO to TMAS classes

**File 3**: `configs/models/obstacle_detection.yaml`
- Hyperparameters: lr=1e-3, batch=16, epochs=50
- Class weights (dynamic=5.0, static=1.0)
- NMS thresholds, confidence thresholds
- Data paths, augmentation settings

**Verification**:
- Model loads COCO pretrained weights
- Forward pass on test images works
- Output format: {boxes: [N, 4], scores: [N], labels: [N]}
- Inference speed: >25 FPS on RTX 3090

**Estimated Time**: 4-6 hours implementation + testing

---

### 🎯 Step 2: Implement Monocular Depth Estimation for Distance Calculation

**Objective**: Add depth estimation to compute obstacle distances (critical for TTC and safety zones)

**Deliverables**:
1. `src/tmas/depth/monocular_depth.py` - Depth estimation module
2. `src/tmas/depth/depth_calibration.py` - Metric depth calibration
3. Integration with obstacle detector for distance output

**Technical Details**:

**Why Monocular Depth?**
- **No stereo required**: Single RGB camera (cost-effective)
- **Deployment flexibility**: Works with existing camera setup
- **Recent advances**: Models like ZoeDepth achieve ±5% accuracy
- **SPEC requirement**: ±0.5m accuracy up to 20m

**Architecture**: **ZoeDepth** (chosen over MiDaS/DPT)
- **Advantages**:
  - Metric depth (not just relative)
  - Better zero-shot generalization
  - Indoor/outdoor robustness
  - Lighter than DPT (faster inference)

**Pipeline**:
```
RGB Image (640×480)
    ↓
ZoeDepth Encoder (Swin-Large)
    ↓
Multi-scale features
    ↓
Decoder with metric bins
    ↓
Depth Map (640×480) - metric depth [0-50m]
    ↓
Calibration (camera intrinsics + extrinsics)
    ↓
Real-world distance (meters)
```

**Integration with Obstacle Detector**:
```python
# Detection + Depth Pipeline
detections = obstacle_detector(rgb)  # Bboxes, classes, scores
depth_map = depth_estimator(rgb)     # H×W depth map

for bbox in detections:
    x1, y1, x2, y2 = bbox
    # Sample depth in bbox region
    roi_depth = depth_map[y1:y2, x1:x2]
    median_depth = torch.median(roi_depth)  # Robust to outliers

    # Apply calibration
    distance = calibrate_depth(median_depth, camera_params)

    # Assign to detection
    detections['distance'] = distance
```

**Calibration Strategy**:
1. **Camera intrinsics**: Focal length, principal point
2. **Ground truth**: LiDAR or known distances for validation
3. **Scale factor**: Learned from calibration dataset
4. **Per-zone calibration**: Different scales for near/mid/far zones

**Safety Zones** (from SPEC):
- **Critical**: 0-10m (TTC < 1s) → Immediate alert
- **Warning**: 10-20m (TTC < 3s) → Prepare to brake
- **Observation**: 20-50m → Monitor

**Implementation Files**:

**File 1**: `src/tmas/depth/monocular_depth.py` (~350 lines)
- MonocularDepthEstimator class
- Load ZoeDepth pretrained model
- Inference pipeline (RGB → depth map)
- Multi-scale support for different resolutions
- GPU/CPU compatibility

**File 2**: `src/tmas/depth/depth_calibration.py` (~250 lines)
- DepthCalibrator class
- Load camera intrinsics (focal length, cx, cy)
- Scale factor calibration
- Metric conversion (pixels → meters)
- Per-zone calibration coefficients

**File 3**: Update `obstacle_detector.py` integration (~100 lines added)
- Add depth_estimator parameter
- Compute distance for each detection
- Filter by distance (ignore >50m)
- Output format: {boxes, labels, scores, distances}

**Verification**:
- Depth maps generated for test images
- Distance accuracy: ±0.5m for 5-20m range (SPEC target)
- Inference speed: <20ms per frame (combined with detection)
- Integration: Obstacle detector outputs include distance

**Estimated Time**: 5-7 hours implementation + calibration testing

---

### 🎯 Step 3: Implement Time-to-Collision (TTC) and Trajectory Prediction

**Objective**: Enable dynamic obstacle tracking with TTC estimation for collision avoidance

**Deliverables**:
1. `src/tmas/tracking/byte_tracker.py` - ByteTrack for multi-object tracking
2. `src/tmas/tracking/trajectory_prediction.py` - Kalman filter + trajectory extrapolation
3. `src/tmas/detection/ttc.py` - Time-to-collision calculation
4. Integration with obstacle detector for complete pipeline

**Technical Details**:

**Why ByteTrack?**
- **No appearance model**: Faster, lighter (critical for real-time)
- **Low/high confidence tracking**: Recovers temporarily occluded objects
- **SOTA performance**: Best on MOT benchmarks
- **Simple**: IoU-based matching (no ReID network needed)

**TTC Calculation Formula**:
```
TTC = distance / relative_velocity

Where:
- distance: from depth estimation (meters)
- relative_velocity: obstacle_velocity - ego_velocity (m/s)
- ego_velocity: from vehicle odometry/GPS
```

**Safety Logic**:
```python
if TTC < 1.0 and distance < 10.0:
    alert_level = "CRITICAL"
    action = "EMERGENCY_BRAKE"
elif TTC < 3.0 and distance < 20.0:
    alert_level = "WARNING"
    action = "PREPARE_TO_BRAKE"
elif distance < 50.0:
    alert_level = "OBSERVATION"
    action = "MONITOR"
else:
    alert_level = "NONE"
    action = "CONTINUE"
```

**Trajectory Prediction**:
- **Model**: Constant velocity Kalman filter
- **Prediction horizon**: 1-3 seconds
- **State vector**: [x, y, vx, vy, ax, ay]
- **Update frequency**: 20 Hz (matches FPS)

**Pipeline**:
```
Frame t:
  RGB → Obstacle Detector → Detections [boxes, classes, scores]
  RGB → Depth Estimator → Distances [N]

  Detections + Distances → ByteTrack → Tracked objects [ID, box, class, distance]

  Tracked objects → Trajectory Predictor:
    - Kalman filter update
    - Velocity estimation (dx/dt, dy/dt)
    - Position extrapolation (t+1, t+2, t+3 seconds)

  Tracked objects + Vehicle ego-motion → TTC Calculator:
    - Relative velocity = obstacle_v - ego_v
    - TTC = distance / relative_velocity
    - Safety zone classification
    - Alert generation
```

**Implementation Files**:

**File 1**: `src/tmas/tracking/byte_tracker.py` (~400 lines)
- ByteTracker class
- Low/high confidence detection matching
- Track lifecycle management (new, active, lost, removed)
- IoU-based matching with Hungarian algorithm
- Output: Tracked objects with persistent IDs

**File 2**: `src/tmas/tracking/trajectory_prediction.py` (~350 lines)
- TrajectoryPredictor class
- Kalman filter implementation (position + velocity state)
- Velocity estimation from tracked positions
- Trajectory extrapolation (1-3 seconds)
- Collision zone prediction (intersects vehicle path?)

**File 3**: `src/tmas/detection/ttc.py` (~300 lines)
- TTCCalculator class
- Distance + velocity → TTC computation
- Ego-motion compensation (from vehicle odometry)
- Safety zone classification (critical/warning/observation)
- Alert message generation

**File 4**: Update `obstacle_detector.py` for full pipeline (~150 lines added)
- Integrate ByteTrack
- Integrate trajectory prediction
- Integrate TTC calculation
- Output format: Complete tracked obstacles with TTC

**Output Format**:
```python
{
    'track_id': int,
    'bbox': [x1, y1, x2, y2],
    'class': int,
    'class_name': str,
    'score': float,
    'distance': float,  # meters
    'velocity': [vx, vy],  # m/s
    'trajectory': [[x1,y1], [x2,y2], ...],  # predicted positions
    'ttc': float,  # seconds (or None if static)
    'alert_level': str,  # CRITICAL/WARNING/OBSERVATION/NONE
    'action': str  # EMERGENCY_BRAKE/PREPARE/MONITOR/CONTINUE
}
```

**Verification**:
- Tracking maintains ID across frames (90%+ MOTA on MOT test)
- Velocity estimation accuracy: ±1 m/s
- TTC accuracy: ±0.3s (SPEC target)
- Alert latency: <50ms (1-2 frames)
- Full pipeline runs at >20 FPS

**Estimated Time**: 6-8 hours implementation + testing

---

## Summary of Proposed Next 3 Steps

| Step | Component | Deliverable | Lines | Time | Priority |
|------|-----------|-------------|-------|------|----------|
| 1 | Obstacle Detection | YOLOv8-L model + dataset | ~750 | 4-6h | CRITICAL |
| 2 | Depth Estimation | ZoeDepth + calibration | ~700 | 5-7h | HIGH |
| 3 | TTC + Tracking | ByteTrack + Kalman + TTC | ~1200 | 6-8h | HIGH |

**Total Estimated**:
- **Code**: ~2,650 lines
- **Time**: 15-21 hours
- **Components**: 3 major modules
- **Commits**: 6-8 atomic commits

---

## Technical Justifications

### Why YOLOv8 for Obstacles (vs RT-DETR)?

| Criteria | YOLOv8-L | RT-DETR-L |
|----------|----------|-----------|
| Speed (RTX 3090) | 25-30 FPS | 20 FPS |
| Speed (Jetson Orin) | 15-18 FPS | 10-12 FPS |
| Large object detection | Excellent | Good |
| Small object detection | Good | Excellent |
| COCO pretraining | ✓ Complete | ✓ Complete |
| NMS-free | ✗ | ✓ |
| Latency (critical for TTC) | Lower | Higher |

**Decision**: Use YOLOv8 for obstacles (speed + large objects) and RT-DETR for mines (small objects).

### Why ZoeDepth for Depth (vs MiDaS/DPT)?

| Criteria | ZoeDepth | MiDaS v3.1 | DPT-Large |
|----------|----------|------------|-----------|
| Metric depth | ✓ Yes | ✗ Relative | ✗ Relative |
| Zero-shot accuracy | ±5% | ±8% | ±6% |
| Model size | 400MB | 343MB | 1.3GB |
| Inference speed (640×480) | 25ms | 20ms | 50ms |
| Outdoor robustness | Excellent | Good | Excellent |

**Decision**: ZoeDepth for metric depth + speed balance.

### Why ByteTrack for Tracking (vs DeepSORT/FairMOT)?

| Criteria | ByteTrack | DeepSORT | FairMOT |
|----------|-----------|----------|---------|
| Speed | Fast | Medium | Slow |
| ReID network needed | ✗ No | ✓ Yes | ✓ Yes |
| MOTA (MOT17) | 80.3% | 75.7% | 73.7% |
| Low-confidence recovery | ✓ Yes | ✗ No | ✗ No |
| Real-time capable | ✓ Yes | ~ Maybe | ✗ No |

**Decision**: ByteTrack for speed + no ReID overhead.

---

## Risk Assessment & Mitigation

### Risk 1: Depth Estimation Accuracy
**Risk**: Monocular depth may not meet ±0.5m target at 20m
**Mitigation**:
- Use ZoeDepth's metric depth (better than relative)
- Calibrate per-zone (near/mid/far) with ground truth
- Fallback: Stereo camera or LiDAR fusion (Phase 7)
**Impact**: Medium (affects TTC accuracy)

### Risk 2: Real-Time Performance on Jetson
**Risk**: YOLOv8 + ZoeDepth + ByteTrack may not hit 20 FPS
**Mitigation**:
- Use TensorRT optimization (Phase 9)
- Resolution adjustment (640×480 instead of 720p)
- Model pruning/quantization if needed
**Impact**: High (core requirement)

### Risk 3: Obstacle Dataset Completeness
**Risk**: Limited military-specific obstacle data
**Mitigation**:
- Use COCO for common classes (90% coverage)
- Generate synthetic military obstacles in Blender
- Augment with domain randomization
**Impact**: Low (COCO covers critical classes)

---

## Success Criteria for Next 3 Steps

### Step 1 Success (Obstacle Detection):
✓ Model loads and runs on test images
✓ Person recall > 99% on COCO val
✓ Vehicle recall > 99% on COCO val
✓ Inference speed > 25 FPS on RTX 3090
✓ 20 obstacle classes supported

### Step 2 Success (Depth Estimation):
✓ Depth maps generated for all test images
✓ Distance accuracy ±0.5m for 5-20m range
✓ Integration with obstacle detector works
✓ Inference added overhead < 20ms
✓ Safety zones classified correctly

### Step 3 Success (TTC + Tracking):
✓ Object tracking maintains ID across frames (MOTA > 80%)
✓ TTC calculated with ±0.3s accuracy
✓ Alert system triggers at correct thresholds
✓ Full pipeline runs at > 20 FPS
✓ Emergency brake logic tested on video sequences

---

## Alignment with SPEC & IMPLEMENTATION_PLAN

### SPEC Alignment:
- ✓ Section 3.4: Obstacle detection module architecture
- ✓ Table: 20+ obstacle classes defined
- ✓ TTC accuracy target: ±0.3s
- ✓ Distance accuracy target: ±0.5m @ 20m
- ✓ Latency target: <50ms for sudden obstacles
- ✓ Recall targets: >99% persons/vehicles, >95% static obstacles

### IMPLEMENTATION_PLAN Alignment:
- ✓ Phase 6, Steps 6.2, 6.3, 6.4, 6.5 covered
- ✓ Obstacle detection model (Step 6.2)
- ✓ Monocular depth (Step 6.3)
- ✓ Trajectory prediction (Step 6.4)
- ✓ TTC estimation (Step 6.5)

**Remaining from Phase 6**: Steps 6.1, 6.6-6.10 (dataset prep, sudden obstacle, training, evaluation)

---

## Conclusion

The proposed next 3 steps will:
1. **Complete the detection triad**: Terrain ✓, Mine ✓, Obstacle ✓
2. **Enable safety-critical features**: TTC, collision avoidance, emergency brake
3. **Build on existing architecture**: Reuse patterns from mine detection
4. **Meet SPEC requirements**: All targets from Section 3.4
5. **Prepare for integration**: Phase 7 (BEV fusion) requires all 3 modules ready

**Recommendation**: Proceed with Step 1 (Obstacle Detection) immediately.

---

**Next Actions**:
1. User approval of proposed steps
2. Begin Step 1 implementation
3. Ship in atomic commits (as always)
4. Verify against SPEC requirements
5. Document progress

**Estimated Completion**: 2-3 development sessions (15-21 hours total)
