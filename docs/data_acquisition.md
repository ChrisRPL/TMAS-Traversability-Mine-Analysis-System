# Data Acquisition Guide

This document provides instructions for acquiring all datasets required for TMAS development.

## Overview

TMAS requires multiple datasets for training its three core modules:

1. **Terrain Segmentation**: RELLIS-3D, TartanDrive
2. **Mine Detection**: GICHD Mine Database, public mine datasets, synthetic data
3. **Obstacle Detection**: COCO, Open Images, synthetic data
4. **Thermal Imaging**: FLIR Thermal, KAIST Multispectral

---

## Dataset 1: RELLIS-3D (Terrain Segmentation)

**Purpose**: Off-road terrain segmentation with 14 terrain classes

**Size**: ~160GB
**Frames**: 13,556 RGB images with annotations
**Classes**: 20 total (14 used for terrain types)

### Download Instructions

1. Visit the official repository:
   ```
   https://github.com/unmannedlab/RELLIS-3D
   ```

2. Access the dataset download link:
   ```
   https://utdallas.box.com/v/RELLIS-3D
   ```

3. Download all sequences (00000 - 00004)

4. Extract to:
   ```
   data/raw/rellis3d/Rellis-3D/
   ```

5. Verify structure:
   ```
   data/raw/rellis3d/
   └── Rellis-3D/
       ├── 00000/
       │   ├── pylon_camera_node/          # RGB images
       │   └── pylon_camera_node_label_id/  # Annotations
       ├── 00001/
       ├── 00002/
       ├── 00003/
       └── 00004/
   ```

### Automated Download

```bash
python scripts/data_preparation/download_rellis3d.py --output-dir data/raw/rellis3d
python scripts/data_preparation/download_rellis3d.py --verify  # Check integrity
```

### Terrain Classes

The 14 terrain classes used in TMAS:

1. Void
2. Dirt
3. Grass
4. Tree
5. Pole
6. Water
7. Sky
8. Vehicle
9. Object
10. Asphalt
11. Building
12. Log
13. Person
14. Fence

---

## Dataset 2: TartanDrive (Depth Estimation)

**Purpose**: Depth estimation for obstacle distance calculation

**Size**: ~100GB
**Frames**: 200,000+ RGB + stereo images

### Download Instructions

1. Visit the official website:
   ```
   https://github.com/castacks/tartan_drive
   ```

2. Follow download instructions in the repository

3. Extract to:
   ```
   data/raw/tartandrive/
   ```

---

## Dataset 3: GICHD Mine Database (Mine Detection)

**Purpose**: Real mine images for detection training

**Size**: ~5GB (estimated)
**Images**: 10,000+ annotated mine images

### Access Request

**IMPORTANT**: This dataset requires formal access request.

1. Visit GICHD website:
   ```
   https://www.gichd.org/
   ```

2. Submit data access request for mine action database

3. Expected approval time: 2-4 weeks

4. **Alternative**: Proceed with synthetic data and public datasets while waiting

### Status

- [ ] Access request submitted
- [ ] Approval received
- [ ] Dataset downloaded

---

## Dataset 4: Public Mine Detection Datasets

**Purpose**: Supplementary mine detection training data

**Target**: 1,000+ annotated images

### Search Sources

1. **Kaggle**:
   - Search: "landmine", "mine detection", "UXO"
   - https://www.kaggle.com/datasets

2. **Papers with Code**:
   - https://paperswithcode.com/
   - Search for mine detection papers with datasets

3. **IEEE DataPort**:
   - https://ieee-dataport.org/
   - Search for humanitarian demining datasets

4. **Academic Repositories**:
   - Check university research groups working on mine detection
   - Contact authors of recent papers for dataset access

### Found Datasets

_To be updated as datasets are identified_

- [ ] Dataset 1: [Name] - [Source] - [Size]
- [ ] Dataset 2: [Name] - [Source] - [Size]

---

## Dataset 5: Thermal Imaging Datasets

**Purpose**: Training thermal branch for mine detection

**Target**: 5,000+ thermal images

### Recommended Datasets

1. **FLIR Thermal Dataset (Free)**:
   - Source: https://www.flir.com/oem/adas/adas-dataset-form/
   - Size: 10,000+ annotated thermal images
   - Good for general thermal object detection

2. **KAIST Multispectral Dataset**:
   - Source: https://soonminhwang.github.io/rgbt-ped-detection/
   - RGB-Thermal paired pedestrian detection
   - Size: 95k color-thermal pairs

3. **CVC-14 Thermal Dataset**:
   - Source: http://adas.cvc.uab.es/elektra/enigma-portfolio/cvc-14-visible-fir-day-night-pedestrian-sequence-dataset/
   - Thermal pedestrian detection

### Download Priority

1. FLIR Thermal Dataset (free, easy to obtain)
2. KAIST Multispectral (RGB-Thermal pairs useful for fusion)
3. CVC-14 (if needed for additional data)

---

## Dataset 6: COCO & Open Images (Obstacle Detection)

**Purpose**: General obstacle detection (persons, vehicles, animals)

### COCO Dataset

```bash
# Download COCO 2017
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Extract to data/raw/coco/
unzip train2017.zip -d data/raw/coco/
unzip val2017.zip -d data/raw/coco/
unzip annotations_trainval2017.zip -d data/raw/coco/
```

### Open Images

Download specific classes:
- Person
- Vehicle (car, truck, military vehicle)
- Animal (large)
- Debris
- Barrier

Use OIDv4 Toolkit:
```bash
pip install oidv4-toolkit
oidv4 downloader --classes Person Vehicle Animal --type_csv train
```

---

## Synthetic Data Generation

**Purpose**: Primary training data for mine detection (100,000 images)

See Phase 3 of IMPLEMENTATION_PLAN.md for synthetic data generation pipeline.

Synthetic data generation will create:
- 100,000 RGB images with mines
- 100,000 thermal images
- COCO format annotations
- Multiple terrain types and burial depths

---

## Dataset Registry

After downloading datasets, update the dataset registry:

```bash
# Location: data/registry.json
{
  "datasets": {
    "rellis3d": {
      "path": "data/raw/rellis3d",
      "type": "terrain_segmentation",
      "num_samples": 13556,
      "status": "downloaded",
      "date_acquired": "2026-02-22"
    },
    "tartandrive": {
      "path": "data/raw/tartandrive",
      "type": "depth_estimation",
      "status": "pending"
    }
  }
}
```

---

## Verification Checklist

Before proceeding to model training:

- [ ] RELLIS-3D downloaded and verified (13k+ frames)
- [ ] TartanDrive downloaded (200k+ frames) OR using synthetic depth
- [ ] At least 1 public mine dataset acquired (1k+ images)
- [ ] At least 1 thermal dataset acquired (5k+ images)
- [ ] COCO dataset downloaded for obstacle detection
- [ ] Dataset registry updated with all datasets
- [ ] Data loaders implemented and tested

---

## Storage Requirements

Total storage needed:

- RELLIS-3D: ~160GB
- TartanDrive: ~100GB
- Mine datasets: ~10GB
- Thermal datasets: ~20GB
- COCO: ~25GB
- Synthetic data (generated): ~100GB

**Total: ~415GB**

Ensure sufficient disk space before downloading.

---

## Contact & Support

For dataset access issues:
- RELLIS-3D: Check GitHub issues
- GICHD: Contact GICHD directly
- Academic datasets: Email paper authors

For questions about data preparation, see IMPLEMENTATION_PLAN.md or CLAUDE.md.
