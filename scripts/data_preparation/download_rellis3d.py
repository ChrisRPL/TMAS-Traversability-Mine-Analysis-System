"""Download and setup RELLIS-3D dataset for terrain segmentation.

RELLIS-3D is an off-road dataset with RGB images and 14 terrain class annotations.
Source: https://github.com/unmannedlab/RELLIS-3D
"""

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def download_rellis3d(output_dir: str = "data/raw/rellis3d", verify: bool = True) -> Dict[str, Any]:
    """Download RELLIS-3D dataset.

    Args:
        output_dir: Directory to save dataset
        verify: Verify dataset after download

    Returns:
        Dictionary with download statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading RELLIS-3D dataset to {output_path}")

    # RELLIS-3D repository URL
    repo_url = "https://github.com/unmannedlab/RELLIS-3D.git"

    # Clone repository (contains download links and metadata)
    repo_dir = output_path / "RELLIS-3D-repo"
    if not repo_dir.exists():
        logger.info("Cloning RELLIS-3D repository...")
        try:
            subprocess.run(
                ["git", "clone", repo_url, str(repo_dir)],
                check=True,
                capture_output=True,
            )
            logger.info("Repository cloned successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e}")
            return {"status": "error", "message": str(e)}
    else:
        logger.info("Repository already exists")

    # Note: Actual dataset download requires following instructions from the repository
    # The full dataset is ~160GB and hosted externally
    logger.info("=" * 80)
    logger.info("RELLIS-3D Dataset Download Instructions:")
    logger.info("=" * 80)
    logger.info("1. Visit: https://github.com/unmannedlab/RELLIS-3D")
    logger.info("2. Follow the download instructions in the README")
    logger.info("3. The dataset is hosted on: https://utdallas.box.com/v/RELLIS-3D")
    logger.info("4. Download size: ~160GB")
    logger.info("5. Extract to: data/raw/rellis3d/")
    logger.info("=" * 80)
    logger.info("Expected structure:")
    logger.info("data/raw/rellis3d/")
    logger.info("├── Rellis-3D/")
    logger.info("│   ├── 00000/")
    logger.info("│   │   ├── pylon_camera_node/")
    logger.info("│   │   └── pylon_camera_node_label_id/")
    logger.info("│   ├── 00001/")
    logger.info("│   └── ...")
    logger.info("=" * 80)

    stats = {
        "status": "instructions_provided",
        "dataset_name": "RELLIS-3D",
        "download_url": "https://utdallas.box.com/v/RELLIS-3D",
        "expected_sequences": 5,
        "expected_frames": 13556,
        "num_classes": 20,  # Total classes (14 used for terrain)
        "output_dir": str(output_path),
    }

    # Save download info
    info_file = output_path / "download_info.json"
    with open(info_file, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Download info saved to: {info_file}")

    return stats


def verify_rellis3d(data_dir: str = "data/raw/rellis3d") -> bool:
    """Verify RELLIS-3D dataset integrity.

    Args:
        data_dir: Dataset directory

    Returns:
        True if dataset is valid
    """
    data_path = Path(data_dir)
    logger.info(f"Verifying RELLIS-3D dataset in {data_path}")

    # Check for main dataset directory
    rellis_dir = data_path / "Rellis-3D"
    if not rellis_dir.exists():
        logger.warning(f"Dataset directory not found: {rellis_dir}")
        logger.warning("Please download the dataset manually from:")
        logger.warning("https://utdallas.box.com/v/RELLIS-3D")
        return False

    # Check for sequences
    sequences = list(rellis_dir.glob("0*"))
    if not sequences:
        logger.warning("No sequences found in dataset")
        return False

    logger.info(f"Found {len(sequences)} sequences")

    # Count total frames
    total_frames = 0
    for seq in sequences:
        rgb_dir = seq / "pylon_camera_node"
        if rgb_dir.exists():
            frames = list(rgb_dir.glob("*.jpg"))
            total_frames += len(frames)
            logger.info(f"Sequence {seq.name}: {len(frames)} frames")

    logger.info(f"Total frames: {total_frames}")

    if total_frames < 10000:
        logger.warning(f"Expected ~13556 frames, found {total_frames}")
        return False

    logger.info("Dataset verification passed!")
    return True


def main():
    """Main function for dataset download."""
    parser = argparse.ArgumentParser(description="Download RELLIS-3D dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/rellis3d",
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify dataset after download",
    )

    args = parser.parse_args()

    # Download dataset (provides instructions)
    stats = download_rellis3d(args.output_dir, args.verify)

    # Verify if requested
    if args.verify:
        verify_rellis3d(args.output_dir)


if __name__ == "__main__":
    main()
