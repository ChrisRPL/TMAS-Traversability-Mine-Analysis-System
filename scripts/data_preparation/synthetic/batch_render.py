"""Batch rendering pipeline for generating synthetic mine detection dataset.

This module orchestrates the generation of thousands of synthetic images
by coordinating terrain generation, mine placement, lighting, and rendering
in a multi-threaded pipeline.
"""

import subprocess
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Optional
import json
import time
import random
from datetime import datetime


class BatchRenderer:
    """Multi-threaded batch rendering system."""

    def __init__(
        self,
        blender_path: str = "blender",
        output_dir: str = "data/synthetic/mines",
        num_workers: int = 4
    ):
        """Initialize batch renderer.

        Args:
            blender_path: Path to Blender executable
            output_dir: Output directory for rendered images
            num_workers: Number of parallel rendering processes
        """
        self.blender_path = blender_path
        self.output_dir = Path(output_dir)
        self.num_workers = num_workers

        # Create output directories
        self.rgb_dir = self.output_dir / "rgb"
        self.thermal_dir = self.output_dir / "thermal"
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.thermal_dir.mkdir(parents=True, exist_ok=True)

        self.progress_file = self.output_dir / "progress.json"
        self.annotation_file = self.output_dir / "annotations.json"

    def generate_scene_config(self, scene_id: int) -> Dict:
        """Generate random configuration for a scene.

        Args:
            scene_id: Scene identifier

        Returns:
            Configuration dictionary
        """
        terrain_types = ["desert", "grassland", "rocky", "forest"]
        times_of_day = ["dawn", "day", "dusk", "night"]
        weather_types = ["clear", "overcast", "fog", "rain"]

        config = {
            "scene_id": scene_id,
            "terrain_type": random.choice(terrain_types),
            "time_of_day": random.choice(times_of_day),
            "weather": random.choice(weather_types),
            "num_mines": random.randint(2, 8),
            "camera_height": random.uniform(3.0, 8.0),
            "camera_distance": random.uniform(10.0, 20.0),
            "camera_angle": random.uniform(-15.0, 15.0),
            "seed": scene_id * 1000 + random.randint(0, 999)
        }

        return config

    def create_render_script(self, config: Dict, output_prefix: str) -> str:
        """Create Blender Python script for rendering.

        Args:
            config: Scene configuration
            output_prefix: Output file prefix

        Returns:
            Python script as string
        """
        script = f'''
import bpy
import sys
import random
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))

from terrain_generator import TerrainGenerator
from mine_placement import MinePlacement
from lighting_weather import setup_random_lighting
from thermal_simulation import setup_thermal_rendering
from auto_annotate import COCOAnnotator

# Set random seed
random.seed({config['seed']})

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

print("Generating terrain...")
terrain_gen = TerrainGenerator(size=20.0)
terrain = terrain_gen.generate(
    terrain_type="{config['terrain_type']}",
    add_scatter=True
)

print("Placing mines...")
mine_placer = MinePlacement(terrain)
mines = mine_placer.place_random_mines(
    num_mines={config['num_mines']},
    burial_depth_range=(0.0, 0.15)
)
print(f"Placed {{len(mines)}} mines")

# Setup camera
print("Setting up camera...")
bpy.ops.object.camera_add(
    location=({config['camera_distance']}, 0, {config['camera_height']})
)
camera = bpy.context.object
camera.rotation_euler = (1.2, 0, 1.57)  # Point at terrain
bpy.context.scene.camera = camera

# Setup lighting
print("Setting up lighting...")
setup_random_lighting(
    weather_type="{config['weather']}",
    time="{config['time_of_day']}"
)

# Render RGB
print("Rendering RGB...")
scene = bpy.context.scene
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.filepath = "{output_prefix}_rgb.png"
bpy.ops.render.render(write_still=True)

# Generate annotations
print("Generating annotations...")
annotator = COCOAnnotator()
num_annotations = annotator.annotate_current_scene(
    image_filename=Path("{output_prefix}_rgb.png").name,
    thermal_filename=Path("{output_prefix}_thermal.png").name
)
annotator.save_annotations("{output_prefix}_annotations.json")
print(f"Created {{num_annotations}} annotations")

# Setup thermal rendering
print("Setting up thermal rendering...")
setup_thermal_rendering()

# Render thermal
print("Rendering thermal...")
scene.render.resolution_x = 640
scene.render.resolution_y = 512
scene.render.filepath = "{output_prefix}_thermal.png"
bpy.ops.render.render(write_still=True)

print(f"Scene {config['scene_id']} complete")
'''
        return script

    def render_scene(self, config: Dict) -> bool:
        """Render a single scene with Blender.

        Args:
            config: Scene configuration

        Returns:
            True if successful, False otherwise
        """
        scene_id = config["scene_id"]
        output_prefix = str(self.output_dir / f"scene_{scene_id:06d}")

        # Create temporary script file
        script_path = self.output_dir / f"render_script_{scene_id}.py"
        script_content = self.create_render_script(config, output_prefix)

        with open(script_path, 'w') as f:
            f.write(script_content)

        # Run Blender
        cmd = [
            self.blender_path,
            "--background",
            "--python", str(script_path)
        ]

        try:
            print(f"Rendering scene {scene_id}...")
            start_time = time.time()

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout per scene
            )

            elapsed = time.time() - start_time

            if result.returncode == 0:
                print(f"Scene {scene_id} completed in {elapsed:.1f}s")
                # Clean up script
                script_path.unlink()
                return True
            else:
                print(f"Scene {scene_id} failed:")
                print(result.stderr)
                return False

        except subprocess.TimeoutExpired:
            print(f"Scene {scene_id} timed out")
            return False
        except Exception as e:
            print(f"Scene {scene_id} error: {e}")
            return False

    def render_worker(self, scene_configs: List[Dict], worker_id: int):
        """Worker function for parallel rendering.

        Args:
            scene_configs: List of scene configurations to render
            worker_id: Worker identifier
        """
        print(f"Worker {worker_id} starting with {len(scene_configs)} scenes")

        for config in scene_configs:
            success = self.render_scene(config)

            # Update progress
            self.update_progress(config["scene_id"], success)

    def update_progress(self, scene_id: int, success: bool):
        """Update progress tracking file.

        Args:
            scene_id: Scene identifier
            success: Whether rendering succeeded
        """
        # Simple file-based progress tracking
        progress_line = f"{scene_id},{success},{datetime.now().isoformat()}\n"

        with open(self.progress_file, 'a') as f:
            f.write(progress_line)

    def merge_annotations(self):
        """Merge all individual annotation files into single COCO file."""
        print("Merging annotations...")

        merged = {
            "info": {
                "description": "TMAS Synthetic Mine Detection Dataset",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "TMAS Project",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }

        annotation_files = list(self.output_dir.glob("scene_*_annotations.json"))

        image_id = 1
        annotation_id = 1

        for ann_file in sorted(annotation_files):
            try:
                with open(ann_file) as f:
                    data = json.load(f)

                # Set categories from first file
                if not merged["categories"] and "categories" in data:
                    merged["categories"] = data["categories"]

                # Add images with new IDs
                for img in data.get("images", []):
                    img["id"] = image_id
                    merged["images"].append(img)
                    image_id += 1

                # Add annotations with new IDs
                for ann in data.get("annotations", []):
                    ann["id"] = annotation_id
                    ann["image_id"] = image_id - 1  # Link to last added image
                    merged["annotations"].append(ann)
                    annotation_id += 1

                # Clean up individual annotation file
                ann_file.unlink()

            except Exception as e:
                print(f"Error processing {ann_file}: {e}")

        # Save merged annotations
        with open(self.annotation_file, 'w') as f:
            json.dump(merged, f, indent=2)

        print(f"Merged {len(merged['images'])} images with "
              f"{len(merged['annotations'])} annotations")

    def generate_dataset(
        self,
        num_scenes: int = 100,
        resume: bool = False
    ) -> Dict:
        """Generate synthetic dataset.

        Args:
            num_scenes: Number of scenes to generate
            resume: Whether to resume from previous progress

        Returns:
            Statistics dictionary
        """
        print(f"Generating {num_scenes} synthetic scenes...")
        print(f"Output directory: {self.output_dir}")
        print(f"Workers: {self.num_workers}")

        # Check for existing progress
        start_scene = 0
        if resume and self.progress_file.exists():
            with open(self.progress_file) as f:
                completed = len(f.readlines())
            start_scene = completed
            print(f"Resuming from scene {start_scene}")
        else:
            # Clear progress file
            self.progress_file.unlink(missing_ok=True)

        # Generate scene configurations
        configs = [
            self.generate_scene_config(i)
            for i in range(start_scene, start_scene + num_scenes)
        ]

        # Split work among workers
        chunk_size = len(configs) // self.num_workers
        chunks = [
            configs[i:i + chunk_size]
            for i in range(0, len(configs), chunk_size)
        ]

        # Start workers
        start_time = time.time()

        if self.num_workers > 1:
            # Multi-threaded rendering
            processes = []
            for i, chunk in enumerate(chunks):
                if chunk:
                    p = mp.Process(target=self.render_worker, args=(chunk, i))
                    p.start()
                    processes.append(p)

            # Wait for completion
            for p in processes:
                p.join()
        else:
            # Single-threaded for debugging
            self.render_worker(configs, 0)

        elapsed = time.time() - start_time

        # Merge annotations
        self.merge_annotations()

        # Gather statistics
        stats = {
            "total_scenes": num_scenes,
            "total_time_seconds": elapsed,
            "avg_time_per_scene": elapsed / num_scenes if num_scenes > 0 else 0,
            "output_directory": str(self.output_dir)
        }

        print(f"\nDataset generation complete!")
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Average: {stats['avg_time_per_scene']:.1f} seconds per scene")

        return stats


def main():
    """Main entry point for batch rendering."""
    import argparse

    parser = argparse.ArgumentParser(description="Batch render synthetic mine dataset")
    parser.add_argument("--num-scenes", type=int, default=100,
                       help="Number of scenes to generate")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--output-dir", type=str, default="data/synthetic/mines",
                       help="Output directory")
    parser.add_argument("--blender", type=str, default="blender",
                       help="Path to Blender executable")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from previous progress")

    args = parser.parse_args()

    renderer = BatchRenderer(
        blender_path=args.blender,
        output_dir=args.output_dir,
        num_workers=args.workers
    )

    stats = renderer.generate_dataset(
        num_scenes=args.num_scenes,
        resume=args.resume
    )

    # Save statistics
    stats_file = Path(args.output_dir) / "generation_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Statistics saved to {stats_file}")


if __name__ == "__main__":
    main()
