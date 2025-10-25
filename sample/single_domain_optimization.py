#!/usr/bin/env python3
"""
Single-Domain Parameter Optimization for Cross-Domain Low-Light Enhancement

This script optimizes single-domain inference parameters for:
- Photography (Sony and Fuji separately)
- Microscopy
- Astronomy

Target: Best SSIM and PSNR while minimizing LPIPS and NIQE
Methods: Both Gaussian and PG guidance
"""

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch

class SingleDomainOptimizer:
    """Single-domain parameter optimization for cross-domain low-light enhancement."""
    
    def __init__(self, base_args: Dict[str, Any]):
        self.base_args = base_args
        self.results = []
        
        # Domain-specific parameter ranges for single-domain optimization (coarse: min, median, max)
        self.domain_configs = {
            "photography": {
                "sony": {
                    "s": 15871,
                    "sensor_name": "sony_a7s_ii",
                    "sensor_filter": "sony",
                    "kappa_range": [0.2, 0.5, 0.8],  # min, median, max
                    "sigma_r_range": [2.0, 4.0, 6.0],  # min, median, max
                    "num_steps_range": [15, 20, 25]  # min, median, max
                },
                "fuji": {
                    "s": 15871,
                    "sensor_name": "fuji_xt2",
                    "sensor_filter": "fuji",
                    "kappa_range": [0.2, 0.5, 0.8],  # min, median, max
                    "sigma_r_range": [2.0, 4.0, 6.0],  # min, median, max
                    "num_steps_range": [15, 20, 25]  # min, median, max
                }
            },
            "microscopy": {
                "default": {
                    "s": 65535,
                    "sensor_name": "hamamatsu_orca_flash4_v3",
                    "sensor_filter": None,
                    "kappa_range": [0.4, 1.0, 1.4],  # min, median, max
                    "sigma_r_range": [0.5, 1.5, 2.5],  # min, median, max
                    "num_steps_range": [20, 30, 40]  # min, median, max
                }
            },
            "astronomy": {
                "default": {
                    "s": 450,
                    "sensor_name": "hubble_wfc3",
                    "sensor_filter": None,
                    "kappa_range": [0.05, 0.2, 0.3],  # min, median, max
                    "sigma_r_range": [2.0, 5.0, 9.0],  # min, median, max
                    "num_steps_range": [25, 35, 45]  # min, median, max
                }
            }
        }
    
    def run_inference_single_tile(self, args: Dict[str, Any], tile_id: str, output_dir: str) -> Dict[str, Any]:
        """Run inference on a single tile with given parameters."""

        print(f"  🔹 Running inference on tile: {tile_id}")

        cmd = [
            "python", "sample/sample_noisy_pt_lle_PGguidance.py",
            "--model_path", args["model_path"],
            "--metadata_json", args["metadata_json"],
            "--noisy_dir", args["noisy_dir"],
            "--clean_dir", args["clean_dir"],
            "--output_dir", output_dir,
            "--domain", args["domain"],
            "--tile_ids", tile_id,  # Process specific tile
            "--use_sensor_calibration",
            "--sensor_name", args["sensor_name"],
            "--s", str(args["s"]),
            "--sigma_r", str(args["sigma_r"]),
            "--kappa", str(args["kappa"]),
            "--num_steps", str(args["num_steps"]),
            # Note: preserve_details, adaptive_strength, edge_aware are not supported by inference script
            "--compare_gaussian",
            "--run_methods", "noisy", "clean", "exposure_scaled", "gaussian_x0", "pg_x0"  # Only run single-domain methods for optimization
        ]

        # Add sensor filter if specified
        if args.get("sensor_filter"):
            cmd.extend(["--sensor_filter", args["sensor_filter"]])

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)  # 30 min timeout per tile
            print(f"    ✅ Tile {tile_id} completed successfully!")

            # Extract results from this tile run
            tile_results = self.extract_comprehensive_results(output_dir)
            return tile_results

        except subprocess.CalledProcessError as e:
            print(f"    ❌ Tile {tile_id} failed with exit code {e.returncode}")
            print(f"    Error: {e.stderr}")
            return {}
        except subprocess.TimeoutExpired as e:
            print(f"    ❌ Tile {tile_id} timed out after {e.timeout} seconds")
            return {}

    def select_random_tiles(self, args: Dict[str, Any], num_tiles: int = 1) -> List[str]:
        """Select random tiles from the filesystem that match the filter criteria and have clean pairs."""
        try:
            # Scan filesystem for available tiles
            noisy_dir = Path(args["noisy_dir"])
            clean_dir = Path(args["clean_dir"])
            available_tiles = []

            # Get all .pt files in the noisy directory
            if noisy_dir.exists():
                for pt_file in noisy_dir.glob("*.pt"):
                    tile_id = pt_file.stem  # Remove .pt extension

                    # Filter by sensor if specified
                    if args.get("sensor_filter"):
                        if args["sensor_filter"].lower() not in tile_id.lower():
                            continue

                    # Filter by domain (check if domain is in tile_id)
                    if args["domain"].lower() not in tile_id.lower():
                        continue

                    # Check if corresponding clean tile exists using domain-specific logic
                    clean_tile_id = self.find_clean_pair(tile_id, args["domain"], clean_dir)

                    if clean_tile_id:
                        clean_path = clean_dir / f"{clean_tile_id}.pt"
                        if clean_path.exists():
                            available_tiles.append(tile_id)
                        else:
                            # Try alternative patterns if primary doesn't exist
                            if args["domain"] == "microscopy" and "RawSIMData_gt" in tile_id:
                                # For microscopy, try SIM_gt_a if SIM_gt doesn't exist
                                alt_clean_id = tile_id.replace("RawSIMData_gt", "SIM_gt_a")
                                alt_clean_path = clean_dir / f"{alt_clean_id}.pt"
                                if alt_clean_path.exists():
                                    available_tiles.append(tile_id)
                            elif args["domain"] == "photography":
                                # For photography, try other exposure times if first one doesn't exist
                                parts = tile_id.split('_')
                                for i, part in enumerate(parts):
                                    if part.endswith('s') and '.' in part:  # Find exposure time part
                                        # Try other exposure times
                                        for alt_exp in ['30s', '10s', '4s', '1s']:
                                            if alt_exp != part:  # Don't try the same one again
                                                new_parts = parts.copy()
                                                new_parts[i] = alt_exp
                                                alt_clean_id = '_'.join(new_parts)
                                                alt_clean_path = clean_dir / f"{alt_clean_id}.pt"
                                                if alt_clean_path.exists():
                                                    available_tiles.append(tile_id)
                                                    break
                                        break

            print(f"Found {len(available_tiles)} tiles with matching clean pairs")

            if len(available_tiles) < num_tiles:
                print(f"Warning: Only {len(available_tiles)} tiles with clean pairs, requested {num_tiles}")

            # Randomly select tiles
            rng = np.random.RandomState(42)  # Fixed seed for reproducibility
            selected_indices = rng.choice(len(available_tiles), size=min(num_tiles, len(available_tiles)), replace=False)
            selected_tiles = [available_tiles[i] for i in selected_indices]

            return selected_tiles

        except Exception as e:
            print(f"Error selecting tiles: {e}")
            return []

    def find_clean_pair(self, noisy_tile_id: str, domain: str, clean_dir: Optional[Path] = None) -> Optional[str]:
        """Find the corresponding clean tile ID for a noisy tile using domain-specific logic."""
        if domain == "astronomy":
            # Astronomy: Replace "g800l_sci" with "detection_sci"
            return noisy_tile_id.replace("g800l_sci", "detection_sci")

        elif domain == "microscopy":
            # Microscopy: Try multiple clean naming patterns in order of preference

            # Pattern 1: RawGTSIMData → GTSIM (for RawGTSIMData tiles)
            if "RawGTSIMData" in noisy_tile_id:
                return noisy_tile_id.replace("RawGTSIMData", "GTSIM")

            # Pattern 2: RawSIMData_gt → SIM_gt (most common for RawSIMData_gt tiles)
            elif "RawSIMData_gt" in noisy_tile_id:
                return noisy_tile_id.replace("RawSIMData_gt", "SIM_gt")

            return None

        elif domain == "photography":
            # Photography: Try different clean exposure times in order of preference
            parts = noisy_tile_id.split('_')

            # Detect camera type to choose the most likely exposure time first
            camera_type = None
            for part in parts:
                if part.startswith('sony'):
                    camera_type = 'sony'
                    break
                elif part.startswith('fuji'):
                    camera_type = 'fuji'
                    break

            for i, part in enumerate(parts):
                if part.endswith('s') and '.' in part:  # Find exposure time part
                    # Choose exposure order based on camera type
                    if camera_type == 'sony':
                        # Sony: try 30s first (most common), then 10s, 4s, 1s
                        exposure_order = ['30s', '10s', '4s', '1s']
                    elif camera_type == 'fuji':
                        # Fuji: try 10s first (only available), then others
                        exposure_order = ['10s', '30s', '4s', '1s']
                    else:
                        # Unknown camera: try 30s first (safe default)
                        exposure_order = ['30s', '10s', '4s', '1s']

                    # First check if the original tile_id already exists as a clean tile
                    if clean_dir and clean_dir.exists():
                        original_clean_path = clean_dir / f"{noisy_tile_id}.pt"
                        if original_clean_path.exists():
                            return noisy_tile_id

                    # Try different exposure times
                    for clean_exp in exposure_order:
                        new_parts = parts.copy()
                        new_parts[i] = clean_exp
                        potential_clean_id = '_'.join(new_parts)
                        return potential_clean_id

            return None

        return None

    def load_clean_image(self, tile_id: str, clean_dir: Path, device: torch.device, target_channels: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Load a clean .pt file and return both tensor and metadata."""
        image_path = clean_dir / f"{tile_id}.pt"

        if not image_path.exists():
            raise FileNotFoundError(f"Clean image file not found: {image_path}")

        # Load the image
        tensor = torch.load(image_path, map_location=device)

        # Handle different tensor formats
        if isinstance(tensor, dict):
            if 'clean' in tensor:
                tensor = tensor['clean']
            elif 'clean_norm' in tensor:
                tensor = tensor['clean_norm']
            elif 'image' in tensor:
                tensor = tensor['image']
            else:
                raise ValueError(f"Unrecognized dict structure in {image_path}")

        # Ensure float32
        tensor = tensor.float()

        # Ensure CHW format
        if tensor.ndim == 2:  # (H, W)
            tensor = tensor.unsqueeze(0)  # (1, H, W)
        elif tensor.ndim == 3 and tensor.shape[-1] in [1, 3]:  # (H, W, C)
            tensor = tensor.permute(2, 0, 1)  # (C, H, W)

        # Handle channel conversion
        if target_channels is not None and tensor.shape[0] != target_channels:
            if tensor.shape[0] == 1 and target_channels == 3:
                tensor = tensor.repeat(3, 1, 1)
            elif tensor.shape[0] == 3 and target_channels == 1:
                tensor = tensor.mean(dim=0, keepdim=True)

        # Track metadata
        metadata = {
            'offset': 0.0,
            'original_range': [tensor.min().item(), tensor.max().item()],
            'processed_range': [tensor.min().item(), tensor.max().item()],
            'domain': 'other'
        }

        return tensor, metadata

    def find_clean_exposure(self, tile_id: str, clean_dir: Path) -> Optional[float]:
        """Find matching clean exposure time for the given tile."""
        try:
            parts = tile_id.split('_')

            # Try different clean exposure times in order of preference
            for clean_exp_str in ['30s', '10s', '4s', '1s']:
                new_parts = parts.copy()
                for i, part in enumerate(parts):
                    if part.endswith('s') and '.' in part:  # Find exposure time part
                        new_parts[i] = clean_exp_str
                        break

                clean_tile_id = '_'.join(new_parts)
                clean_path = clean_dir / f"{clean_tile_id}.pt"

                if clean_path.exists():
                    return float(clean_exp_str[:-1])  # Remove 's' and convert

            # If no standard exposure found, search for any clean file with same tile number
            for clean_file in clean_dir.glob("*.pt"):
                try:
                    clean_tile_id = clean_file.stem
                    # Extract exposure from found clean tile
                    clean_parts = clean_tile_id.split('_')
                    for part in clean_parts:
                        if part.endswith('s') and '.' in part:
                            return float(part[:-1])
                except (ValueError, IndexError):
                    continue

        except Exception as e:
            print(f"  Clean exposure search failed: {e}")

        return None

    def extract_exposure_ratio(self, tile_id: str, clean_dir: Optional[Path] = None) -> Optional[float]:
        """Extract exposure ratio α = t_short / t_long from tile ID."""
        try:
            # Extract exposure from tile_id
            short_exp = self.extract_exposure_from_id(tile_id)

            if short_exp is None:
                return None

            # Find matching clean exposure if clean_dir provided
            if clean_dir is not None:
                clean_exp = self.find_clean_exposure(tile_id, clean_dir)
                if clean_exp is not None:
                    alpha = short_exp / clean_exp
                    return alpha

            # Fallback: assume target exposure is 4.0s (typical long exposure)
            alpha = short_exp / 4.0
            return alpha

        except Exception as e:
            print(f"Exposure ratio extraction failed for {tile_id}: {e}")
            return None

    def extract_exposure_from_id(self, tile_id: str) -> Optional[float]:
        """Extract exposure time from tile ID."""
        try:
            parts = tile_id.split('_')
            for part in parts:
                if part.endswith('s') and '.' in part:
                    return float(part[:-1])  # Remove 's' and convert to float
        except (ValueError, IndexError):
            pass
        return None

    def load_and_save_clean_tile(self, tile_id: str, args: Dict[str, Any], tile_output_dir: str) -> Optional[torch.Tensor]:
        """Load clean tile and save it to the output directory."""
        try:
            clean_dir = Path(args["clean_dir"])
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Find clean tile ID using domain-specific logic
            clean_tile_id = self.find_clean_pair(tile_id, args["domain"], clean_dir)

            if clean_tile_id is None:
                print(f"    No clean pair found for {tile_id}")
                return None

            clean_path = clean_dir / f"{clean_tile_id}.pt"
            if not clean_path.exists():
                print(f"    Clean tile file not found: {clean_path}")
                return None

            # Load clean image
            clean_image, clean_metadata = self.load_clean_image(clean_tile_id, clean_dir, device)

            # Create output directory if it doesn't exist
            output_dir = Path(tile_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save clean tile to output directory
            clean_output_path = output_dir / "clean.pt"
            torch.save(clean_image.cpu(), clean_output_path)
            print(f"    ✅ Saved clean tile: {clean_tile_id}")

            return clean_image

        except Exception as e:
            print(f"    ❌ Failed to load/save clean tile for {tile_id}: {e}")
            return None

    def aggregate_tile_results(self, tile_results: List[Dict[str, Any]], output_dir: str) -> Dict[str, Any]:
        """Aggregate results from multiple tile runs."""
        if not tile_results:
            return {}

        # Initialize aggregated structure
        aggregated = {
            "num_samples": len(tile_results),
            "comprehensive_aggregate_metrics": {},
            "results": []
        }

        # Include individual tile results in the aggregated structure
        for tile_result in tile_results:
            if "results" in tile_result:
                # Each tile result should have a "results" array with the actual tile data
                aggregated["results"].extend(tile_result["results"])

        # Aggregate metrics for each method
        methods = ["gaussian_x0", "pg_x0"]

        for method in methods:
            method_metrics = []
            for tile_result in tile_results:
                if method in tile_result.get("comprehensive_aggregate_metrics", {}):
                    method_metrics.append(tile_result["comprehensive_aggregate_metrics"][method])

            if method_metrics:
                # Calculate mean for each metric
                mean_metrics = {}
                for metric in ["mean_ssim", "mean_psnr", "mean_lpips", "mean_niqe"]:
                    values = [m.get(metric, 0) for m in method_metrics if metric in m]
                    if values:
                        mean_metrics[metric] = np.mean(values)

                if mean_metrics:
                    aggregated["comprehensive_aggregate_metrics"][method] = mean_metrics

        return aggregated

    def extract_aggregated_metrics(self, aggregated_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract metrics from aggregated results for scoring."""
        metrics = {}

        # Use PG x0 metrics if available
        if "pg_x0" in aggregated_results.get("comprehensive_aggregate_metrics", {}):
            pg_metrics = aggregated_results["comprehensive_aggregate_metrics"]["pg_x0"]
            metrics = {
                "ssim": pg_metrics.get("mean_ssim", 0.0),
                "psnr": pg_metrics.get("mean_psnr", 0.0),
                "lpips": pg_metrics.get("mean_lpips", 0.0),
                "niqe": pg_metrics.get("mean_niqe", 0.0)
            }
        elif "gaussian_x0" in aggregated_results.get("comprehensive_aggregate_metrics", {}):
            gauss_metrics = aggregated_results["comprehensive_aggregate_metrics"]["gaussian_x0"]
            metrics = {
                "ssim": gauss_metrics.get("mean_ssim", 0.0),
                "psnr": gauss_metrics.get("mean_psnr", 0.0),
                "lpips": gauss_metrics.get("mean_lpips", 0.0),
                "niqe": gauss_metrics.get("mean_niqe", 0.0)
            }

        return metrics

    def extract_comprehensive_results(self, output_dir: str) -> Dict[str, Any]:
        """Extract comprehensive results from the output directory."""
        output_path = Path(output_dir)
        if not output_path.exists():
            return {}

        # Check for comprehensive results.json first (this should contain all detailed info)
        results_file = output_path / "results.json"
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    results_data = json.load(f)
                return results_data
            except Exception as e:
                print(f"Warning: Could not read comprehensive results from {results_file}: {e}")

        # Fallback: Check for summary.json
        summary_file = output_path / "summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)
                return summary_data
            except Exception as e:
                print(f"Warning: Could not read summary from {summary_file}: {e}")

        return {}

    def extract_metrics(self, output_dir: str) -> Dict[str, float]:
        """Extract metrics from the output directory for specific methods only."""
        results_data = self.extract_comprehensive_results(output_dir)

        if not results_data:
            return {}

        # Extract metrics only for gaussian_x0 and pg_x0 methods from comprehensive_aggregate_metrics
        if "comprehensive_aggregate_metrics" in results_data:
            # Only extract metrics for the methods we want
            target_methods = ["gaussian_x0", "pg_x0"]
            all_metrics = {}

            for method in target_methods:
                method_metrics = results_data["comprehensive_aggregate_metrics"].get(method, {})
                if method_metrics:
                    metrics = {
                        "ssim": method_metrics.get("mean_ssim", 0.0),
                        "psnr": method_metrics.get("mean_psnr", 0.0),
                        "lpips": method_metrics.get("mean_lpips", 0.0),
                        "niqe": method_metrics.get("mean_niqe", 0.0),
                        "mse": method_metrics.get("mean_mse", 0.0)
                    }
                    all_metrics[method] = metrics

            # Return metrics for the first available method (they should be similar)
            for method in target_methods:
                if method in all_metrics:
                    return all_metrics[method]

        # Fallback to individual results if available
        if "results" in results_data:
            all_metrics = []
            for result in results_data["results"]:
                if "comprehensive_metrics" in result:
                    # Only extract metrics for target methods
                    target_methods = ["gaussian_x0", "pg_x0"]
                    for method in target_methods:
                        if method in result["comprehensive_metrics"]:
                            metrics = result["comprehensive_metrics"][method]
                            if metrics:
                                all_metrics.append(metrics)

            if all_metrics:
                # Average metrics across examples
                for key in all_metrics[0].keys():
                    values = [m.get(key, 0) for m in all_metrics if key in m]
                    if values:
                        # Filter out NaN values
                        numeric_values = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
                        if numeric_values:
                            return {key: np.mean(numeric_values) for key in all_metrics[0].keys()}
                        else:
                            return {key: 0.0 for key in all_metrics[0].keys()}

        return {}
    
    def optimize_domain(self, domain: str, sensor: str = "default") -> Dict[str, Any]:
        """Optimize single-domain parameters for a given domain and sensor."""
        
        print(f"\n{'='*80}")
        print(f"🔍 Optimizing Single-Domain Parameters: {domain.upper()} ({sensor.upper()})")
        print(f"{'='*80}")
        
        config = self.domain_configs[domain][sensor]
        
        # Generate parameter combinations
        param_combinations = list(itertools.product(
            config["kappa_range"],
            config["sigma_r_range"],
            config["num_steps_range"]
        ))
        
        print(f"Testing {len(param_combinations)} parameter combinations...")
        
        best_params = None
        best_score = -float('inf')
        results = []
        
        for i, (kappa, sigma_r, num_steps) in enumerate(param_combinations):
            run_id = f"single_domain_{domain}_{sensor}_{i+1:03d}"

            # Create output directory
            output_dir = f"results/single_domain_optimization/{domain}_{sensor}/kappa_{kappa}_sigma_{sigma_r}_steps_{num_steps}"

            # Prepare arguments
            args = self.base_args.copy()
            args.update({
                "domain": domain,
                "sensor_name": config["sensor_name"],
                "s": config["s"],
                "sigma_r": sigma_r,
                "kappa": kappa,
                "num_steps": num_steps,
                "output_dir": output_dir,
                "run_id": run_id
            })

            if config["sensor_filter"]:
                args["sensor_filter"] = config["sensor_filter"]

            # Select tiles for this parameter combination (configurable via --num_examples)
            num_tiles = args.get("num_examples", 1)
            selected_tiles = self.select_random_tiles(args, num_tiles=num_tiles)

            if not selected_tiles:
                print(f"❌ No tiles available for parameter combination {i+1}")
                continue

            print(f"\n🎲 Selected {len(selected_tiles)} tiles for parameter combination {i+1}/{len(param_combinations)} (using --num_examples {num_tiles})")

            # Run inference on each tile separately
            tile_results = []
            for tile_idx, tile_id in enumerate(selected_tiles):
                # Run inference on this single tile
                tile_result = self.run_inference_single_tile(args, tile_id, output_dir)

                if tile_result:
                    tile_results.append(tile_result)

            if tile_results:
                # Aggregate results from all tiles
                aggregated_results = self.aggregate_tile_results(tile_results, output_dir)

                if aggregated_results:
                    # Extract metrics for scoring
                    metrics = self.extract_aggregated_metrics(aggregated_results)

                    if metrics:
                        # Calculate composite score: maximize SSIM+PSNR, minimize LPIPS+NIQE
                        # Weight PSNR by 0.01 to normalize scale
                        score = metrics.get("ssim", 0) + metrics.get("psnr", 0) * 0.01 - metrics.get("lpips", 1) - metrics.get("niqe", 10) * 0.1

                        # Save aggregated results.json in parameter sweep directory
                        sweep_results_path = Path(output_dir) / "results.json"
                        try:
                            with open(sweep_results_path, 'w') as f:
                                json.dump(aggregated_results, f, indent=2)
                            print(f"✓ Saved aggregated results to {sweep_results_path}")
                        except Exception as e:
                            print(f"Warning: Could not save comprehensive results: {e}")

                        result = {
                            "run_id": run_id,
                            "domain": domain,
                            "sensor": sensor,
                            "mode": "single_domain",
                            "kappa": kappa,
                            "sigma_r": sigma_r,
                            "num_steps": num_steps,
                            "score": score,
                            "metrics": metrics,
                            "comprehensive_results": aggregated_results  # Include full results
                        }

                        results.append(result)

                        if score > best_score:
                            best_score = score
                            best_params = result

                        print(f"Score: {score:.4f} (SSIM: {metrics.get('ssim', 0):.4f}, PSNR: {metrics.get('psnr', 0):.2f}, LPIPS: {metrics.get('lpips', 1):.4f}, NIQE: {metrics.get('niqe', 10):.2f})")
                    else:
                        print(f"Warning: No metrics found for {run_id}")
                else:
                    print(f"Warning: No comprehensive results found for {run_id}")
            else:
                print(f"Failed: {run_id}")
            
            # Small delay to prevent overwhelming the system
            time.sleep(1)
        
        print(f"\n🏆 Best Single-Domain Parameters for {domain.upper()} ({sensor.upper()}):")
        if best_params:
            print(f"   κ={best_params['kappa']}, σ_r={best_params['sigma_r']}, steps={best_params['num_steps']}")
            print(f"   Score: {best_params['score']:.4f}")
            print(f"   SSIM: {best_params['metrics'].get('ssim', 0):.4f}")
            print(f"   PSNR: {best_params['metrics'].get('psnr', 0):.2f} dB")
            print(f"   LPIPS: {best_params['metrics'].get('lpips', 1):.4f}")
            print(f"   NIQE: {best_params['metrics'].get('niqe', 10):.2f}")
        
        return best_params
    
    def run_optimization(self):
        """Run single-domain optimization for specified domains."""

        print("🎯 Starting Single-Domain Parameter Optimization")
        print("=" * 80)
        print("Targets:")
        print("1. Best SSIM and PSNR while minimizing LPIPS and NIQE")
        print("2. Single-domain optimization for specified domains")
        print("3. Both Gaussian and PG guidance methods")
        print("=" * 80)

        all_results = []

        # Run optimization for specified domains
        for domain in self.base_args.get("domains", ["photography", "microscopy", "astronomy"]):
            if domain == "photography":
                # Photography - split examples between Sony and Fuji
                total_examples = self.base_args.get("num_examples", 20)

                # Create modified base args for each sensor with half the examples
                sony_base_args = self.base_args.copy()
                sony_base_args["num_examples"] = total_examples // 2

                fuji_base_args = self.base_args.copy()
                fuji_base_args["num_examples"] = total_examples // 2

                # Photography - Sony
                sony_optimizer = SingleDomainOptimizer(sony_base_args)
                result = sony_optimizer.optimize_domain("photography", "sony")
                if result:
                    all_results.append(result)

                # Photography - Fuji
                fuji_optimizer = SingleDomainOptimizer(fuji_base_args)
                result = fuji_optimizer.optimize_domain("photography", "fuji")
                if result:
                    all_results.append(result)
            elif domain == "microscopy":
                # Microscopy
                result = self.optimize_domain("microscopy", "default")
                if result:
                    all_results.append(result)
            elif domain == "astronomy":
                # Astronomy
                result = self.optimize_domain("astronomy", "default")
                if result:
                    all_results.append(result)
        
        # Save results
        self.save_results(all_results)
        
        # Print summary
        self.print_summary(all_results)
        
        return all_results
    
    def save_results(self, results: List[Dict[str, Any]]):
        """Save optimization results to files."""
        
        # Create results directory
        results_dir = Path("results/single_domain_optimization")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        with open(results_dir / "single_domain_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save as CSV
        df_data = []
        for result in results:
            row = {
                "domain": result["domain"],
                "sensor": result["sensor"],
                "mode": result["mode"],
                "score": result["score"],
                "kappa": result["kappa"],
                "sigma_r": result["sigma_r"],
                "num_steps": result["num_steps"]
            }
            
            # Add metrics
            for metric, value in result["metrics"].items():
                row[f"metric_{metric}"] = value
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(results_dir / "single_domain_results.csv", index=False)
        
        print(f"\n💾 Results saved to:")
        print(f"   - {results_dir / 'single_domain_results.json'}")
        print(f"   - {results_dir / 'single_domain_results.csv'}")
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """Print optimization summary."""
        
        print("\n" + "="*80)
        print("📊 SINGLE-DOMAIN OPTIMIZATION SUMMARY")
        print("="*80)
        
        # Print best parameters for each configuration
        for result in results:
            print(f"\n🏆 {result['domain'].upper()} ({result['sensor'].upper()}):")
            print(f"   κ={result['kappa']}, σ_r={result['sigma_r']}, steps={result['num_steps']}")
            print(f"   Score: {result['score']:.4f}")
            print(f"   SSIM: {result['metrics'].get('ssim', 0):.4f}")
            print(f"   PSNR: {result['metrics'].get('psnr', 0):.2f} dB")
            print(f"   LPIPS: {result['metrics'].get('lpips', 1):.4f}")
            print(f"   NIQE: {result['metrics'].get('niqe', 10):.2f}")


def main():
    """Main function to run single-domain optimization."""
    
    parser = argparse.ArgumentParser(description="Single-domain parameter optimization for cross-domain low-light enhancement")
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint (.pkl)")
    parser.add_argument("--metadata_json", type=str, required=True, help="Path to metadata JSON file")
    parser.add_argument("--noisy_dir", type=str, required=True, help="Directory containing noisy .pt files")
    parser.add_argument("--clean_dir", type=str, required=True, help="Directory containing clean reference .pt files")
    
    # Optional arguments
    parser.add_argument("--num_examples", type=int, default=20, help="Number of example images to test")
    parser.add_argument("--domains", type=str, nargs="+", default=["photography", "microscopy", "astronomy"],
                       help="Domains to optimize")
    parser.add_argument("--sensor_filter", type=str, help="Filter tiles by sensor type (sony, fuji, etc.)")
    parser.add_argument("--output_base", type=str, default="results/single_domain_optimization",
                       help="Base output directory for results")
    parser.add_argument("--kappa_range", type=float, nargs=3, default=[0.2, 0.5, 0.8],
                       help="Kappa values to test (min, mid, max)")
    parser.add_argument("--sigma_r_range", type=float, nargs=3, default=[2.0, 4.0, 6.0],
                       help="Sigma_r values to test (min, mid, max)")
    parser.add_argument("--num_steps_range", type=int, nargs=3, default=[15, 20, 25],
                       help="Num_steps values to test (min, mid, max)")
    
    args = parser.parse_args()
    
    # Check if required files exist
    required_files = [args.model_path, args.metadata_json, args.noisy_dir, args.clean_dir]
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print("❌ Missing required files:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nPlease ensure all required files exist before running optimization.")
        return False
    
    # Create base arguments
    base_args = {
        "model_path": args.model_path,
        "metadata_json": args.metadata_json,
        "noisy_dir": args.noisy_dir,
        "clean_dir": args.clean_dir,
        "num_examples": args.num_examples,
        "domains": args.domains,
        "sensor_filter": args.sensor_filter,
        "output_base": args.output_base,
        "kappa_range": args.kappa_range,
        "sigma_r_range": args.sigma_r_range,
        "num_steps_range": args.num_steps_range
    }
    
    # Create optimizer
    optimizer = SingleDomainOptimizer(base_args)
    
    # Run optimization
    results = optimizer.run_optimization()
    
    print(f"\n🎉 Single-domain optimization completed!")
    print(f"✅ Total configurations tested: {len(results)}")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

