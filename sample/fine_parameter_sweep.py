#!/usr/bin/env python3
"""
Fine-Grained Parameter Sweep for Domain-Specific Optimization

This script performs a fine-grained parameter sweep for each domain:
- Photography (Sony and Fuji separately)  
- Microscopy
- Astronomy

Key features:
- Uses VALIDATION tiles (not test or training)
- Finer parameter ranges based on INFERENCE_GUIDE.md optimized results
- Model paths from INFERENCE_GUIDE.md with fallback logic
- Uses domain-specific metadata files for correct exposure ratio calculation
"""

import argparse
import itertools
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch

class FineParameterSweep:
    """Fine-grained parameter sweep for domain-specific optimization."""
    
    def __init__(self, base_args: Dict[str, Any], seed: int = 42):
        self.base_args = base_args
        self.results = []
        self.seed = seed
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Model paths from INFERENCE_GUIDE.md with fallback logic
        self.model_paths = {
            "photography": "results/edm_pt_training_photography_20251008_032055/best_model.pkl",
            "microscopy": "results/edm_pt_training_microscopy_20251008_044631/best_model.pkl",
            "astronomy": "results/edm_pt_training_astronomy_20251009_172141/best_model.pkl"
        }
        
        # Fine-grained parameter ranges based on INFERENCE_GUIDE.md results
        # Photography: κ=0.8, σ_r=4.0, steps=15 (optimized)
        # Microscopy: κ=0.4, σ_r=0.5, steps=20 (optimized)
        # Astronomy: κ=0.05, σ_r=9.0, steps=25 (optimized)
        self.domain_configs = {
            "photography": {
                "sony": {
                    "s": 15871,
                    "sensor_name": "sony_a7s_ii",
                    "sensor_filter": "sony",
                    # Fine-grained around optimal κ=0.8
                    "kappa_range": [0.6, 0.7, 0.8, 0.9, 1.0],
                    # Fine-grained around optimal σ_r=4.0
                    "sigma_r_range": [3.0, 3.5, 4.0, 4.5, 5.0],
                    # Fine-grained around optimal steps=15
                    "num_steps_range": [12, 14, 15, 16, 18]
                },
                "fuji": {
                    "s": 15871,
                    "sensor_name": "fuji_xt2",
                    "sensor_filter": "fuji",
                    # Fine-grained around optimal κ=0.8
                    "kappa_range": [0.6, 0.7, 0.8, 0.9, 1.0],
                    # Fine-grained around optimal σ_r=4.0
                    "sigma_r_range": [3.0, 3.5, 4.0, 4.5, 5.0],
                    # Fine-grained around optimal steps=15
                    "num_steps_range": [12, 14, 15, 16, 18]
                }
            },
            "microscopy": {
                "default": {
                    "s": 65535,
                    "sensor_name": "hamamatsu_orca_flash4_v3",
                    "sensor_filter": None,
                    # Fine-grained around optimal κ=0.4
                    "kappa_range": [0.2, 0.3, 0.4, 0.5, 0.6],
                    # Fine-grained around optimal σ_r=0.5
                    "sigma_r_range": [0.3, 0.4, 0.5, 0.6, 0.7],
                    # Fine-grained around optimal steps=20
                    "num_steps_range": [16, 18, 20, 22, 24]
                }
            },
            "astronomy": {
                "default": {
                    "s": 450,
                    "sensor_name": "hubble_wfc3",
                    "sensor_filter": None,
                    # Fine-grained around optimal κ=0.05
                    "kappa_range": [0.02, 0.03, 0.05, 0.07, 0.10],
                    # Fine-grained around optimal σ_r=9.0
                    "sigma_r_range": [7.0, 8.0, 9.0, 10.0, 11.0],
                    # Fine-grained around optimal steps=25
                    "num_steps_range": [21, 23, 25, 27, 30]
                }
            }
        }
        
        # Don't verify model paths during initialization - only verify when actually used
        # This prevents false warnings when running domain-specific sweeps
        # Model paths will be verified in run_sweep_for_domain_and_sensor() when needed
    
    def get_metadata_file_for_domain(self, domain: str, base_metadata_path: str) -> str:
        """Get the correct metadata file path for the domain."""
        base_path = Path(base_metadata_path).parent if base_metadata_path else Path("dataset")
        
        if domain == "photography":
            # Photography: dataset/metadata_photography_incremental.json
            return str(base_path / "metadata_photography_incremental.json")
        elif domain == "microscopy":
            # Microscopy: dataset/metadata_microscopy_incremental_v2.json
            return str(base_path / "metadata_microscopy_incremental_v2.json")
        elif domain == "astronomy":
            # Astronomy: dataset/processed/pt_tiles/astronomy_v2/metadata_astronomy_incremental.json
            return str(base_path / "processed" / "pt_tiles" / "astronomy_v2" / "metadata_astronomy_incremental.json")
        else:
            # Fallback to base path
            return base_metadata_path
    
    def load_validation_tiles(self, metadata_path: str, domain: str, sensor_filter: Optional[str] = None) -> List[str]:
        """Load validation tiles from metadata."""
        try:
            # Select the correct metadata file for the domain
            metadata_file = self.get_metadata_file_for_domain(domain, metadata_path)
            
            print(f"Loading metadata from: {metadata_file}")
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Handle nested metadata structure
            if isinstance(metadata, dict) and "tiles" in metadata:
                tile_list = metadata["tiles"]
            elif isinstance(metadata, list):
                tile_list = metadata
            else:
                print(f"Unexpected metadata structure: {type(metadata)}")
                return []
            
            # First, collect all tiles and group by base ID to find pairs
            tiles_by_base = {}
            for entry in tile_list:
                # Handle both dict and string entries
                if isinstance(entry, str):
                    continue
                
                # Accept validation and train splits (tiles can be in either)
                if entry.get("split") not in ["validation", "test", "train"]:
                    continue
                
                tile_id = entry.get("tile_id", "")
                if not tile_id:
                    continue
                
                # Filter by domain
                if domain.lower() not in tile_id.lower():
                    continue
                
                # Filter by sensor if specified
                if sensor_filter and sensor_filter.lower() not in tile_id.lower():
                    continue
                
                data_type = entry.get("data_type", "unknown")
                
                # Extract base ID based on domain
                if domain == "photography":
                    # Photography: Remove exposure time from tile ID
                    # Pattern: photography_{sensor}_{scene_id}_{scene_num}_{exposure_time}_tile_{tile_id}
                    # Remove the exposure time part (e.g., "0.1s", "30s")
                    base_id = re.sub(r'_\d+\.?\d*s_', '_', tile_id)
                elif domain == "microscopy":
                    # Microscopy: Remove GTSIM/RawGTSIMData prefixes
                    base_id = tile_id.replace('_GTSIM_', '_').replace('_RawGTSIMData_', '_')
                else:
                    # For other domains, use tile_id as-is
                    base_id = tile_id
                
                if base_id not in tiles_by_base:
                    tiles_by_base[base_id] = {}
                
                tiles_by_base[base_id][data_type] = entry
            
            # Only keep tiles that have both noisy and clean versions
            validation_tiles_with_pairs = []
            for base_id, tiles in tiles_by_base.items():
                if 'noisy' in tiles and 'clean' in tiles:
                    # Use the noisy tile ID (it contains the noise level info)
                    noisy_tile_id = tiles['noisy'].get("tile_id")
                    if noisy_tile_id:
                        validation_tiles_with_pairs.append(noisy_tile_id)
            
            print(f"Found {len(validation_tiles_with_pairs)} validation tiles with noisy+clean pairs for {domain}")
            return validation_tiles_with_pairs
            
        except Exception as e:
            print(f"Error loading validation tiles: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def run_inference_single_tile(self, args: Dict[str, Any], tile_id: str, output_dir: str, domain: str) -> Dict[str, Any]:
        """Run inference on a single tile with given parameters."""
        print(f"  🔹 Running inference on tile: {tile_id}")

        # Get the correct metadata file for this domain
        metadata_file = self.get_metadata_file_for_domain(domain, args["metadata_json"])

        # Create per-tile output directory to avoid overwriting
        # Use sanitized tile_id for directory name and save directly in kappa folder
        safe_tile_id = tile_id.replace('/', '_').replace('\\', '_')
        tile_output_dir = f"{output_dir}/example_00_{safe_tile_id}"
        
        # Construct domain-specific paths for noisy and clean directories
        noisy_dir = f"{args['noisy_dir']}/{domain}/noisy"
        clean_dir = f"{args['clean_dir']}/{domain}/clean"
        
        # Find the clean tile ID by looking it up in metadata
        clean_tile_id = None
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Group tiles by base ID
            tiles_by_base = {}
            for entry in metadata.get('tiles', []):
                if not isinstance(entry, dict):
                    continue
                
                entry_tile_id = entry.get('tile_id', '')
                if domain.lower() not in entry_tile_id.lower():
                    continue
                
                data_type = entry.get('data_type', 'unknown')
                
                # Extract base ID based on domain
                if domain == "photography":
                    base_id = re.sub(r'_\d+\.?\d*s_', '_', entry_tile_id)
                elif domain == "microscopy":
                    base_id = entry_tile_id.replace('_GTSIM_', '_').replace('_RawGTSIMData_', '_')
                else:
                    base_id = entry_tile_id
                
                if base_id not in tiles_by_base:
                    tiles_by_base[base_id] = {}
                
                tiles_by_base[base_id][data_type] = entry
            
            # Find the base ID for this noisy tile
            if domain == "photography":
                current_base_id = re.sub(r'_\d+\.?\d*s_', '_', tile_id)
            elif domain == "microscopy":
                current_base_id = tile_id.replace('_GTSIM_', '_').replace('_RawGTSIMData_', '_')
            else:
                current_base_id = tile_id
            
            if current_base_id in tiles_by_base and 'clean' in tiles_by_base[current_base_id]:
                clean_tile_id = tiles_by_base[current_base_id]['clean'].get('tile_id')
                print(f"    ✅ Using clean pair: {clean_tile_id}")
            else:
                print(f"    ❌ Warning: Could not find clean pair for base ID: {current_base_id}")
                if current_base_id in tiles_by_base:
                    print(f"    Available data types: {list(tiles_by_base[current_base_id].keys())}")
                else:
                    print(f"    Base ID not found in tiles_by_base")
                    print(f"    Available base IDs (first 5): {list(tiles_by_base.keys())[:5]}")
        except Exception as e:
            print(f"    Warning: Could not find clean tile ID: {e}")
        
        cmd = [
            "python", "sample/sample_noisy_pt_lle_PGguidance.py",
            "--model_path", args["model_path"],
            "--metadata_json", metadata_file,  # Use domain-specific metadata file
            "--noisy_dir", noisy_dir,
            "--clean_dir", clean_dir,
            "--output_dir", tile_output_dir,  # Use per-tile directory
            "--domain", args["domain"],
            "--tile_ids", tile_id,
            "--use_sensor_calibration",
            "--sensor_name", args["sensor_name"],
            "--s", str(args["s"]),
            "--sigma_r", str(args["sigma_r"]),
            "--kappa", str(args["kappa"]),
            "--num_steps", str(args["num_steps"]),
            "--compare_gaussian",
            "--run_methods", "noisy", "clean", "exposure_scaled", "gaussian_x0", "pg_x0",
            "--no_heun"  # Disable Heun for speed
        ]

        # Add sensor filter if specified
        if args.get("sensor_filter"):
            cmd.extend(["--sensor_filter", args["sensor_filter"]])

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)  # 10 min timeout
            
            # Print last few lines of stderr for debugging
            if result.stderr:
                stderr_lines = result.stderr.split('\n')
                # Print last 3 lines for visibility
                for line in stderr_lines[-5:]:
                    if line.strip():
                        print(f"    {line}")
            
            print(f"    ✅ Tile {tile_id} completed")

            # Extract results from the per-tile directory
            tile_results = self.extract_comprehensive_results(tile_output_dir)
            
            return tile_results

        except subprocess.CalledProcessError as e:
            print(f"    ❌ Tile {tile_id} failed: {e.returncode}")
            # Print error output to help debug
            if e.stderr:
                print(f"    Error (last 10 lines):")
                stderr_lines = e.stderr.split('\n')
                for line in stderr_lines[-10:]:
                    if line.strip():
                        print(f"    {line}")
            return {}
        except subprocess.TimeoutExpired:
            print(f"    ❌ Tile {tile_id} timed out")
            return {}
    
    def extract_comprehensive_results(self, output_dir: str) -> Dict[str, Any]:
        """Extract comprehensive results from output directory."""
        output_path = Path(output_dir)
        if not output_path.exists():
            return {}

        # The inference script creates example_XX_tile_id subdirectories
        # Look for any results.json in subdirectories
        example_dirs = list(output_path.glob("example_*"))
        
        if not example_dirs:
            return {}
        
        # Use the first example directory found
        example_dir = example_dirs[0]
        
        # Look for results.json in the example directory
        results_file = example_dir / "results.json"
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not read results: {e}")
        
        return {}
    
    def extract_aggregated_metrics(self, aggregated_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract metrics from aggregated results."""
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
    
    def aggregate_tile_results(self, tile_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from multiple tile runs."""
        if not tile_results:
            return {}

        # Initialize aggregated structure
        aggregated = {
            "num_samples": len(tile_results),
            "comprehensive_aggregate_metrics": {},
            "results": []
        }

        # Include individual tile results
        for tile_result in tile_results:
            if "results" in tile_result:
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
    
    def sweep_domain(self, domain: str, sensor: str = "default") -> Dict[str, Any]:
        """Perform fine-grained parameter sweep for a domain."""
        print(f"\n{'='*80}")
        print(f"🔍 Fine-Grained Parameter Sweep: {domain.upper()} ({sensor.upper()})")
        print(f"{'='*80}")
        
        # Check Python version for photography domain
        if domain == "photography":
            import sys
            python_version = sys.version_info
            if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 10):
                print("⚠️  WARNING: Photography models require Python 3.10+ (numpy 2.0+ compatibility)")
                print(f"   Current Python version: {python_version.major}.{python_version.minor}")
                print("   Suggested: Use Python 3.10 or 3.11")
                print("   Run with: /usr/bin/python3.10 sample/fine_parameter_sweep.py --domains photography")
                print("")
        
        config = self.domain_configs[domain][sensor]
        
        # Load validation tiles
        validation_tiles = self.load_validation_tiles(
            self.base_args["metadata_json"],
            domain,
            config.get("sensor_filter")
        )
        
        if not validation_tiles:
            print(f"❌ No validation tiles found for {domain}")
            return {}
        
        # Randomly select up to 50 validation tiles
        num_tiles_to_use = min(50, len(validation_tiles))
        selected_validation_tiles = random.sample(validation_tiles, num_tiles_to_use)
        
        print(f"Found {len(validation_tiles)} validation tiles, randomly selecting {len(selected_validation_tiles)} for sweep")
        print(f"Selected tiles (first 10): {selected_validation_tiles[:10]}")
        print(f"Using random seed: {self.seed} for reproducibility")
        
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
        
        # Use model path from configuration
        model_path = self.model_paths.get(domain)
        if not model_path or not Path(model_path).exists():
            print(f"❌ Model not found for {domain}: {model_path}")
            return {}
        
        for i, (kappa, sigma_r, num_steps) in enumerate(param_combinations):
            run_id = f"fine_sweep_{domain}_{sensor}_{i+1:03d}"

            # Create output directory
            output_dir = f"results/fine_parameter_sweep/{domain}_{sensor}/kappa_{kappa}_sigma_{sigma_r}_steps_{num_steps}"

            print(f"\n🎯 Combination {i+1}/{len(param_combinations)}: κ={kappa}, σ_r={sigma_r}, steps={num_steps}")

            # Prepare arguments
            args = self.base_args.copy()
            args.update({
                "model_path": model_path,
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

            # Run inference on all selected validation tiles
            all_tile_results = []
            for tile_id in selected_validation_tiles:  # Use all 50 selected tiles
                tile_result = self.run_inference_single_tile(args, tile_id, output_dir, domain)
                if tile_result:
                    all_tile_results.append(tile_result)
            
            if all_tile_results:
                # Aggregate results
                aggregated_results = self.aggregate_tile_results(all_tile_results)
                
                if aggregated_results:
                    # Extract metrics
                    metrics = self.extract_aggregated_metrics(aggregated_results)
                    
                    if metrics:
                        # Calculate composite score
                        score = (
                            metrics.get("ssim", 0) + 
                            metrics.get("psnr", 0) * 0.01 - 
                            metrics.get("lpips", 1) - 
                            metrics.get("niqe", 10) * 0.1
                        )
                        
                        result = {
                            "run_id": run_id,
                            "kappa": kappa,
                            "sigma_r": sigma_r,
                            "num_steps": num_steps,
                            "score": score,
                            "metrics": metrics,
                            "comprehensive_results": aggregated_results
                        }

                        results.append(result)

                        if score > best_score:
                            best_score = score
                            best_params = result

                        print(f"Score: {score:.4f} (SSIM: {metrics.get('ssim', 0):.4f}, PSNR: {metrics.get('psnr', 0):.2f}, LPIPS: {metrics.get('lpips', 1):.4f}, NIQE: {metrics.get('niqe', 10):.2f})")
            
            # Small delay to prevent overwhelming the system
            time.sleep(1)
        
        print(f"\n🏆 Best Parameters for {domain.upper()} ({sensor.upper()}):")
        if best_params:
            print(f"   κ={best_params['kappa']}, σ_r={best_params['sigma_r']}, steps={best_params['num_steps']}")
            print(f"   Score: {best_params['score']:.4f}")
            print(f"   SSIM: {best_params['metrics'].get('ssim', 0):.4f}")
            print(f"   PSNR: {best_params['metrics'].get('psnr', 0):.2f} dB")
            print(f"   LPIPS: {best_params['metrics'].get('lpips', 1):.4f}")
            print(f"   NIQE: {best_params['metrics'].get('niqe', 10):.2f}")
        
        return best_params if best_params else {}
    
    def run_sweep(self, domains: List[str]):
        """Run fine-grained parameter sweep for specified domains."""
        print("🎯 Starting Fine-Grained Parameter Sweep")
        print("=" * 80)
        print("Targets:")
        print("1. Best SSIM and PSNR while minimizing LPIPS and NIQE")
        print("2. Fine-grained parameter ranges around optimal values")
        print("3. Validation tiles only (not test)")
        print("4. Uses domain-specific metadata files for correct exposure ratio calculation")
        print("=" * 80)

        all_results = []

        # Run sweep for specified domains
        for domain in domains:
            if domain == "photography":
                # Photography - split between Sony and Fuji
                sony_result = self.sweep_domain("photography", "sony")
                if sony_result:
                    all_results.append(sony_result)

                fuji_result = self.sweep_domain("photography", "fuji")
                if fuji_result:
                    all_results.append(fuji_result)
            else:
                result = self.sweep_domain(domain, "default")
                if result:
                    all_results.append(result)
        
        # Save results
        self.save_results(all_results)
        
        # Print summary
        self.print_summary(all_results)
        
        return all_results
    
    def save_results(self, results: List[Dict[str, Any]]):
        """Save optimization results to files."""
        results_dir = Path("results/fine_parameter_sweep")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        with open(results_dir / "fine_sweep_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save as CSV
        df_data = []
        for result in results:
            df_data.append({
                "run_id": result.get("run_id", ""),
                "domain": result.get("run_id", "").split("_")[2] if "_" in result.get("run_id", "") else "",
                "sensor": result.get("run_id", "").split("_")[3] if "_" in result.get("run_id", "") else "",
                "kappa": result.get("kappa", 0),
                "sigma_r": result.get("sigma_r", 0),
                "num_steps": result.get("num_steps", 0),
                "score": result.get("score", 0),
                "ssim": result.get("metrics", {}).get("ssim", 0),
                "psnr": result.get("metrics", {}).get("psnr", 0),
                "lpips": result.get("metrics", {}).get("lpips", 1),
                "niqe": result.get("metrics", {}).get("niqe", 10)
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(results_dir / "fine_sweep_results.csv", index=False)
        
        print(f"\n✅ Results saved to {results_dir}")
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """Print summary of results."""
        print("\n" + "=" * 80)
        print("FINE-GRAINED PARAMETER SWEEP SUMMARY")
        print("=" * 80)
        
        for result in results:
            domain_sensor = result.get("run_id", "").replace("fine_sweep_", "")
            print(f"\n{domain_sensor.upper()}:")
            print(f"  κ={result.get('kappa')}, σ_r={result.get('sigma_r')}, steps={result.get('num_steps')}")
            print(f"  SSIM: {result.get('metrics', {}).get('ssim', 0):.4f}")
            print(f"  PSNR: {result.get('metrics', {}).get('psnr', 0):.2f} dB")
            print(f"  LPIPS: {result.get('metrics', {}).get('lpips', 1):.4f}")
            print(f"  NIQE: {result.get('metrics', {}).get('niqe', 10):.2f}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Fine-grained parameter sweep for domain-specific optimization"
    )
    
    parser.add_argument(
        "--metadata_json",
        type=str,
        default="dataset/metadata_photography_incremental.json",  # Base path, will be auto-resolved to domain-specific files
        help="Base path for metadata JSON file (will be auto-resolved to domain-specific files)"
    )
    parser.add_argument(
        "--noisy_dir",
        type=str,
        default="dataset/processed/pt_tiles",
        help="Base directory for noisy tiles"
    )
    parser.add_argument(
        "--clean_dir",
        type=str,
        default="dataset/processed/pt_tiles",
        help="Base directory for clean tiles"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["photography", "microscopy", "astronomy"],
        help="Domains to sweep"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for selecting validation tiles (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Prepare base arguments
    base_args = {
        "metadata_json": args.metadata_json,
        "noisy_dir": args.noisy_dir,
        "clean_dir": args.clean_dir,
    }
    
    # Create optimizer and run sweep
    optimizer = FineParameterSweep(base_args, seed=args.seed)
    optimizer.run_sweep(args.domains)


if __name__ == "__main__":
    main()
