#!/usr/bin/env python3
"""
Run optimized inference on all test tiles across all domains using cross-domain model.

This script:
1. Uses the cross-domain model (results/edm_pt_training_multi_domain_20251010_182807/best_model.pkl)
2. Runs on all test tiles in all three domains (photography, microscopy, astronomy)
3. Uses optimized unified cross-domain parameters (κ=0.2, σ_r=2.0, steps=15)
4. Saves metrics results to JSON (no visualizations)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

class OptimizedInferenceRunner:
    """Run optimized inference on all test tiles."""
    
    def __init__(self, model_path: str, output_dir: str):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Unified cross-domain optimized parameters
        self.unified_params = {
            "kappa": 0.2,
            "sigma_r": 2.0,
            "num_steps": 15
        }
        
        # Domain configurations
        self.domain_configs = {
            "photography": {
                "noisy_dir": "dataset/processed/pt_tiles/photography/noisy",
                "clean_dir": "dataset/processed/pt_tiles/photography/clean",
                "metadata": "dataset/processed/comprehensive_tiles_metadata.json",
                "s": None  # Will auto-set
            },
            "microscopy": {
                "noisy_dir": "dataset/processed/pt_tiles/microscopy/noisy",
                "clean_dir": "dataset/processed/pt_tiles/microscopy/clean",
                "metadata": "dataset/processed/comprehensive_tiles_metadata.json",
                "s": None  # Will auto-set
            },
            "astronomy": {
                "noisy_dir": "dataset/processed/pt_tiles/astronomy/noisy",
                "clean_dir": "dataset/processed/pt_tiles/astronomy/clean",
                "metadata": "dataset/processed/comprehensive_tiles_metadata.json",
                "s": None  # Will auto-set (astronomy domain_max=450)
            }
        }
    
    def get_test_tiles(self, domain: str, sensor_filter: Optional[str] = None) -> List[str]:
        """Get all test tiles for a domain from the filesystem."""
        config = self.domain_configs[domain]
        noisy_dir = Path(config["noisy_dir"])
        
        if not noisy_dir.exists():
            print(f"Warning: Noisy directory not found: {noisy_dir}")
            return []
        
        tiles = []
        for pt_file in noisy_dir.glob("*.pt"):
            tile_id = pt_file.stem
            
            # Filter by sensor if specified
            if sensor_filter and sensor_filter.lower() not in tile_id.lower():
                continue
            
            # Verify it's from the correct domain
            if domain.lower() in tile_id.lower():
                tiles.append(tile_id)
        
        return sorted(tiles)
    
    def run_inference_domain(self, domain: str, num_tiles: int = 1000) -> Dict[str, Any]:
        """Run inference on all test tiles for a domain."""
        config = self.domain_configs[domain]
        
        # Create domain-specific output directory
        domain_output = self.output_dir / domain
        domain_output.mkdir(exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"Running inference on {domain.upper()}: up to {num_tiles} test tiles")
        print(f"Parameters: κ={self.unified_params['kappa']}, σ_r={self.unified_params['sigma_r']}, steps={self.unified_params['num_steps']}")
        print(f"{'='*80}")
        
        # Build command - let inference script select tiles from test split
        cmd = [
            "python", "sample/sample_noisy_pt_lle_PGguidance.py",
            "--model_path", self.model_path,
            "--metadata_json", config["metadata"],
            "--noisy_dir", config["noisy_dir"],
            "--clean_dir", config["clean_dir"],
            "--output_dir", str(domain_output),
            "--domain", domain,
            "--num_examples", str(num_tiles),  # Let script select from test split
            "--cross_domain_kappa", str(self.unified_params["kappa"]),
            "--cross_domain_sigma_r", str(self.unified_params["sigma_r"]),
            "--cross_domain_num_steps", str(self.unified_params["num_steps"]),
            "--use_sensor_calibration",
            "--run_methods", "pg_x0_cross",  # Only run cross-domain PG method
            "--device", "cuda:0"
        ]
        
        # Add s parameter if specified
        if config["s"] is not None:
            cmd.extend(["--s", str(config["s"])])
        
        try:
            print(f"\n  Running inference...")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=36000  # 10 hour timeout for all tiles
            )
            
            print(f"  ✅ {domain.upper()} completed successfully")
            
            # Extract results
            results_file = domain_output / "results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    domain_results = json.load(f)
                return domain_results
            else:
                print(f"  Warning: No results file found at {results_file}")
                return {}
            
        except subprocess.CalledProcessError as e:
            print(f"  ❌ {domain.upper()} failed with exit code {e.returncode}")
            print(f"  Error: {e.stderr[:1000]}")  # Print first 1000 chars of error
            return {}
        except subprocess.TimeoutExpired:
            print(f"  ❌ {domain.upper()} timed out after 10 hours")
            return {}
    
    def aggregate_results(self, batch_results: List[Dict[str, Any]], domain: str) -> Dict[str, Any]:
        """Aggregate results from multiple batches."""
        if not batch_results:
            return {}
        
        # Collect all individual tile results
        all_tile_results = []
        for batch in batch_results:
            if "results" in batch:
                all_tile_results.extend(batch["results"])
        
        # Calculate aggregate metrics
        aggregate_metrics = {}
        
        if all_tile_results:
            # Collect metrics from all tiles
            all_metrics = []
            for tile_result in all_tile_results:
                if "comprehensive_metrics" in tile_result and "pg_x0_cross" in tile_result["comprehensive_metrics"]:
                    all_metrics.append(tile_result["comprehensive_metrics"]["pg_x0_cross"])
            
            if all_metrics:
                # Calculate mean metrics
                metric_keys = ["ssim", "psnr", "lpips", "niqe", "mse"]
                for key in metric_keys:
                    values = [m.get(key, np.nan) for m in all_metrics]
                    # Filter out NaN values
                    valid_values = [v for v in values if not np.isnan(v)]
                    if valid_values:
                        aggregate_metrics[f"mean_{key}"] = float(np.mean(valid_values))
                        aggregate_metrics[f"std_{key}"] = float(np.std(valid_values))
                        aggregate_metrics[f"median_{key}"] = float(np.median(valid_values))
        
        return {
            "domain": domain,
            "num_tiles": len(all_tile_results),
            "parameters": self.unified_params,
            "aggregate_metrics": aggregate_metrics,
            "individual_results": all_tile_results
        }
    
    def run_all_domains(self, num_tiles_per_domain: int = 1000):
        """Run inference on all domains."""
        all_domain_results = {}
        
        # Photography domain
        photo_results = self.run_inference_domain("photography", num_tiles_per_domain)
        if photo_results:
            all_domain_results["photography"] = photo_results
        
        # Microscopy domain
        micro_results = self.run_inference_domain("microscopy", num_tiles_per_domain)
        if micro_results:
            all_domain_results["microscopy"] = micro_results
        
        # Astronomy domain
        astro_results = self.run_inference_domain("astronomy", num_tiles_per_domain)
        if astro_results:
            all_domain_results["astronomy"] = astro_results
        
        # Save combined results
        combined_file = self.output_dir / "all_domains_results.json"
        with open(combined_file, 'w') as f:
            json.dump(all_domain_results, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"✅ All domains completed!")
        print(f"{'='*80}")
        print(f"\nSaved combined results to: {combined_file}")
        
        # Print summary
        print("\n📊 SUMMARY:")
        for domain, results in all_domain_results.items():
            print(f"\n{domain.upper()}:")
            print(f"  Tiles processed: {results.get('num_samples', 0)}")
            if "comprehensive_aggregate_metrics" in results:
                # Look for pg_x0_cross metrics
                if "pg_x0_cross" in results["comprehensive_aggregate_metrics"]:
                    metrics = results["comprehensive_aggregate_metrics"]["pg_x0_cross"]
                    if "mean_ssim" in metrics:
                        print(f"  SSIM: {metrics['mean_ssim']:.4f} ± {metrics.get('std_ssim', 0):.4f}")
                    if "mean_psnr" in metrics:
                        print(f"  PSNR: {metrics['mean_psnr']:.2f} ± {metrics.get('std_psnr', 0):.2f} dB")
                    if "mean_lpips" in metrics:
                        print(f"  LPIPS: {metrics['mean_lpips']:.4f} ± {metrics.get('std_lpips', 0):.4f}")
                    if "mean_niqe" in metrics:
                        print(f"  NIQE: {metrics['mean_niqe']:.2f} ± {metrics.get('std_niqe', 0):.2f}")
        
        return all_domain_results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run optimized inference on all test tiles across all domains"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="results/edm_pt_training_multi_domain_20251010_182807/best_model.pkl",
        help="Path to cross-domain model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/inference_optimized_all_domains",
        help="Output directory for results"
    )
    parser.add_argument(
        "--num_tiles",
        type=int,
        default=1000,
        help="Number of tiles to process per domain"
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model_path).exists():
        print(f"❌ Model not found: {args.model_path}")
        return False
    
    # Create runner and execute
    runner = OptimizedInferenceRunner(args.model_path, args.output_dir)
    runner.run_all_domains(num_tiles_per_domain=args.num_tiles)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

