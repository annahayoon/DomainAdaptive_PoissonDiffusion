#!/usr/bin/env python3
"""
Scale .pt image files using domain-specific preprocessing scale factors.
"""

import json
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import get_image_range, load_tensor_from_pt


def get_domain_scaling_factor(domain, tile_path):
    """Get the domain scaling factor from preprocessing (s parameter)."""
    results_file = tile_path / "results.json"
    if not results_file.exists():
        print(f"Warning: No results.json found for {tile_path}")
        return None

    try:
        with open(results_file, "r") as f:
            data = json.load(f)

        pg_params = data.get("pg_guidance_params", {})
        s = pg_params.get("s", 1.0)

        return s
    except Exception as e:
        print(f"Error extracting scaling factor for {tile_path}: {e}")
        return None


def scale_and_save_image(image_path, scale_factor, output_path):
    """Scale image by multiplying with scale factor and save."""
    try:
        # Load the original image using shared utility
        tensor = load_tensor_from_pt(image_path)

        # Scale the image
        scaled_tensor = tensor * scale_factor

        # Save the scaled image
        torch.save(scaled_tensor, output_path)
        print(f"Scaled {image_path} by {scale_factor} -> {output_path}")

        # Return the scaled tensor for range calculation
        return scaled_tensor

    except Exception as e:
        print(f"Error scaling {image_path}: {e}")
        return None


def process_domain_examples(domain, examples):
    print(f"\n=== Processing {domain.upper()} ===")

    # Map domain to optimized directory
    domain_to_opt_dir = {
        "astronomy": "astronomy_optimized",
        "microscopy": "microscopy_optimized",
        "photography_sony": "photography_sony_optimized",
        "photography_fuji": "photography_fuji_optimized",
    }

    opt_dir = domain_to_opt_dir.get(domain)
    if not opt_dir:
        print(f"Unknown domain: {domain}")
        return

    base_path = Path("/home/jilab/Jae/results/optimized_inference_all_tiles") / opt_dir

    scaled_data = {}

    for example_name in examples:
        example_path = base_path / example_name
        if not example_path.exists():
            print(f"Example not found: {example_path}")
            continue

        # Get scaling factor
        scale_factor = get_domain_scaling_factor(domain, example_path)
        if scale_factor is None:
            print(f"Could not get scaling factor for {example_name}")
            continue

        print(f"Processing {example_name} with scale factor: {scale_factor}")

        # Create output directory for scaled images
        scaled_dir = example_path / "scaled"
        scaled_dir.mkdir(exist_ok=True)

        # Process each method
        methods = [
            "noisy",
            "clean",
            "restored_exposure_scaled",
            "restored_gaussian_x0",
            "restored_pg_x0",
        ]
        method_ranges = {}

        for method in methods:
            img_path = example_path / f"{method}.pt"
            if img_path.exists():
                scaled_path = scaled_dir / f"{method}_scaled.pt"
                scaled_tensor = scale_and_save_image(
                    img_path, scale_factor, scaled_path
                )

                if scaled_tensor is not None:
                    img_range = get_image_range(scaled_tensor)
                    method_ranges[method] = img_range
                    print(
                        f"  {method}: [{img_range['min']:.3f}, {img_range['max']:.3f}]"
                    )

        scaled_data[example_name] = {
            "scale_factor": scale_factor,
            "method_ranges": method_ranges,
        }

    return scaled_data


def main():
    # Define representative examples for each domain
    examples = {
        "astronomy": [
            "example_00_astronomy_j8g6z3jdq_g800l_sci_tile_0071",
            "example_01_astronomy_j8hpakg9q_g800l_sci_tile_0044",
            "example_02_astronomy_j8hqe3c6q_g800l_sci_tile_0005",
        ],
        "microscopy": [
            "example_00_microscopy_ER_Cell_002_RawGTSIMData_level_06_tile_0002",
            "example_01_microscopy_F-actin_Cell_024_RawSIMData_gt_tile_0006",
            "example_02_microscopy_ER_Cell_058_RawGTSIMData_level_05_tile_0001",
        ],
        "photography_sony": [
            "example_00_photography_sony_00135_00_0.1s_tile_0005",
            "example_02_photography_sony_00135_00_0.1s_tile_0034",
        ],
        "photography_fuji": [
            "example_00_photography_fuji_00017_00_0.1s_tile_0009",
            "example_01_photography_fuji_20184_00_0.033s_tile_0011",
            "example_02_photography_fuji_00077_00_0.04s_tile_0022",
        ],
    }

    all_scaled_data = {}

    for domain, domain_examples in examples.items():
        domain_data = process_domain_examples(domain, domain_examples)
        all_scaled_data[domain] = domain_data

    # Save scaling summary
    output_dir = Path("scaled_images_summary")
    output_dir.mkdir(exist_ok=True)

    import json

    with open(output_dir / "scaled_ranges.json", "w") as f:
        json.dump(all_scaled_data, f, indent=2)

    print(
        f"\nScaling complete! Scaled images saved to individual 'scaled/' directories"
    )
    print(f"Summary saved to {output_dir / 'scaled_ranges.json'}")


if __name__ == "__main__":
    main()
