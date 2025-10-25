#!/usr/bin/env python3
"""
Create comparison visualization with domain-specific scaling and representative examples.
Uses example_00, example_01, example_02 from each domain.
Properly separates single-domain vs cross-domain results.
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_image_tensor(image_path, domain=None):
    """Load a .pt file and return as numpy array, denormalizing from [-1,1] if needed."""
    if not image_path.exists():
        return None, False

    try:
        tensor = torch.load(image_path)
        # Convert to numpy and squeeze singleton dimensions
        array = tensor.numpy()
        is_rgb = False

        # Handle different tensor shapes
        if len(array.shape) == 4:  # (1, 1, 256, 256) or (1, 3, 256, 256)
            if array.shape[1] == 3:  # RGB
                array = array[0]  # Remove batch dimension, keep (3, 256, 256)
                is_rgb = True
            else:
                array = array[0, 0]  # Remove batch and channel dimensions
        elif len(array.shape) == 3:
            if array.shape[0] == 1:  # (1, 256, 256)
                array = array[0]  # Remove channel dimension
            elif array.shape[0] == 3:  # (3, 256, 256) - RGB image
                # Keep RGB for photography, convert to grayscale for others
                if 'photography' in domain:
                    is_rgb = True
                    # array stays as (3, 256, 256)
                else:
                    array = array.mean(axis=0)  # Convert to grayscale
            else:  # (256, 256, 1) or similar
                array = array.squeeze()
        elif len(array.shape) == 2:  # (256, 256)
            pass  # Already correct shape
        else:
            print(f"Unexpected array shape: {array.shape} for {image_path}")
            return None, False

        # Check if image is in [-1,1] range and denormalize if needed
        array = denormalize_from_model_space(array, domain)

        return array, is_rgb
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None, False

def get_domain_ranges():
    """Get the domain-specific normalization ranges from preprocessing."""
    # These are the ranges defined in process_tiles_pipeline.py line 94-98
    return {
        'astronomy': {'min': -65.00, 'max': 385.00},
        'microscopy': {'min': 0.00, 'max': 65535.00},
        'photography': {'min': 0.00, 'max': 15871.00}
    }

def denormalize_from_model_space(image, domain):
    """
    Denormalize image from [-1,1] model space to domain-specific physical units.
    
    This follows the same logic as sample_noisy_pt_lle_PGguidance.py denormalize_to_physical:
    1. [-1,1] → [0,1]: (image + 1.0) / 2.0
    2. [0,1] → [domain_min, domain_max]: tensor_norm * (domain_max - domain_min) + domain_min
    
    Args:
        image: numpy array potentially in [-1,1] range
        domain: domain name for range lookup
        
    Returns:
        numpy array in domain-specific physical units
    """
    if image is None:
        return None
    
    # Get domain-specific range
    domain_ranges = get_domain_ranges()
    
    # Map photography_sony and photography_fuji to photography
    domain_key = domain
    if 'photography' in domain:
        domain_key = 'photography'
    
    if domain_key not in domain_ranges:
        print(f"Warning: Unknown domain {domain}, returning image as-is")
        return image
    
    domain_range = domain_ranges[domain_key]
    domain_min = domain_range['min']
    domain_max = domain_range['max']
    
    # Check if image is in [-1,1] range (allowing some tolerance)
    img_min = float(np.min(image))
    img_max = float(np.max(image))
    
    # Check if values are in [-1,1] range (±0.1 tolerance)
    is_normalized = (img_min >= -1.1 and img_max <= 1.1)
    
    if is_normalized:
        # Denormalize from [-1,1] to domain range
        # Step 1: [-1,1] → [0,1]
        image_norm = (image + 1.0) / 2.0
        image_norm = np.clip(image_norm, 0, 1)
        
        # Step 2: [0,1] → [domain_min, domain_max]
        image_phys = image_norm * (domain_max - domain_min) + domain_min
        
        return image_phys
    else:
        # Image is already in physical units
        return image

def convert_brightness_to_domain_range(image, domain):
    """
    Convert brightness-scaled images to original domain pixel ranges.
    
    The stored .pt files are in brightness space [0, s] where s = domain_max - domain_min.
    To get back to original domain range, we just add domain_min.
    
    - Astronomy: [0, 450] + (-65) → [-65, 385]
    - Microscopy: [0, 65535] + 0 → [0, 65535]
    - Photography: [0, 15871] + 0 → [0, 15871]
    """
    if image is None:
        return None
    
    # Get domain-specific range
    domain_ranges = get_domain_ranges()
    
    # Map photography_sony and photography_fuji to photography
    domain_key = domain
    if 'photography' in domain:
        domain_key = 'photography'
    
    if domain_key not in domain_ranges:
        print(f"Warning: Unknown domain {domain}, returning image as-is")
        return image
    
    domain_range = domain_ranges[domain_key]
    domain_min = domain_range['min']
    
    # Simple shift: brightness [0, s] + domain_min → [domain_min, domain_max]
    return image + domain_min

def find_cross_domain_example(domain, single_domain_example):
    """Find the corresponding cross-domain example for a single-domain example."""
    # Extract the base tile identifier from the single-domain example name
    # Format: example_XX_domain_base_tile_id
    parts = single_domain_example.split('_')
    if len(parts) >= 4:
        # Find the base tile identifier (everything after the domain)
        domain_idx = None
        for i, part in enumerate(parts):
            if part in ['astronomy', 'microscopy', 'photography']:
                domain_idx = i
                break

        if domain_idx is not None:
            base_identifier = '_'.join(parts[domain_idx + 1:])

            # Search for cross-domain example with the same base identifier
            cross_domain_dir = None
            if domain == 'astronomy':
                cross_domain_dir = Path('/home/jilab/Jae/results/cross_domain_inference_all_tiles/astronomy_cross_domain')
            elif domain == 'microscopy':
                cross_domain_dir = Path('/home/jilab/Jae/results/cross_domain_inference_all_tiles/microscopy_cross_domain')
            elif domain in ['photography_sony', 'photography_fuji']:
                cross_domain_dir = Path('/home/jilab/Jae/results/cross_domain_inference_all_tiles/photography_cross_domain')

            if cross_domain_dir and cross_domain_dir.exists():
                for item in cross_domain_dir.iterdir():
                    if item.is_dir() and base_identifier in item.name:
                        return item.name

    return None

def get_s_parameter(source_path):
    """Get the s parameter from results.json."""
    results_file = source_path / 'results.json'
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            pg_params = data.get('pg_guidance_params', {})
            return pg_params.get('s', 1.0)
        except:
            pass
    return 1.0

def get_image_for_method(domain, example_dir, method):
    """Get image tensor for a specific method from the appropriate source."""
    tile_path = Path(example_dir)

    # Determine the correct source directory based on method type
    if 'cross' in method:
        # Cross-domain methods should come from cross_domain directories
        # Find the corresponding cross-domain example
        cross_example = find_cross_domain_example(domain, tile_path.name)
        if not cross_example:
            print(f"Warning: No cross-domain example found for {domain}/{tile_path.name}")
            return None

        if domain == 'astronomy':
            source_path = Path('/home/jilab/Jae/results/cross_domain_inference_all_tiles/astronomy_cross_domain') / cross_example
        elif domain == 'microscopy':
            source_path = Path('/home/jilab/Jae/results/cross_domain_inference_all_tiles/microscopy_cross_domain') / cross_example
        elif domain in ['photography_sony', 'photography_fuji']:
            source_path = Path('/home/jilab/Jae/results/cross_domain_inference_all_tiles/photography_cross_domain') / cross_example
        else:
            return None
    else:
        # Single-domain methods should come from optimized directories
        if domain == 'astronomy':
            source_path = Path('/home/jilab/Jae/results/optimized_inference_all_tiles/astronomy_optimized') / tile_path.name
        elif domain == 'microscopy':
            source_path = Path('/home/jilab/Jae/results/optimized_inference_all_tiles/microscopy_optimized') / tile_path.name
        elif domain == 'photography_sony':
            source_path = Path('/home/jilab/Jae/results/optimized_inference_all_tiles/photography_sony_optimized') / tile_path.name
        elif domain == 'photography_fuji':
            source_path = Path('/home/jilab/Jae/results/optimized_inference_all_tiles/photography_fuji_optimized') / tile_path.name
        else:
            return None

    # Map method names to filenames
    method_to_original_filename = {
        'noisy': 'noisy.pt',
        'clean': 'clean.pt',
        'exposure_scaled': 'restored_exposure_scaled.pt',
        'gaussian_x0': 'restored_gaussian_x0.pt',
        'pg_x0_single': 'restored_pg_x0.pt',
        'gaussian_x0_cross': 'restored_gaussian_x0_cross.pt',
        'pg_x0_cross': 'restored_pg_x0_cross.pt'
    }

    if method in method_to_original_filename:
        original_image_path = source_path / method_to_original_filename[method]
        image, is_rgb = load_image_tensor(original_image_path, domain=domain)
        
        # Image is now in domain-specific physical units (already denormalized in load_image_tensor)
        return image, is_rgb

    return None, False

def extract_pixel_range(domain, tile_path):
    """Extract pixel range from results.json."""
    results_file = tile_path / 'results.json'
    if not results_file.exists():
        return None

    try:
        with open(results_file, 'r') as f:
            data = json.load(f)

        brightness = data.get('brightness_analysis', {})
        return {
            'min': brightness.get('min', 0),
            'max': brightness.get('max', 1),
            'mean': brightness.get('mean', 0),
            'std': brightness.get('std', 0)
        }
    except:
        return None

def format_pixel_range(pixel_range):
    """Format pixel range for display."""
    return f"[{pixel_range['min']:.0f}, {pixel_range['max']:.0f}]"

def format_metrics(metrics_dict, methods_to_show):
    """Format metrics string for display."""
    lines = []
    for method in methods_to_show:
        if method in metrics_dict:
            m = metrics_dict[method]
            line = f"{method.replace('_', '-').replace('x0', 'x0')}: "
            line += f"PSNR={m.get('psnr', 0):.1f}, SSIM={m.get('ssim', 0):.3f}, "
            line += f"LPIPS={m.get('lpips', 0):.3f}, NIQE={m.get('niqe', 'N/A')}"
            lines.append(line)
    return '\n'.join(lines)

def normalize_image_to_domain_range(image, domain_range):
    """Normalize image to domain-specific range."""
    if image is None or domain_range is None:
        return image

    min_val, max_val = domain_range['min'], domain_range['max']
    img_min, img_max = image.min(), image.max()

    if img_max > img_min:
        normalized = (image - img_min) / (img_max - img_min)
        return normalized * (max_val - min_val) + min_val

    return image

def create_domain_subplot(ax, domain, example_dir, methods, metrics_dict=None, scale_mode='pg_x0_cross'):
    """Create subplot for a single domain.
    
    Args:
        scale_mode: 'pg_x0_cross' or 'clean' - determines which image's range to use for scaling
    """
    tile_path = Path(example_dir)
    images = []
    pixel_ranges = []
    is_rgb_flags = []

    # First, load all images
    for method in methods:
        img, is_rgb = get_image_for_method(domain, tile_path, method)
        if img is not None:
            images.append(img)
            is_rgb_flags.append(is_rgb)
            pixel_ranges.append({
                'min': float(img.min()),
                'max': float(img.max())
            })
            print(f"  {method}: range [{pixel_ranges[-1]['min']:.2f}, {pixel_ranges[-1]['max']:.2f}], RGB={is_rgb}")
        else:
            # Create empty placeholder
            images.append(np.zeros((256, 256)))
            is_rgb_flags.append(False)
            pixel_ranges.append({'min': 0, 'max': 0})
            print(f"  {method}: no image found")

    # Get the dynamic range based on scale_mode
    if scale_mode == 'clean':
        # Use clean image range (index 1 in methods list)
        clean_idx = 1  # clean is second method
        if len(images) > clean_idx and images[clean_idx] is not None:
            vmin = float(images[clean_idx].min())
            vmax = float(images[clean_idx].max())
        else:
            # Fallback: use the range across all images
            all_mins = [r['min'] for r in pixel_ranges if r['min'] != 0 or r['max'] != 0]
            all_maxs = [r['max'] for r in pixel_ranges if r['min'] != 0 or r['max'] != 0]
            vmin = min(all_mins) if all_mins else 0
            vmax = max(all_maxs) if all_maxs else 1
    else:  # pg_x0_cross (default)
        # Get the dynamic range from PG x0-cross (last method)
        pg_x0_cross_idx = len(methods) - 1  # pg_x0_cross is the last method
        if len(images) > pg_x0_cross_idx and images[pg_x0_cross_idx] is not None:
            vmin = float(images[pg_x0_cross_idx].min())
            vmax = float(images[pg_x0_cross_idx].max())
        else:
            # Fallback: use the range across all images
            all_mins = [r['min'] for r in pixel_ranges if r['min'] != 0 or r['max'] != 0]
            all_maxs = [r['max'] for r in pixel_ranges if r['min'] != 0 or r['max'] != 0]
            vmin = min(all_mins) if all_mins else 0
            vmax = max(all_maxs) if all_maxs else 1

    # Check if any image is RGB
    has_rgb = any(is_rgb_flags)
    
    # Create composite image
    n_methods = len(methods)
    if has_rgb:
        composite_img = np.zeros((256, 256 * n_methods, 3))
    else:
        composite_img = np.zeros((256, 256 * n_methods))

    for i, img in enumerate(images):
        # Handle grayscale or RGB
        if len(img.shape) == 3 and img.shape[0] == 3:  # RGB (3, 256, 256)
            # Convert CHW to HWC for display
            img_display = np.transpose(img, (1, 2, 0))
            # Normalize RGB to [0,1] for display
            img_display = (img_display - vmin) / (vmax - vmin + 1e-8)
            img_display = np.clip(img_display, 0, 1)
            composite_img[:, i*256:(i+1)*256, :] = img_display
        elif len(img.shape) == 2:  # Grayscale (256, 256)
            composite_img[:, i*256:(i+1)*256] = img
        else:
            # Handle other shapes
            composite_img[:, i*256:(i+1)*256] = img if len(img.shape) == 2 else img.mean(axis=0)

    # Display with appropriate colormap
    if has_rgb:
        ax.imshow(composite_img)
    else:
        ax.imshow(composite_img, cmap='gray', vmin=vmin, vmax=vmax)
    
    # Set axis limits to accommodate text below images
    ax.set_xlim(0, composite_img.shape[1])
    ax.set_ylim(-90, composite_img.shape[0])
    ax.axis('off')

    # Add pixel range labels and metrics grouped together - all aligned at same vertical position
    # All boxes start at y=-30 and use white background
    for i, (method, pixel_range) in enumerate(zip(methods, pixel_ranges)):
        range_text = format_pixel_range(pixel_range)
        
        # Handle noisy panels: show pixel range + PSNR only
        if method == 'noisy' and metrics_dict and method in metrics_dict:
            metrics = metrics_dict[method]
            if metrics:
                psnr_val = metrics.get('psnr', 0)
                noisy_text = f"{range_text}\nPSNR={psnr_val:.3f}"
                ax.text(i*256 + 128, -30, noisy_text, ha='center', va='top', fontsize=7,
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
        # Handle clean panels: show only pixel range
        elif method == 'clean':
            ax.text(i*256 + 128, -30, range_text, ha='center', va='top', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
        # Handle other panels: show full metrics
        elif metrics_dict and method in metrics_dict:
            metrics = metrics_dict[method]
            if metrics:
                # Combine pixel range and metrics into one panel, each metric on its own row
                psnr_val = metrics.get('psnr', 0)
                ssim_val = metrics.get('ssim', 0)
                lpips_val = metrics.get('lpips', 0)
                niqe_val = metrics.get('niqe', 'N/A')
                
                # Format NIQE: if it's a number, format to 3 decimals, otherwise show as-is
                if isinstance(niqe_val, (int, float)):
                    niqe_str = f"{niqe_val:.3f}"
                else:
                    niqe_str = str(niqe_val)
                
                metric_text = f"{range_text}\nPSNR={psnr_val:.3f}\nSSIM={ssim_val:.3f}\nLPIPS={lpips_val:.3f}\nNIQE={niqe_str}"
                ax.text(i*256 + 128, -30, metric_text, ha='center', va='top', fontsize=7,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
        else:
            # Fallback: just show pixel range
            ax.text(i*256 + 128, -30, range_text, ha='center', va='top', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))

    return pixel_ranges, images, vmin, vmax

def normalize_image_to_domain_range(image, domain_range):
    """Normalize image to domain-specific range."""
    if image is None or domain_range is None:
        return image

    min_val, max_val = domain_range['min'], domain_range['max']
    img_min, img_max = image.min(), image.max()

    if img_max > img_min:
        normalized = (image - img_min) / (img_max - img_min)
        return normalized * (max_val - min_val) + min_val

    return image

def get_domain_scaling_factor(domain, tile_path):
    """Get the domain scaling factor from preprocessing (s parameter)."""
    results_file = tile_path / 'results.json'
    if not results_file.exists():
        # Fallback to default ranges if results file not found
        return get_domain_pixel_range(domain)

    try:
        with open(results_file, 'r') as f:
            data = json.load(f)

        pg_params = data.get('pg_guidance_params', {})
        s = pg_params.get('s', 1.0)

        # Get brightness analysis for the actual pixel range
        brightness = data.get('brightness_analysis', {})
        pixel_min = brightness.get('min', 0)
        pixel_max = brightness.get('max', 1)

        return {'scale_factor': s, 'pixel_min': pixel_min, 'pixel_max': pixel_max}
    except:
        # Fallback to default ranges
        return get_domain_pixel_range(domain)

def get_domain_pixel_range(domain):
    """Get the typical pixel range for a domain (fallback)."""
    # Domain-specific typical ranges based on the data
    domain_ranges = {
        'astronomy': {'min': 0.5, 'max': 225.5},
        'microscopy': {'min': 0.001, 'max': 0.05},
        'photography_sony': {'min': 0.0, 'max': 0.03},
        'photography_fuji': {'min': 0.0, 'max': 0.01}
    }
    return domain_ranges.get(domain, {'min': 0, 'max': 1})

def extract_metrics(domain, example_dir, method):
    """Extract metrics for a specific method from the appropriate results file."""
    # Determine the correct source directory based on method type
    if 'cross' in method:
        # Cross-domain methods should come from cross_domain directories
        # Find the corresponding cross-domain example
        cross_example = find_cross_domain_example(domain, Path(example_dir).name)
        if not cross_example:
            print(f"Warning: No cross-domain example found for {domain}/{Path(example_dir).name}")
            return None

        if domain == 'astronomy':
            source_path = Path('/home/jilab/Jae/results/cross_domain_inference_all_tiles/astronomy_cross_domain') / cross_example
        elif domain == 'microscopy':
            source_path = Path('/home/jilab/Jae/results/cross_domain_inference_all_tiles/microscopy_cross_domain') / cross_example
        elif domain in ['photography_sony', 'photography_fuji']:
            source_path = Path('/home/jilab/Jae/results/cross_domain_inference_all_tiles/photography_cross_domain') / cross_example
        else:
            return None
    else:
        # Single-domain methods should come from optimized directories
        if domain == 'astronomy':
            source_path = Path('/home/jilab/Jae/results/optimized_inference_all_tiles/astronomy_optimized') / Path(example_dir).name
        elif domain == 'microscopy':
            source_path = Path('/home/jilab/Jae/results/optimized_inference_all_tiles/microscopy_optimized') / Path(example_dir).name
        elif domain == 'photography_sony':
            source_path = Path('/home/jilab/Jae/results/optimized_inference_all_tiles/photography_sony_optimized') / Path(example_dir).name
        elif domain == 'photography_fuji':
            source_path = Path('/home/jilab/Jae/results/optimized_inference_all_tiles/photography_fuji_optimized') / Path(example_dir).name
        else:
            return None

    results_file = source_path / 'results.json'
    if not results_file.exists():
        return None

    try:
        with open(results_file, 'r') as f:
            data = json.load(f)

        comprehensive_metrics = data.get('comprehensive_metrics', {})

        # Map method names to the keys in the results
        method_key_map = {
            'noisy': 'noisy',
            'clean': 'clean',
            'exposure_scaled': 'exposure_scaled',
            'gaussian_x0': 'gaussian_x0',
            'pg_x0_single': 'pg_x0',
            'gaussian_x0_cross': 'gaussian_x0_cross',
            'pg_x0_cross': 'pg_x0_cross'
        }

        if method in method_key_map:
            key = method_key_map[method]
            return comprehensive_metrics.get(key)

    except Exception as e:
        print(f"Error extracting metrics for {method}: {e}")

    return None

def main():
    # Define representative examples for each domain
    examples = {
        'astronomy': [
            'example_00_astronomy_j8g6z3jdq_g800l_sci_tile_0071',
            'example_01_astronomy_j8hpakg9q_g800l_sci_tile_0044',
            'example_02_astronomy_j8hqe3c6q_g800l_sci_tile_0005'
        ],
        'microscopy': [
            'example_00_microscopy_ER_Cell_002_RawGTSIMData_level_06_tile_0002',
            'example_01_microscopy_F-actin_Cell_024_RawSIMData_gt_tile_0006',
            'example_02_microscopy_ER_Cell_058_RawGTSIMData_level_05_tile_0001'
        ],
        'photography_sony': [
            'example_00_photography_sony_00135_00_0.1s_tile_0005',
            'example_02_photography_sony_00135_00_0.1s_tile_0034'
        ],
        'photography_fuji': [
            'example_00_photography_fuji_00017_00_0.1s_tile_0009',
            'example_01_photography_fuji_20184_00_0.033s_tile_0011',
            'example_02_photography_fuji_00077_00_0.04s_tile_0022'
        ]
    }

    # Define the layout
    domains = list(examples.keys())
    methods = ['noisy', 'clean', 'exposure_scaled', 'gaussian_x0_cross', 'pg_x0_single', 'pg_x0_cross']
    method_names = ['Noisy Input', 'Clean Reference', 'Exposure Scaled', 'Gaussian x0-cross', 'PG x0-single', 'PG x0-cross']

    # Create figure with optimized vertical spacing (increased height for metrics)
    fig_height = len(domains) * 3.5  # 3.5 inches per domain (reduced for tighter layout)
    fig_width = len(methods) * 2   # 2 inches per method

    fig, axes = plt.subplots(len(domains), 1, figsize=(fig_width, fig_height))
    if len(domains) == 1:
        axes = [axes]

    # Process each domain
    for i, domain in enumerate(domains):
        print(f"Processing {domain}...")

        # Use the best example for each domain based on PG x0-cross metrics
        if domain == 'astronomy':
            example_name = examples[domain][1]  # example_01 (best PSNR)
        elif domain == 'microscopy':
            example_name = examples[domain][0]  # example_00 (best PSNR)
        elif domain == 'photography_sony':
            example_name = examples[domain][0]  # example_00 (best PSNR)
        elif domain == 'photography_fuji':
            example_name = examples[domain][0]  # example_00 (best PSNR)
        else:
            example_name = examples[domain][0]  # Use example_00
        
        example_path = f'/home/jilab/Jae/results/optimized_inference_all_tiles/{domain}_optimized/{example_name}'

        # Extract metrics for all methods
        metrics_dict = {}
        for method in methods:
            metrics = extract_metrics(domain, example_path, method)
            if metrics:
                metrics_dict[method] = metrics

        # Create subplot for this domain (scaled by PG x0-cross dynamic range)
        pixel_ranges, images, vmin, vmax = create_domain_subplot(axes[i], domain, example_path, methods, metrics_dict=metrics_dict, scale_mode='pg_x0_cross')

        print(f"  Visualization scaled to PG x0-cross range: [{vmin:.2f}, {vmax:.2f}]")

        # Set domain title with full domain name higher up
        domain_display_name = {
            'astronomy': 'Astronomy',
            'microscopy': 'Microscopy',
            'photography_sony': 'Photography (Sony)',
            'photography_fuji': 'Photography (Fuji)'
        }.get(domain, domain.replace('_', ' ').title())

        axes[i].set_title(f"{domain_display_name}", fontsize=12, fontweight='bold', y=1.02)

    # Overall title
    fig.suptitle('Low-light imaging restoration comparison',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.925, bottom=0.15, hspace=0.3)
    
    # Add column labels aligned to panel centers (after tight_layout to get actual positions)
    # Panel centers are at i*256 + 128, total width is len(methods)*256
    total_width = len(methods) * 256
    for idx, domain in enumerate(domains):
        ax = axes[idx]
        # Get the axis position in figure coordinates
        ax_pos = ax.get_position()
        ax_left = ax_pos.x0
        ax_width = ax_pos.width
        
        # Calculate label positions for each method in figure coordinates
        for i, method_name in enumerate(method_names):
            # Panel center in axis coordinates
            panel_center = i * 256 + 128
            # Convert to figure coordinates
            x_position = ax_left + (panel_center / total_width) * ax_width
            # Place label above the first domain's axis with more space
            if idx == 0:
                fig.text(x_position, ax_pos.y1 + 0.03, method_name,
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Save the figure
    output_path = 'comparison_visualization_domain_scaled.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")

    # Also save a high-resolution version
    plt.savefig('comparison_visualization_domain_scaled_high_res.png', dpi=300, bbox_inches='tight')
    print("High resolution version saved to comparison_visualization_domain_scaled_high_res.png")

    # Create version scaled to clean range
    print("\nCreating clean-range scaled version...")
    fig_clean, axes_clean = plt.subplots(len(domains), 1, figsize=(fig_width, fig_height))
    if len(domains) == 1:
        axes_clean = [axes_clean]

    # Process each domain with clean scaling
    for i, domain in enumerate(domains):
        print(f"Processing {domain} (clean-scaled)...")
        # Use the same best example selection as above
        if domain == 'astronomy':
            example_name = examples[domain][1]  # example_01 (best PSNR)
        elif domain == 'microscopy':
            example_name = examples[domain][0]  # example_00 (best PSNR)
        elif domain == 'photography_sony':
            example_name = examples[domain][0]  # example_00 (best PSNR)
        elif domain == 'photography_fuji':
            example_name = examples[domain][0]  # example_00 (best PSNR)
        else:
            example_name = examples[domain][0]  # Use example_00
        
        example_path = f'/home/jilab/Jae/results/optimized_inference_all_tiles/{domain}_optimized/{example_name}'

        # Extract metrics for all methods
        metrics_dict = {}
        for method in methods:
            metrics = extract_metrics(domain, example_path, method)
            if metrics:
                metrics_dict[method] = metrics

        # Create subplot scaled to clean range
        pixel_ranges, images, vmin, vmax = create_domain_subplot(axes_clean[i], domain, example_path, methods, metrics_dict=metrics_dict, scale_mode='clean')

        print(f"  Visualization scaled to clean range: [{vmin:.2f}, {vmax:.2f}]")

        # Set domain title
        domain_display_name = {
            'astronomy': 'Astronomy',
            'microscopy': 'Microscopy',
            'photography_sony': 'Photography (Sony)',
            'photography_fuji': 'Photography (Fuji)'
        }.get(domain, domain.replace('_', ' ').title())

        axes_clean[i].set_title(f"{domain_display_name}", fontsize=12, fontweight='bold', y=1.02)

    # Overall title
    fig_clean.suptitle('Low-light imaging restoration comparison\n(Clean Reference Scaled)',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15, hspace=0.3)
    
    # Add column labels aligned to panel centers (after tight_layout to get actual positions)
    total_width = len(methods) * 256
    for idx, domain in enumerate(domains):
        ax = axes_clean[idx]
        # Get the axis position in figure coordinates
        ax_pos = ax.get_position()
        ax_left = ax_pos.x0
        ax_width = ax_pos.width
        
        # Calculate label positions for each method in figure coordinates
        for i, method_name in enumerate(method_names):
            # Panel center in axis coordinates
            panel_center = i * 256 + 128
            # Convert to figure coordinates
            x_position = ax_left + (panel_center / total_width) * ax_width
            # Place label above the first domain's axis with more space
            if idx == 0:
                fig_clean.text(x_position, ax_pos.y1 + 0.03, method_name,
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Save clean-scaled version
    plt.savefig('comparison_visualization_domain_scaled_clean_range.png', dpi=150, bbox_inches='tight')
    print("Clean-range version saved to comparison_visualization_domain_scaled_clean_range.png")
    
    plt.savefig('comparison_visualization_domain_scaled_clean_range_high_res.png', dpi=300, bbox_inches='tight')
    print("High resolution clean-range version saved")

    plt.show()

if __name__ == "__main__":
    main()
