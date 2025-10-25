#!/usr/bin/env python3
"""
Create Astronomy V2 Dataset
===========================

This script creates a new astronomy_v2 dataset with:
1. Copies astronomy folder to astronomy_v2
2. Creates noisy folder structure
3. Identifies low exposure files from coordinate groups
4. Moves corresponding tiles to noisy folder
5. Updates metadata with exposure information and noisy/clean splits

Usage:
    python create_astronomy_v2_dataset.py

Author: AI Assistant
Date: 2024-10-25
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

class AstronomyV2DatasetCreator:
    """Create astronomy_v2 dataset with exposure-based splits."""
    
    def __init__(self, base_dir, comprehensive_analysis_file, exposure_analysis_file):
        self.base_dir = Path(base_dir)
        self.astronomy_dir = self.base_dir / 'dataset' / 'processed' / 'pt_tiles' / 'astronomy'
        self.astronomy_v2_dir = self.base_dir / 'dataset' / 'processed' / 'pt_tiles' / 'astronomy_v2'
        self.comprehensive_analysis_file = comprehensive_analysis_file
        self.exposure_analysis_file = exposure_analysis_file
        
    def load_analysis_data(self):
        """Load comprehensive and exposure analysis data."""
        print("Loading analysis data...")
        
        with open(self.comprehensive_analysis_file, 'r') as f:
            self.comprehensive_data = json.load(f)
        
        with open(self.exposure_analysis_file, 'r') as f:
            self.exposure_data = json.load(f)
        
        print(f"Loaded data for {len(self.comprehensive_data['individual_files'])} files")
        
    def copy_astronomy_folder(self):
        """Copy astronomy folder to astronomy_v2."""
        print(f"\n1. Copying astronomy folder to astronomy_v2...")
        
        if self.astronomy_v2_dir.exists():
            print(f"   astronomy_v2 already exists. Removing it first...")
            shutil.rmtree(self.astronomy_v2_dir)
        
        print(f"   Copying from {self.astronomy_dir} to {self.astronomy_v2_dir}")
        shutil.copytree(self.astronomy_dir, self.astronomy_v2_dir)
        print(f"   ✓ Copy complete")
        
    def create_noisy_folder_structure(self):
        """Create noisy folder structure."""
        print(f"\n2. Creating noisy folder structure...")
        
        # Create noisy directory
        noisy_dir = self.astronomy_v2_dir / 'noisy'
        noisy_dir.mkdir(exist_ok=True)
        print(f"   ✓ Created {noisy_dir}")
        
        # Create clean directory (if it doesn't exist)
        clean_dir = self.astronomy_v2_dir / 'clean'
        if not clean_dir.exists():
            clean_dir.mkdir(exist_ok=True)
            print(f"   ✓ Created {clean_dir}")
        
    def identify_low_exposure_files(self):
        """Identify low exposure files from coordinate groups."""
        print(f"\n3. Identifying low exposure files from coordinate groups...")
        
        low_exposure_files = []
        
        # Get coordinate groups with exposure analysis
        for group in self.exposure_data['coordinate_groups_with_exposure_analysis']:
            exp_analysis = group['exposure_analysis']
            
            # Get low exposure files
            for file_info in exp_analysis['low_exposure_files']:
                low_exposure_files.append({
                    'filename': file_info['filename'],
                    'exposure_time': file_info['exposure_time_seconds'],
                    'group_id': group['group_id'],
                    'coordinates': file_info['coordinates']
                })
        
        print(f"   ✓ Identified {len(low_exposure_files)} low exposure files")
        
        # Show some examples
        if low_exposure_files:
            print(f"\n   Example low exposure files:")
            for file_info in low_exposure_files[:5]:
                print(f"     - {file_info['filename']}: {file_info['exposure_time']:.1f}s (Group {file_info['group_id']})")
        
        return low_exposure_files
    
    def move_low_exposure_tiles(self, low_exposure_files):
        """Move low exposure tiles to noisy folder."""
        print(f"\n4. Moving low exposure tiles to noisy folder...")
        
        noisy_dir = self.astronomy_v2_dir / 'noisy'
        clean_dir = self.astronomy_v2_dir / 'clean'
        moved_count = 0
        not_found_count = 0
        
        if not clean_dir.exists():
            print(f"   ✗ Clean directory not found: {clean_dir}")
            return moved_count, not_found_count
        
        # Get all tile files from clean directory
        tile_files = {}
        for tile_file in clean_dir.glob('*.pt'):
            # Extract base filename from tile (remove astronomy_ prefix and _tile_X.pt suffix)
            base_name = tile_file.stem.replace('astronomy_', '').rsplit('_tile_', 1)[0]
            if base_name not in tile_files:
                tile_files[base_name] = []
            tile_files[base_name].append(tile_file)
        
        print(f"   Found {len(tile_files)} unique tile sets in clean directory")
        
        # Create mapping of low exposure basenames
        low_exposure_basenames = set()
        for file_info in low_exposure_files:
            filename = file_info['filename']
            # Remove .fits extension
            base_name = filename.replace('.fits', '')
            low_exposure_basenames.add(base_name)
        
        # Move tiles for low exposure files
        for file_info in low_exposure_files:
            filename = file_info['filename']
            # Remove .fits extension to match tile naming
            base_name = filename.replace('.fits', '')
            
            if base_name in tile_files:
                for tile_file in tile_files[base_name]:
                    dest_file = noisy_dir / tile_file.name
                    shutil.move(str(tile_file), str(dest_file))
                    moved_count += 1
            else:
                not_found_count += 1
                if not_found_count <= 5:  # Only show first 5 warnings
                    print(f"   ⚠ Tile not found for: {filename} (searched for: {base_name})")
        
        print(f"   ✓ Moved {moved_count} tile files to noisy")
        if not_found_count > 0:
            print(f"   ⚠ {not_found_count} low exposure files had no corresponding tiles")
        
        return moved_count, not_found_count
    
    def copy_metadata(self):
        """Copy metadata file to astronomy_v2."""
        print(f"\n5. Copying metadata file...")
        
        metadata_source = self.base_dir / 'dataset' / 'processed' / 'metadata_astronomy_incremental.json'
        metadata_dest = self.astronomy_v2_dir / 'metadata_astronomy_incremental.json'
        
        if not metadata_source.exists():
            print(f"   ✗ Source metadata not found: {metadata_source}")
            return False
        
        shutil.copy2(metadata_source, metadata_dest)
        print(f"   ✓ Copied metadata to {metadata_dest}")
        
        return True
    
    def update_metadata(self, low_exposure_files):
        """Update metadata with exposure information and noisy/clean splits."""
        print(f"\n6. Updating metadata with exposure information and splits...")
        
        metadata_file = self.astronomy_v2_dir / 'metadata_astronomy_incremental.json'
        
        if not metadata_file.exists():
            print(f"   ✗ Metadata file not found: {metadata_file}")
            return False
        
        # Load existing metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Create mapping of low exposure files
        low_exposure_set = {f['filename'] for f in low_exposure_files}
        
        # Create filename to exposure time mapping
        exposure_map = {}
        for exp_info in self.comprehensive_data['individual_files']:
            exposure_map[exp_info['filename']] = exp_info.get('exposure_time_seconds', None)
        
        # Create filename to WCS info mapping
        wcs_map = {}
        for file_info in self.comprehensive_data['individual_files']:
            wcs_map[file_info['filename']] = {
                'coordinates': file_info.get('coordinates', {}),
                'wcs_valid': file_info.get('wcs_valid', False),
                'pixel_scale': file_info.get('wcs_details', {}).get('pixel_scale_arcsec')
            }
        
        # Update metadata entries
        updated_count = 0
        
        # Handle metadata structure - it has a 'tiles' array
        if 'tiles' in metadata:
            items = metadata['tiles']
        elif isinstance(metadata, list):
            items = metadata
        elif isinstance(metadata, dict) and 'items' in metadata:
            items = metadata['items']
        else:
            items = [metadata]
        
        for item in items:
            # Extract filename from tile_id or pt_path
            tile_id = item.get('tile_id', '')
            pt_path = item.get('pt_path', '')
            
            # Extract original filename from tile_id or pt_path
            # Format: astronomy_j6fl7xoyq_detection_sci_tile_0000
            if tile_id:
                # Remove astronomy_ prefix and _tile_X suffix
                base_name = tile_id.replace('astronomy_', '').rsplit('_tile_', 1)[0]
                filename = base_name + '.fits'
            elif pt_path:
                # Extract from path
                basename = os.path.basename(pt_path)
                base_name = basename.replace('astronomy_', '').rsplit('_tile_', 1)[0]
                filename = base_name + '.fits'
            else:
                continue
            
            # Determine if this is a low exposure file
            is_low_exposure = filename in low_exposure_set
            
            # Add exposure information
            if filename in exposure_map:
                item['exposure_time_seconds'] = exposure_map[filename]
                
                # Add to exposure statistics
                if exposure_map[filename]:
                    item['exposure_time_hours'] = exposure_map[filename] / 3600.0
            
            # Add WCS information
            if filename in wcs_map:
                item['wcs_info'] = wcs_map[filename]
            
            # Update split information
            item['split'] = 'noisy' if is_low_exposure else 'clean'
            item['is_low_exposure'] = is_low_exposure
            
            # Update data_type
            item['data_type'] = 'noisy' if is_low_exposure else 'clean'
            
            # Update pt_path to reflect new location
            if 'pt_path' in item:
                old_path = item['pt_path']
                if is_low_exposure:
                    new_path = old_path.replace('/astronomy/', '/astronomy_v2/').replace('/clean/', '/noisy/')
                else:
                    new_path = old_path.replace('/astronomy/', '/astronomy_v2/')
                item['pt_path'] = new_path
            
            updated_count += 1
        
        # Save updated metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✓ Updated {updated_count} metadata entries")
        print(f"   ✓ Added exposure information and split classification")
        
        return True
    
    def run(self):
        """Run the complete dataset creation process."""
        print("=== Creating Astronomy V2 Dataset ===")
        print(f"Base directory: {self.base_dir}")
        print(f"Astronomy directory: {self.astronomy_dir}")
        print(f"Astronomy V2 directory: {self.astronomy_v2_dir}")
        
        # Load analysis data
        self.load_analysis_data()
        
        # 1. Copy astronomy folder
        self.copy_astronomy_folder()
        
        # 2. Create noisy folder structure
        self.create_noisy_folder_structure()
        
        # 3. Identify low exposure files
        low_exposure_files = self.identify_low_exposure_files()
        
        # 4. Move low exposure tiles
        moved_count, not_found_count = self.move_low_exposure_tiles(low_exposure_files)
        
        # 5. Copy metadata
        self.copy_metadata()
        
        # 6. Update metadata
        self.update_metadata(low_exposure_files)
        
        print(f"\n=== Dataset Creation Complete ===")
        print(f"Low exposure files identified: {len(low_exposure_files)}")
        print(f"Tiles moved to noisy: {moved_count}")
        print(f"Files without tiles: {not_found_count}")
        print(f"Output directory: {self.astronomy_v2_dir}")

def main():
    parser = argparse.ArgumentParser(description='Create Astronomy V2 Dataset with Exposure-based Splits')
    parser.add_argument('--base-dir', '-b', 
                       default='/home/jilab/Jae',
                       help='Base directory containing dataset')
    parser.add_argument('--comprehensive-analysis', '-c',
                       default='/home/jilab/Jae/comprehensive_fits_analysis.json',
                       help='Comprehensive FITS analysis JSON file')
    parser.add_argument('--exposure-analysis', '-e',
                       default='/home/jilab/Jae/comprehensive_fits_analysis_exposure_analysis.json',
                       help='Exposure analysis JSON file')
    
    args = parser.parse_args()
    
    # Create dataset creator and run
    creator = AstronomyV2DatasetCreator(
        args.base_dir,
        args.comprehensive_analysis,
        args.exposure_analysis
    )
    creator.run()

if __name__ == "__main__":
    main()
