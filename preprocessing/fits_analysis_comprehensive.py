#!/usr/bin/env python3
"""
Comprehensive FITS Analysis Script
==================================

This script performs complete analysis of FITS files including:
1. Extraction of exposure time information from FITS headers
2. Extraction of WCS (World Coordinate System) information
3. Coordinate grouping and similarity analysis
4. Exposure time variation analysis within groups
5. Generation of comprehensive JSON output

Usage:
    python fits_analysis_comprehensive.py [--input-dir DIR] [--output-dir DIR] [--tolerance ARCMIN]

Author: AI Assistant
Date: 2024-10-25
"""

import os
import json
import glob
import argparse
import numpy as np
import math
from astropy.io import fits
from astropy.wcs import WCS
from collections import defaultdict
from pathlib import Path

class FITSAnalyzer:
    """Comprehensive FITS file analyzer."""
    
    def __init__(self, input_dir, output_dir, tolerance_arcmin=5.0):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.tolerance_arcmin = tolerance_arcmin
        self.tolerance_deg = tolerance_arcmin / 60.0
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def extract_exposure_info(self, fits_file):
        """Extract exposure time from FITS file."""
        try:
            filename = os.path.basename(fits_file)
            with fits.open(fits_file) as hdul:
                header = hdul[0].header
                
                # Try to get exposure time from common header keywords
                exposure_time = None
                exp_keywords = ['EXPTIME', 'EXPOSURE', 'EXPOSURE_TIME', 'TEXPTIME', 'EXPTIME1']
                
                for keyword in exp_keywords:
                    if keyword in header:
                        exposure_time = header[keyword]
                        break
                
                # If no standard exposure time found, try alternative keywords
                if exposure_time is None:
                    alt_keywords = ['DARKTIME', 'LIVETIME', 'ONTIME']
                    for keyword in alt_keywords:
                        if keyword in header:
                            exposure_time = header[keyword]
                            break
                
                # Convert to float if it's a number
                if exposure_time is not None:
                    try:
                        exposure_time = float(exposure_time)
                    except (ValueError, TypeError):
                        pass
                
                return {
                    'filename': filename,
                    'exposure_time': exposure_time
                }
                
        except Exception as e:
            return {
                'filename': os.path.basename(fits_file),
                'exposure_time': None,
                'error': str(e)
            }
    
    def extract_wcs_info(self, fits_file):
        """Extract WCS information from FITS file."""
        try:
            filename = os.path.basename(fits_file)
            with fits.open(fits_file) as hdul:
                header = hdul[0].header
                
                # Extract key WCS parameters
                wcs_info = {}
                
                # Reference pixel coordinates
                wcs_info['crpix1'] = header.get('CRPIX1', None)
                wcs_info['crpix2'] = header.get('CRPIX2', None)
                
                # Reference world coordinates (RA, Dec)
                wcs_info['crval1'] = header.get('CRVAL1', None)  # RA
                wcs_info['crval2'] = header.get('CRVAL2', None)  # Dec
                
                # Pixel scale (degrees per pixel)
                wcs_info['cdelt1'] = header.get('CDELT1', None)
                wcs_info['cdelt2'] = header.get('CDELT2', None)
                
                # CD matrix elements (alternative to CDELT)
                wcs_info['cd1_1'] = header.get('CD1_1', None)
                wcs_info['cd1_2'] = header.get('CD1_2', None)
                wcs_info['cd2_1'] = header.get('CD2_1', None)
                wcs_info['cd2_2'] = header.get('CD2_2', None)
                
                # Coordinate system type
                wcs_info['ctype1'] = header.get('CTYPE1', None)
                wcs_info['ctype2'] = header.get('CTYPE2', None)
                
                # Equinox and Epoch
                wcs_info['equinox'] = header.get('EQUINOX', None)
                wcs_info['epoch'] = header.get('EPOCH', None)
                
                # Image dimensions
                wcs_info['naxis1'] = header.get('NAXIS1', None)
                wcs_info['naxis2'] = header.get('NAXIS2', None)
                
                # Additional useful parameters
                wcs_info['ra_targ'] = header.get('RA_TARG', None)
                wcs_info['dec_targ'] = header.get('DEC_TARG', None)
                
                # Create a WCS object to get more detailed information
                try:
                    wcs_obj = WCS(header)
                    wcs_info['wcs_valid'] = True
                    wcs_info['wcs_type'] = str(type(wcs_obj))
                    
                    # Get pixel scale from WCS
                    if wcs_obj.pixel_scale_matrix is not None:
                        pixel_scale = np.sqrt(np.linalg.det(wcs_obj.pixel_scale_matrix))
                        wcs_info['pixel_scale_arcsec'] = pixel_scale * 3600  # Convert to arcseconds
                    else:
                        wcs_info['pixel_scale_arcsec'] = None
                        
                except Exception as e:
                    wcs_info['wcs_valid'] = False
                    wcs_info['wcs_error'] = str(e)
                    wcs_info['pixel_scale_arcsec'] = None
                
                wcs_info['filename'] = filename
                return wcs_info
                
        except Exception as e:
            return {
                'filename': os.path.basename(fits_file),
                'error': str(e),
                'wcs_valid': False
            }
    
    def angular_distance(self, ra1, dec1, ra2, dec2):
        """Calculate angular distance between two points on the sky in degrees."""
        # Convert to radians
        ra1_rad = math.radians(ra1)
        dec1_rad = math.radians(dec1)
        ra2_rad = math.radians(ra2)
        dec2_rad = math.radians(dec2)
        
        # Calculate angular distance using spherical geometry
        cos_d = (math.sin(dec1_rad) * math.sin(dec2_rad) + 
                 math.cos(dec1_rad) * math.cos(dec2_rad) * math.cos(ra1_rad - ra2_rad))
        
        # Handle numerical precision issues
        cos_d = max(-1.0, min(1.0, cos_d))
        
        return math.degrees(math.acos(cos_d))
    
    def find_coordinate_groups(self, wcs_data, exposure_data):
        """Find groups of images with similar coordinates."""
        # Create a mapping from filename to exposure time
        exposure_map = {}
        for exp_info in exposure_data:
            exposure_map[exp_info['filename']] = exp_info['exposure_time']
        
        groups = []
        used_indices = set()
        
        for i, file1 in enumerate(wcs_data):
            if i in used_indices:
                continue
                
            if not file1.get('wcs_valid', False):
                continue
                
            if file1.get('crval1') is None or file1.get('crval2') is None:
                continue
                
            # Start a new group with this file
            current_group = [{
                'filename': file1['filename'],
                'exposure_time': exposure_map.get(file1['filename'], None),
                'ra': file1['crval1'],
                'dec': file1['crval2']
            }]
            used_indices.add(i)
            
            # Find all files within the distance threshold
            for j, file2 in enumerate(wcs_data):
                if j in used_indices or j == i:
                    continue
                    
                if not file2.get('wcs_valid', False):
                    continue
                    
                if file2.get('crval1') is None or file2.get('crval2') is None:
                    continue
                
                # Calculate angular distance
                distance = self.angular_distance(
                    file1['crval1'], file1['crval2'],
                    file2['crval1'], file2['crval2']
                )
                
                if distance <= self.tolerance_deg:
                    current_group.append({
                        'filename': file2['filename'],
                        'exposure_time': exposure_map.get(file2['filename'], None),
                        'ra': file2['crval1'],
                        'dec': file2['crval2']
                    })
                    used_indices.add(j)
            
            # Add group (even if single file)
            groups.append({
                'center_ra': file1['crval1'],
                'center_dec': file1['crval2'],
                'count': len(current_group),
                'files': current_group,
                'max_distance_arcmin': self.tolerance_arcmin
            })
        
        return groups
    
    def analyze_exposure_variations(self, coordinate_groups):
        """Analyze exposure time variations within coordinate groups."""
        exposure_analysis = {}
        
        for i, group in enumerate(coordinate_groups):
            group_id = i + 1
            
            # Extract exposure times (filter out None values)
            exposure_times = [f['exposure_time'] for f in group['files'] if f['exposure_time'] is not None]
            
            if not exposure_times:
                exposure_analysis[group_id] = {
                    'group_id': group_id,
                    'center_coordinates': {
                        'ra_degrees': group['center_ra'],
                        'dec_degrees': group['center_dec']
                    },
                    'file_count': group['count'],
                    'exposure_analysis': {
                        'valid_exposure_count': 0,
                        'median_exposure_seconds': None,
                        'mean_exposure_seconds': None,
                        'min_exposure_seconds': None,
                        'max_exposure_seconds': None,
                        'exposure_range_seconds': None,
                        'low_exposure_files': [],
                        'high_exposure_files': [],
                        'exposure_variation_ratio': None
                    }
                }
                continue
            
            # Calculate statistics
            median_exp = np.median(exposure_times)
            mean_exp = np.mean(exposure_times)
            min_exp = min(exposure_times)
            max_exp = max(exposure_times)
            exp_range = max_exp - min_exp
            
            # Define low and high exposure thresholds
            # Low exposure: below median
            # High exposure: above median
            low_exposure_files = []
            high_exposure_files = []
            
            for file_info in group['files']:
                if file_info['exposure_time'] is not None:
                    if file_info['exposure_time'] < median_exp:
                        low_exposure_files.append({
                            'filename': file_info['filename'],
                            'exposure_time_seconds': file_info['exposure_time'],
                            'coordinates': {
                                'ra_degrees': file_info['ra'],
                                'dec_degrees': file_info['dec']
                            }
                        })
                    else:
                        high_exposure_files.append({
                            'filename': file_info['filename'],
                            'exposure_time_seconds': file_info['exposure_time'],
                            'coordinates': {
                                'ra_degrees': file_info['ra'],
                                'dec_degrees': file_info['dec']
                            }
                        })
            
            # Calculate variation ratio (max/min)
            variation_ratio = max_exp / min_exp if min_exp > 0 else None
            
            exposure_analysis[group_id] = {
                'group_id': group_id,
                'center_coordinates': {
                    'ra_degrees': group['center_ra'],
                    'dec_degrees': group['center_dec']
                },
                'file_count': group['count'],
                'exposure_analysis': {
                    'valid_exposure_count': len(exposure_times),
                    'median_exposure_seconds': float(median_exp),
                    'mean_exposure_seconds': float(mean_exp),
                    'min_exposure_seconds': float(min_exp),
                    'max_exposure_seconds': float(max_exp),
                    'exposure_range_seconds': float(exp_range),
                    'low_exposure_files': low_exposure_files,
                    'high_exposure_files': high_exposure_files,
                    'exposure_variation_ratio': float(variation_ratio) if variation_ratio else None
                }
            }
        
        return exposure_analysis
    
    def run_analysis(self):
        """Run the complete FITS analysis."""
        print("=== Comprehensive FITS Analysis ===")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Coordinate tolerance: {self.tolerance_arcmin} arcminutes")
        
        # Get all FITS files
        fits_files = glob.glob(os.path.join(self.input_dir, "*.fits"))
        print(f"Found {len(fits_files)} FITS files to process...")
        
        if not fits_files:
            print("No FITS files found in the specified directory!")
            return
        
        # Extract exposure information
        print("\n1. Extracting exposure time information...")
        exposure_data = []
        for fits_file in fits_files:
            exp_info = self.extract_exposure_info(fits_file)
            exposure_data.append(exp_info)
            if exp_info['exposure_time'] is not None:
                print(f"  {exp_info['filename']}: {exp_info['exposure_time']} seconds")
        
        # Extract WCS information
        print("\n2. Extracting WCS information...")
        wcs_data = []
        for fits_file in fits_files:
            wcs_info = self.extract_wcs_info(fits_file)
            wcs_data.append(wcs_info)
            if wcs_info.get('wcs_valid', False):
                print(f"  {wcs_info['filename']}: RA={wcs_info.get('crval1', 'N/A')}, Dec={wcs_info.get('crval2', 'N/A')}")
        
        # Find coordinate groups
        print(f"\n3. Finding coordinate groups (tolerance: {self.tolerance_arcmin} arcmin)...")
        coordinate_groups = self.find_coordinate_groups(wcs_data, exposure_data)
        coordinate_groups.sort(key=lambda x: x['count'], reverse=True)
        
        print(f"Found {len(coordinate_groups)} coordinate groups")
        for i, group in enumerate(coordinate_groups[:10]):  # Show top 10
            print(f"  Group {i+1}: {group['count']} files at RA={group['center_ra']:.3f}°, Dec={group['center_dec']:.3f}°")
        
        # Analyze exposure variations
        print("\n4. Analyzing exposure time variations within groups...")
        exposure_analysis = self.analyze_exposure_variations(coordinate_groups)
        
        # Create comprehensive output
        comprehensive_data = {
            'metadata': {
                'total_files': len(fits_files),
                'analysis_date': '2024-10-25',
                'description': 'Comprehensive FITS file analysis with coordinate grouping and exposure time variations',
                'coordinate_tolerance_arcmin': self.tolerance_arcmin,
                'input_directory': self.input_dir,
                'output_directory': self.output_dir
            },
            'summary_statistics': {
                'total_files': len(fits_files),
                'files_with_exposure_times': len([f for f in exposure_data if f['exposure_time'] is not None]),
                'files_with_valid_wcs': len([f for f in wcs_data if f.get('wcs_valid', False)]),
                'coordinate_groups': len(coordinate_groups),
                'groups_with_exposure_variations': len([g for g in exposure_analysis.values() if g['exposure_analysis']['valid_exposure_count'] > 1])
            },
            'coordinate_groups_with_exposure_analysis': list(exposure_analysis.values()),
            'individual_files': []
        }
        
        # Add individual files information
        exposure_map = {f['filename']: f['exposure_time'] for f in exposure_data}
        for wcs_file in wcs_data:
            filename = wcs_file['filename']
            
            file_info = {
                'filename': filename,
                'exposure_time_seconds': exposure_map.get(filename, None),
                'wcs_valid': wcs_file.get('wcs_valid', False),
                'coordinates': {
                    'ra_degrees': wcs_file.get('crval1'),
                    'dec_degrees': wcs_file.get('crval2')
                },
                'wcs_details': {
                    'pixel_scale_arcsec': wcs_file.get('pixel_scale_arcsec'),
                    'image_dimensions': {
                        'naxis1': wcs_file.get('naxis1'),
                        'naxis2': wcs_file.get('naxis2')
                    },
                    'coordinate_types': {
                        'ctype1': wcs_file.get('ctype1'),
                        'ctype2': wcs_file.get('ctype2')
                    }
                }
            }
            
            comprehensive_data['individual_files'].append(file_info)
        
        # Add exposure time statistics
        exposure_times = [f['exposure_time'] for f in exposure_data if f['exposure_time'] is not None]
        if exposure_times:
            comprehensive_data['exposure_statistics'] = {
                'min_exposure_seconds': min(exposure_times),
                'max_exposure_seconds': max(exposure_times),
                'mean_exposure_seconds': float(np.mean(exposure_times)),
                'median_exposure_seconds': float(np.median(exposure_times)),
                'total_exposure_seconds': sum(exposure_times)
            }
        
        # Save comprehensive analysis
        output_file = os.path.join(self.output_dir, 'comprehensive_fits_analysis.json')
        with open(output_file, 'w') as f:
            json.dump(comprehensive_data, f, indent=2)
        
        print(f"\n=== Analysis Complete ===")
        print(f"Total files: {comprehensive_data['summary_statistics']['total_files']}")
        print(f"Coordinate groups: {len(coordinate_groups)}")
        print(f"Groups with exposure variations: {comprehensive_data['summary_statistics']['groups_with_exposure_variations']}")
        
        if comprehensive_data.get('exposure_statistics'):
            stats = comprehensive_data['exposure_statistics']
            print(f"Exposure time range: {stats['min_exposure_seconds']:.1f} - {stats['max_exposure_seconds']:.1f} seconds")
            print(f"Total exposure time: {stats['total_exposure_seconds']:.1f} seconds ({stats['total_exposure_seconds']/3600:.1f} hours)")
        
        print(f"\nComprehensive analysis saved to: {output_file}")
        
        # Show exposure variation analysis for top groups
        print(f"\n=== Top Groups with Exposure Variations ===")
        groups_with_variations = [g for g in exposure_analysis.values() if g['exposure_analysis']['valid_exposure_count'] > 1]
        groups_with_variations.sort(key=lambda x: x['exposure_analysis']['exposure_variation_ratio'] or 0, reverse=True)
        
        for group in groups_with_variations[:5]:
            exp_analysis = group['exposure_analysis']
            print(f"\nGroup {group['group_id']} (RA={group['center_coordinates']['ra_degrees']:.3f}°, Dec={group['center_coordinates']['dec_degrees']:.3f}°):")
            print(f"  Files: {group['file_count']}, Valid exposures: {exp_analysis['valid_exposure_count']}")
            print(f"  Median exposure: {exp_analysis['median_exposure_seconds']:.1f}s")
            print(f"  Range: {exp_analysis['min_exposure_seconds']:.1f}s - {exp_analysis['max_exposure_seconds']:.1f}s")
            print(f"  Variation ratio: {exp_analysis['exposure_variation_ratio']:.2f}")
            print(f"  Low exposure files: {len(exp_analysis['low_exposure_files'])}")
            print(f"  High exposure files: {len(exp_analysis['high_exposure_files'])}")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive FITS Analysis')
    parser.add_argument('--input-dir', '-i', 
                       default='/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/raw/astronomy/hla_associations/direct_images',
                       help='Input directory containing FITS files')
    parser.add_argument('--output-dir', '-o', 
                       default='/home/jilab/Jae',
                       help='Output directory for analysis results')
    parser.add_argument('--tolerance', '-t', type=float, default=5.0,
                       help='Coordinate grouping tolerance in arcminutes (default: 5.0)')
    
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    analyzer = FITSAnalyzer(args.input_dir, args.output_dir, args.tolerance)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
