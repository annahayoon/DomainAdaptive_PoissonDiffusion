"""
Calibrated Sensor Parameters for Physics-Informed Posterior Sampling

⚠️ IMPORTANT: This module is OPTIONAL and primarily for documentation/reference.

Your preprocessed .pt files already contain measured calibration values (gain, read_noise)
from the actual dataset that may differ from these manufacturer specs due to:
- Different ISO settings during image capture
- Measurement variations in the preprocessing pipeline
- Per-image calibration vs. nominal specifications

ACTUAL DATASET VALUES (from DATASET_VERIFICATION.md):
- Photography: gain 1.8-5.0 (mean 4.13), read_noise 3.56-3.75 e⁻, s=79351
- Microscopy: gain 1.0, read_noise 1.5 e⁻, s=65534
- Astronomy: gain 1.0, read_noise 3.5 e⁻, s=121

USE CASES FOR THIS MODULE:
1. [Optional] Automatic sigma_max estimation via --use_sensor_calibration flag
2. [Documentation] Recording actual sensors used in published datasets
3. [Validation] Sanity-checking measured values against manufacturer specs
4. [Paper Methods] Citing sensor specifications in publications

FOR ACTUAL SAMPLING, YOU CAN DIRECTLY SPECIFY:
    python sample_noisy_pt_lle_guidance.py \
        --s 15871 \
        --sigma_r 5.0 \
        --kappa 0.5

The sampling will work without this file by using command-line parameters.

---

Key Parameters Computed by This Module:
- s: Scale factor (typically full-well capacity in electrons)
- sigma_r: Read noise standard deviation (in electrons or ADU)
- sigma_max: Initial noise level for posterior sampling (based on SNR)
- domain_min/max: Physical measurement range

Datasets and Sensors Documented:
- SID Dataset: Sony A7S II (IMX122) and Fuji X-T2 (X-Trans CMOS III)
- BioSR Dataset: Hamamatsu ORCA-Flash4.0 V3 (C13440-20CU)
- Hubble Legacy Fields: WFC3 and ACS instruments

References:
- Sony A7S II: https://cchen156.github.io/SID.html + empirical measurements
- Fuji X-T2: User measurements (manufacturer specs not publicly available)
- Hamamatsu ORCA-Flash4.0 V3: https://www.hamamatsu.com/us/en/product/cameras/cmos-cameras/C13440-20CU.html
- Hubble WFC3/ACS: NASA/STScI instrument handbooks
"""

from typing import Dict, Any
import numpy as np


class SensorCalibration:
    """Calibrated sensor parameters for different imaging domains."""
    
    # Sensor calibration database - Actual sensors used in datasets
    CALIBRATIONS = {
        "photography": {
            # SID Dataset: Sony A7S II (IMX122 sensor)
            "sony_a7s_ii": {
                "full_well_capacity": 157000,  # electrons (ISO 100)
                "read_noise": 4.2,  # electrons RMS (ISO 1600)
                "read_noise_iso2000": 1.4,  # electrons RMS (ISO 2000)
                "gain": 2.1,  # electrons/ADU (ISO 2000)
                "gain_iso100": 0.5,  # electrons/ADU (ISO 100)
                "domain_min": 0.0,
                "domain_max": 16383.0,  # 14-bit ADU range
                "bit_depth": 14,
                "quantum_efficiency": 0.50,  # Peak QE at 550nm
                "dark_current": 0.13,  # electrons/pixel/second at 20°C
                "unity_gain_iso": 4000,  # ISO where gain ≈ 1.0
                "sensor_model": "IMX122",
                "dataset": "SID",
            },
            # SID Dataset: Fuji X-T2 (X-Trans CMOS III sensor)
            # Note: Manufacturer does not publish detailed sensor specs (FWC, read noise, gain)
            # Values below are estimates based on typical APS-C sensors and user measurements
            # Reference: SID dataset also includes Fuji X-T2 images alongside Sony A7S II
            "fuji_xt2": {
                "full_well_capacity": 100000,  # electrons (ESTIMATED - typical APS-C)
                "read_noise": 3.5,  # electrons RMS (ESTIMATED - ISO 200)
                "read_noise_iso800": 2.0,  # electrons RMS (ESTIMATED - ISO 800)
                "gain": 1.5,  # electrons/ADU (ESTIMATED - ISO 200)
                "gain_iso800": 0.8,  # electrons/ADU (ESTIMATED - ISO 800)
                "domain_min": 0.0,
                "domain_max": 16383.0,  # 14-bit ADU range
                "bit_depth": 14,
                "pixel_size": 3.9,  # μm (verified)
                "quantum_efficiency": 0.55,  # Peak QE (ESTIMATED)
                "dark_current": 0.10,  # electrons/pixel/second (ESTIMATED)
                "unity_gain_iso": 800,  # ISO where gain ≈ 1.0 (ESTIMATED)
                "sensor_model": "X-Trans CMOS III",
                "dataset": "SID",
                "note": "Manufacturer specs not publicly available - values are estimates",
            },
            # Generic photography sensor (for compatibility)
            "generic": {
                "full_well_capacity": 100000,  # electrons (typical)
                "read_noise": 3.0,  # electrons RMS
                "gain": 1.0,  # electrons/ADU
                "domain_min": 0.0,
                "domain_max": 15871.0,
                "bit_depth": 14,
                "quantum_efficiency": 0.6,
                "dark_current": 0.1,
                "sensor_model": "Generic_Photography",
                "dataset": "Generic",
            },
        },
        
        "photography_lolv2": {
            # LOLv2 benchmark dataset - PNG 8-bit RGB images
            # Mixed camera types from various consumer cameras
            "lolv2_mixed": {
                "full_well_capacity": 50000,  # electrons (typical consumer camera)
                "read_noise": 2.0,  # electrons RMS (estimated for PNG 8-bit)
                "gain": 1.0,  # electrons/ADU (normalized for 8-bit)
                "domain_min": 0.0,
                "domain_max": 255.0,  # 8-bit PNG max
                "bit_depth": 8,
                "pixel_size": 5.0,  # μm (typical consumer camera)
                "quantum_efficiency": 0.7,  # typical consumer camera
                "dark_current": 0.05,  # electrons/pixel/second
                "sensor_model": "Mixed_Consumer_Cameras",
                "dataset": "LOLv2",
                "note": "PNG 8-bit RGB images from various consumer cameras",
            },
            # Generic fallback for LOLv2
            "generic": {
                "full_well_capacity": 50000,
                "read_noise": 2.0,
                "gain": 1.0,
                "domain_min": 0.0,
                "domain_max": 255.0,
                "bit_depth": 8,
                "quantum_efficiency": 0.7,
                "dark_current": 0.05,
                "sensor_model": "LOLv2_Generic",
                "dataset": "LOLv2",
            },
        },
        
        "microscopy": {
            # BioSR Dataset: Hamamatsu ORCA-Flash4.0 V3 (C13440-20CU)
            # Reference: https://www.sciencedirect.com/science/article/pii/S0092867418313084
            # Manufacturer datasheet: https://www.hamamatsu.com/us/en/product/cameras/cmos-cameras/C13440-20CU.html
            "hamamatsu_orca_flash4_v3": {
                "full_well_capacity": 30000,  # electrons (typical)
                "read_noise": 1.0,  # electrons RMS (standard scan mode)
                "gain": 0.46,  # electrons/ADU (manufacturer spec)
                "domain_min": 0.0,
                "domain_max": 65535.0,  # 16-bit
                "bit_depth": 16,
                "pixel_size": 6.5,  # μm
                "quantum_efficiency": 0.82,  # Peak QE at 550nm (manufacturer spec)
                "dark_current": 0.01,  # electrons/pixel/second at 0°C
                "sensor_type": "sCMOS",
                "sensor_model": "C13440-20CU",
                "resolution": "2048x2048",
                "dataset": "BioSR",
            }
        },
        
        "astronomy": {
            # Hubble Legacy Fields: WFC3 (Wide Field Camera 3)
            "hubble_wfc3": {
                "full_well_capacity": 80000,  # electrons
                "read_noise": 3.0,  # electrons RMS
                "gain": 2.5,  # electrons/ADU
                "domain_min": 0.0,  # Bias-subtracted
                "domain_max": 65535.0,  # 16-bit
                "bit_depth": 16,
                "quantum_efficiency": 0.45,  # Peak QE at 600nm
                "dark_current": 0.01,  # electrons/pixel/second (cooled to -82°C)
                "operating_temp": -82,  # Celsius
                "instrument": "WFC3",
                "dataset": "Hubble Legacy Fields",
            },
            # Hubble Legacy Fields: ACS (Advanced Camera for Surveys)
            "hubble_acs": {
                "full_well_capacity": 85000,  # electrons
                "read_noise": 4.0,  # electrons RMS
                "gain": 2.0,  # electrons/ADU
                "domain_min": 0.0,  # Bias-subtracted
                "domain_max": 65535.0,  # 16-bit
                "bit_depth": 16,
                "quantum_efficiency": 0.50,  # Peak QE at 650nm
                "dark_current": 0.02,  # electrons/pixel/second (cooled to -77°C)
                "operating_temp": -77,  # Celsius
                "instrument": "ACS",
                "dataset": "Hubble Legacy Fields",
            }
        }
    }
    
    @staticmethod
    def get_sensor_params(domain: str, sensor_name: str = "generic") -> Dict[str, Any]:
        """
        Get calibrated sensor parameters.
        
        Args:
            domain: Imaging domain (photography, microscopy, astronomy)
            sensor_name: Specific sensor model or "generic"
            
        Returns:
            Dictionary with calibrated parameters
        """
        if domain not in SensorCalibration.CALIBRATIONS:
            raise ValueError(f"Unknown domain: {domain}")
        
        domain_sensors = SensorCalibration.CALIBRATIONS[domain]
        
        if sensor_name not in domain_sensors:
            print(f"Warning: Sensor '{sensor_name}' not found, using 'generic' for {domain}")
            sensor_name = "generic"
        
        return domain_sensors[sensor_name].copy()
    
    @staticmethod
    def compute_sigma_max_from_snr(
        mean_signal: float,
        full_well: float,
        read_noise: float,
        domain_min: float,
        domain_max: float,
        conservative_factor: float = 1.0
    ) -> float:
        """
        Compute sigma_max based on signal-to-noise ratio.
        
        Theory:
        - For low-light images, the noise level in normalized space is:
          σ_normalized = √(SNR^-2) where SNR = signal / √(signal + read_noise^2)
        
        Args:
            mean_signal: Mean signal level in physical units (ADU or electrons)
            full_well: Full-well capacity (for normalization reference)
            read_noise: Read noise in same units as signal
            domain_min: Minimum physical value
            domain_max: Maximum physical value
            conservative_factor: Multiplier for safety (default: 1.0)
            
        Returns:
            sigma_max: Initial noise level for posterior sampling in [-1,1] space
        """
        # Normalize signal to [0, 1] space
        signal_normalized = (mean_signal - domain_min) / (domain_max - domain_min)
        signal_normalized = np.clip(signal_normalized, 1e-6, 1.0)
        
        # Compute expected photon count (assuming Poisson)
        # Scale by full_well for physical interpretation
        lambda_mean = signal_normalized * full_well
        
        # Poisson-Gaussian variance in physical space
        variance_physical = lambda_mean + read_noise**2
        
        # SNR in physical space
        snr_physical = lambda_mean / np.sqrt(variance_physical)
        
        # Noise standard deviation in normalized [0,1] space
        # This accounts for the signal-dependent noise
        sigma_normalized = 1.0 / snr_physical if snr_physical > 0 else 1.0
        
        # Apply conservative factor (e.g., 1.5 for safety)
        sigma_max = sigma_normalized * conservative_factor
        
        # Clip to reasonable range [0.001, 0.5]
        # (Too low: no effect, Too high: destroys signal)
        sigma_max = np.clip(sigma_max, 0.001, 0.5)
        
        return float(sigma_max)
    
    @staticmethod
    def compute_s_parameter(
        domain_min: float,
        domain_max: float,
        target_scale: float = 1000.0
    ) -> float:
        """
        Compute the scale parameter 's' for PG guidance.
        
        The 's' parameter should be chosen for numerical stability while
        maintaining physical meaning. A good heuristic:
        - s ~ 1000-10000 for numerical stability
        - s should capture the dynamic range
        
        Args:
            domain_min: Minimum physical value
            domain_max: Maximum physical value
            target_scale: Desired scale for numerical operations
            
        Returns:
            s: Scale parameter for PG guidance
        """
        # Option 1: Use target scale directly (simple, numerically stable)
        # This is what's currently recommended
        return target_scale
    
    @staticmethod
    def compute_sigma_r_normalized(
        read_noise_physical: float,
        domain_min: float,
        domain_max: float,
        s: float
    ) -> float:
        """
        Convert read noise from physical units to normalized scale used in PG guidance.
        
        Args:
            read_noise_physical: Read noise in physical units (electrons or ADU)
            domain_min: Minimum physical value
            domain_max: Maximum physical value  
            s: Scale parameter for PG guidance
            
        Returns:
            sigma_r: Read noise in normalized scale for PG guidance
        """
        # Normalize read noise to [0,1] space first
        domain_range = domain_max - domain_min
        sigma_r_norm_01 = read_noise_physical / domain_range
        
        # Scale to [0, s] range to match PG guidance
        sigma_r_scaled = sigma_r_norm_01 * s
        
        return float(sigma_r_scaled)
    
    @staticmethod
    def get_posterior_sampling_params(
        domain: str,
        sensor_name: str = "generic",
        mean_signal_physical: float = None,
        s: float = 1000.0,
        conservative_factor: float = 1.0
    ) -> Dict[str, float]:
        """
        Get all parameters needed for posterior sampling from calibration.
        
        This is the main function to use in your sampling script.
        
        Args:
            domain: Imaging domain
            sensor_name: Specific sensor or "generic"
            mean_signal_physical: Mean signal level of the noisy image (in physical units)
                                 If None, will use a conservative default
            s: Scale parameter for PG guidance (default: 1000.0)
            conservative_factor: Safety multiplier for sigma_max
            
        Returns:
            Dictionary with:
            - s: Scale parameter
            - sigma_r: Read noise (normalized for PG guidance)
            - sigma_max: Initial noise level for posterior sampling
            - domain_min, domain_max: Physical range
            - sensor_info: Full sensor specifications
        """
        # Get sensor calibration
        sensor_params = SensorCalibration.get_sensor_params(domain, sensor_name)
        
        # Extract key parameters
        domain_min = sensor_params["domain_min"]
        domain_max = sensor_params["domain_max"]
        read_noise_phys = sensor_params["read_noise"]
        full_well = sensor_params["full_well_capacity"]
        
        # Compute sigma_r for PG guidance (normalized to [0, s] scale)
        sigma_r_normalized = SensorCalibration.compute_sigma_r_normalized(
            read_noise_phys, domain_min, domain_max, s
        )
        
        # Compute sigma_max based on signal level
        if mean_signal_physical is None:
            # Conservative default: assume low signal (10% of range)
            mean_signal_physical = domain_min + 0.1 * (domain_max - domain_min)
        
        sigma_max = SensorCalibration.compute_sigma_max_from_snr(
            mean_signal_physical,
            full_well,
            read_noise_phys,
            domain_min,
            domain_max,
            conservative_factor
        )
        
        return {
            "s": s,
            "sigma_r": sigma_r_normalized,
            "sigma_max": sigma_max,
            "domain_min": domain_min,
            "domain_max": domain_max,
            "sensor_info": sensor_params,
        }


def main():
    """Example usage of sensor calibration."""
    
    print("=" * 80)
    print("SENSOR CALIBRATION EXAMPLES")
    print("=" * 80)
    
    # Example 1: Photography - SID Dataset
    print("\n1. Photography - SID Dataset (Sony A7S II, low-light scene)")
    mean_signal = 500.0  # ADU (low-light)
    params = SensorCalibration.get_posterior_sampling_params(
        domain="photography",
        sensor_name="sony_a7s_ii",
        mean_signal_physical=mean_signal,
        s=1000.0,
        conservative_factor=1.0
    )
    print(f"   Mean signal: {mean_signal:.1f} ADU")
    print(f"   s: {params['s']:.1f}")
    print(f"   σ_r: {params['sigma_r']:.3f} (normalized)")
    print(f"   σ_max: {params['sigma_max']:.6f} (for posterior sampling)")
    print(f"   Domain range: [{params['domain_min']}, {params['domain_max']}]")
    print(f"   Sensor: {params['sensor_info']['sensor_model']} ({params['sensor_info']['dataset']})")
    
    # Example 2: Microscopy - BioSR Dataset
    print("\n2. Microscopy - BioSR Dataset (Hamamatsu ORCA-Flash4.0 V3)")
    mean_signal = 5000.0  # ADU (typical fluorescence)
    params = SensorCalibration.get_posterior_sampling_params(
        domain="microscopy",
        sensor_name="hamamatsu_orca_flash4_v3",
        mean_signal_physical=mean_signal,
        s=1000.0,
        conservative_factor=1.0
    )
    print(f"   Mean signal: {mean_signal:.1f} ADU")
    print(f"   s: {params['s']:.1f}")
    print(f"   σ_r: {params['sigma_r']:.3f} (normalized)")
    print(f"   σ_max: {params['sigma_max']:.6f}")
    print(f"   Domain range: [{params['domain_min']}, {params['domain_max']}]")
    print(f"   Sensor: {params['sensor_info']['sensor_model']} ({params['sensor_info']['dataset']})")
    print(f"   QE: {params['sensor_info']['quantum_efficiency']*100:.0f}%, Gain: {params['sensor_info']['gain']:.2f} e-/ADU")
    
    # Example 3: Astronomy - Hubble Legacy Fields
    print("\n3. Astronomy - Hubble Legacy Fields (WFC3, deep sky)")
    mean_signal = 50.0  # ADU (very faint object)
    params = SensorCalibration.get_posterior_sampling_params(
        domain="astronomy",
        sensor_name="hubble_wfc3",
        mean_signal_physical=mean_signal,
        s=1000.0,
        conservative_factor=1.5  # More conservative for astronomy
    )
    print(f"   Mean signal: {mean_signal:.1f} ADU")
    print(f"   s: {params['s']:.1f}")
    print(f"   σ_r: {params['sigma_r']:.3f} (normalized)")
    print(f"   σ_max: {params['sigma_max']:.6f}")
    print(f"   Domain range: [{params['domain_min']}, {params['domain_max']}]")
    print(f"   Instrument: {params['sensor_info']['instrument']} ({params['sensor_info']['dataset']})")
    print(f"   Operating temp: {params['sensor_info']['operating_temp']}°C")
    
    # Example 4: Compare noise levels
    print("\n4. Comparison: Sigma_max vs. Signal Level (SID Dataset)")
    print("   Signal (ADU) | σ_max | SNR")
    print("   " + "-" * 40)
    for signal in [100, 500, 1000, 5000, 10000]:
        params = SensorCalibration.get_posterior_sampling_params(
            "photography", "sony_a7s_ii", signal, 1000.0
        )
        # Approximate SNR
        snr = signal / np.sqrt(signal + params['sensor_info']['read_noise']**2)
        print(f"   {signal:5d}        | {params['sigma_max']:.4f} | {snr:.1f}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

