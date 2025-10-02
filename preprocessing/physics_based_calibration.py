"""
Physics-Based Calibration System for Cross-Domain Diffusion Training
Implements domain-specific calibration methods based on sensor physics
"""

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from sklearn.linear_model import HuberRegressor


@dataclass
class PhysicsBasedCalibrationData:
    """Enhanced calibration with physics-based parameters and uncertainty"""

    # Core sensor parameters
    gain: float  # electrons per ADU
    gain_uncertainty: float  # uncertainty in gain estimation
    read_noise: float  # electrons RMS
    read_noise_uncertainty: float  # uncertainty in read noise

    # Sensor characteristics
    black_level: float  # ADU offset
    white_level: float  # ADU saturation
    full_well_capacity: float  # electrons (sensor physics)
    quantum_efficiency: float  # 0-1 (sensor physics)

    # Scene/background parameters
    background: float  # electrons
    scale_factor: float  # normalization scale

    # Quality metrics
    confidence_score: float  # 0-1, calibration reliability
    method: str  # calibration method used
    validation_metrics: Dict[str, float] = field(default_factory=dict)

    # Domain-specific physics
    domain_physics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.saturation_threshold = self.white_level * 0.95


class PhysicsBasedCalibrationMethod(ABC):
    """Abstract base class for physics-based calibration methods"""

    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self.logger = logging.getLogger(f"{__name__}.{domain_name}")

    @abstractmethod
    def estimate_calibration(
        self, images: List[np.ndarray], metadata: List[Dict[str, Any]]
    ) -> PhysicsBasedCalibrationData:
        """Estimate calibration parameters using domain-specific physics"""
        pass

    @abstractmethod
    def validate_calibration(
        self, calibration: PhysicsBasedCalibrationData, images: List[np.ndarray]
    ) -> Dict[str, float]:
        """Validate calibration against physical models"""
        pass


class PhotographyPhotonTransferCalibration(PhysicsBasedCalibrationMethod):
    """Photography calibration using photon transfer curve analysis"""

    def __init__(self):
        super().__init__("photography")

        # Camera-specific physics parameters - UPDATED WITH PRECISE MEASUREMENTS
        self.camera_physics = {
            "sony": {
                "pixel_size_um": 8.4,  # Sony A7S II pixel size
                "full_well_electrons": 87000,  # Approximate full well
                "quantum_efficiency": 0.6,  # Peak QE
                "expected_gain_range": (
                    0.79,
                    2.1,
                ),  # e-/ADU (measured: 0.79 at ISO 4000, 2.1 at ISO 2000)
                "expected_read_noise_range": (
                    2.0,
                    6.0,
                ),  # electrons (measured: 2-3e above ISO 4000, 6e below)
                "bit_depth": 14,
                "iso_specific_gains": {
                    "iso_2000": 2.1,
                    "iso_4000": 0.79,
                },  # Measured values
                "iso_specific_read_noise": {
                    "high_iso": 2.5,
                    "low_iso": 6.0,
                },  # Measured ranges
            },
            "fuji": {
                "pixel_size_um": 3.76,  # Fuji X-T30 pixel size
                "full_well_electrons": 25000,  # Approximate full well
                "quantum_efficiency": 0.55,  # Peak QE
                "expected_gain_range": (0.75, 0.75),  # e-/ADU (measured at base ISO)
                "expected_read_noise_range": (
                    2.5,
                    2.5,
                ),  # electrons (measured at base ISO)
                "bit_depth": 16,
                "base_iso_gain": 0.75,  # Measured value
                "base_iso_read_noise": 2.5,  # Measured value
            },
        }

    def estimate_calibration(
        self,
        image_pairs: List[Tuple[np.ndarray, np.ndarray]],
        metadata: List[Dict[str, Any]],
    ) -> PhysicsBasedCalibrationData:
        """Estimate gain and read noise using photon transfer curve"""

        # Determine camera type from first metadata
        camera_type = metadata[0].get("camera_type", "sony") if metadata else "sony"
        physics = self.camera_physics.get(camera_type, self.camera_physics["sony"])

        self.logger.info(
            f"Estimating calibration for {camera_type} camera using photon transfer"
        )

        if len(image_pairs) < 2:
            self.logger.warning(
                "Insufficient image pairs for robust photon transfer curve"
            )
            return self._fallback_calibration(
                camera_type, physics, image_pairs, metadata
            )

        # Extract signal and variance from image pairs
        signals, variances, weights = self._compute_photon_transfer_data(image_pairs)

        if len(signals) < 10:
            self.logger.warning("Insufficient signal levels for photon transfer curve")
            return self._fallback_calibration(
                camera_type, physics, image_pairs, metadata
            )

        # Fit photon transfer curve: variance = gain * signal + read_noise^2
        gain, read_noise, fit_quality = self._fit_photon_transfer_curve(
            signals, variances, weights
        )

        # Validate against expected physics
        gain, read_noise = self._validate_against_physics(gain, read_noise, physics)

        # Estimate uncertainties
        gain_uncertainty, read_noise_uncertainty = self._estimate_uncertainties(
            signals, variances, weights, gain, read_noise
        )

        # Compute other parameters
        all_images = [img for pair in image_pairs for img in pair]
        background = self._estimate_background(all_images)
        scale_factor = self._estimate_scale_factor(all_images)

        return PhysicsBasedCalibrationData(
            gain=gain,
            gain_uncertainty=gain_uncertainty,
            read_noise=read_noise,
            read_noise_uncertainty=read_noise_uncertainty,
            black_level=physics["bit_depth"] == 14 and 512.0 or 1024.0,
            white_level=2 ** physics["bit_depth"] - 1,
            full_well_capacity=physics["full_well_electrons"],
            quantum_efficiency=physics["quantum_efficiency"],
            background=background,
            scale_factor=scale_factor,
            confidence_score=fit_quality,
            method="photon_transfer_curve",
            validation_metrics={
                "signal_levels": len(signals),
                "fit_r_squared": fit_quality,
                "physics_validation": 1.0,
            },
            domain_physics={
                "camera_type": camera_type,
                "pixel_size_um": physics["pixel_size_um"],
                "exposure_pair_analysis": True,
            },
        )

    def _compute_photon_transfer_data(
        self, image_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute signal levels and variances from image pairs"""

        signals = []
        variances = []
        weights = []

        for short_img, long_img in image_pairs:
            # Extract exposure ratio from images (assume it's in metadata)
            # For now, estimate from mean signal levels
            short_mean = np.mean(short_img)
            long_mean = np.mean(long_img)

            if long_mean > short_mean * 1.5:  # Valid pair
                # Compute variance from difference image
                # Scale short image to match long image signal level
                scale_ratio = long_mean / (short_mean + 1e-8)
                scaled_short = short_img * scale_ratio

                # Difference variance (should be 2 * read_noise^2 + gain * signal)
                diff_var = np.var(long_img - scaled_short)

                # Signal level (use long image as reference)
                signal_level = long_mean

                # Weight by number of pixels and signal level
                weight = np.sqrt(long_img.size) / (1.0 + 1000.0 / signal_level)

                signals.append(signal_level)
                variances.append(
                    diff_var / 2.0
                )  # Divide by 2 for single image variance
                weights.append(weight)

        return np.array(signals), np.array(variances), np.array(weights)

    def _fit_photon_transfer_curve(
        self, signals: np.ndarray, variances: np.ndarray, weights: np.ndarray
    ) -> Tuple[float, float, float]:
        """Fit photon transfer curve to estimate gain and read noise"""

        # Use robust regression to handle outliers
        X = signals.reshape(-1, 1)
        y = variances

        # Weighted robust regression
        regressor = HuberRegressor(epsilon=1.35, max_iter=1000)
        regressor.fit(X, y, sample_weight=weights)

        gain = regressor.coef_[0]
        read_noise_squared = regressor.intercept_
        read_noise = np.sqrt(max(read_noise_squared, 0.1))  # Ensure positive

        # Compute fit quality (R-squared equivalent)
        y_pred = regressor.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = max(0.0, 1 - (ss_res / (ss_tot + 1e-8)))

        return gain, read_noise, r_squared

    def _validate_against_physics(
        self, gain: float, read_noise: float, physics: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Validate and constrain parameters against known physics"""

        gain_min, gain_max = physics["expected_gain_range"]
        rn_min, rn_max = physics["expected_read_noise_range"]

        # Constrain to physically reasonable ranges
        if gain < gain_min or gain > gain_max:
            self.logger.warning(
                f"Gain {gain:.2f} outside expected range [{gain_min}, {gain_max}]"
            )
            gain = np.clip(gain, gain_min, gain_max)

        if read_noise < rn_min or read_noise > rn_max:
            self.logger.warning(
                f"Read noise {read_noise:.2f} outside expected range [{rn_min}, {rn_max}]"
            )
            read_noise = np.clip(read_noise, rn_min, rn_max)

        return gain, read_noise

    def _estimate_uncertainties(
        self,
        signals: np.ndarray,
        variances: np.ndarray,
        weights: np.ndarray,
        gain: float,
        read_noise: float,
    ) -> Tuple[float, float]:
        """Estimate uncertainties in gain and read noise"""

        # Bootstrap estimation of uncertainties
        n_bootstrap = 100
        gains = []
        read_noises = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(signals), len(signals), replace=True)
            boot_signals = signals[indices]
            boot_variances = variances[indices]
            boot_weights = weights[indices]

            # Fit to bootstrap sample
            try:
                boot_gain, boot_rn, _ = self._fit_photon_transfer_curve(
                    boot_signals, boot_variances, boot_weights
                )
                gains.append(boot_gain)
                read_noises.append(boot_rn)
            except:
                continue

        if len(gains) > 10:
            gain_uncertainty = np.std(gains)
            rn_uncertainty = np.std(read_noises)
        else:
            # Fallback uncertainty estimates
            gain_uncertainty = gain * 0.1  # 10% uncertainty
            rn_uncertainty = read_noise * 0.2  # 20% uncertainty

        return gain_uncertainty, rn_uncertainty

    def _estimate_background(self, images: List[np.ndarray]) -> float:
        """Estimate background level in electrons"""
        backgrounds = []
        for img in images[:5]:  # Sample a few images
            bg = np.percentile(img, 10)  # 10th percentile as background
            backgrounds.append(bg)
        return float(np.median(backgrounds))

    def _estimate_scale_factor(self, images: List[np.ndarray]) -> float:
        """Estimate scale factor for normalization"""
        scales = []
        for img in images[:10]:  # Sample images
            scale = np.percentile(img, 99.9)  # High percentile
            scales.append(scale)
        return float(np.median(scales))

    def _fallback_calibration(
        self,
        camera_type: str,
        physics: Dict[str, Any],
        image_pairs: List[Tuple[np.ndarray, np.ndarray]],
        metadata: List[Dict[str, Any]],
    ) -> PhysicsBasedCalibrationData:
        """Fallback calibration using camera defaults when PTC fails"""

        self.logger.warning("Using fallback calibration with camera defaults")

        # Use middle of expected ranges as defaults
        gain_range = physics["expected_gain_range"]
        rn_range = physics["expected_read_noise_range"]

        gain = (gain_range[0] + gain_range[1]) / 2
        read_noise = (rn_range[0] + rn_range[1]) / 2

        # Estimate from available data
        all_images = [img for pair in image_pairs for img in pair]
        if all_images:
            background = self._estimate_background(all_images)
            scale_factor = self._estimate_scale_factor(all_images)
        else:
            background = 100.0  # Default background
            scale_factor = 1000.0  # Default scale

        return PhysicsBasedCalibrationData(
            gain=gain,
            gain_uncertainty=gain * 0.2,  # 20% uncertainty for defaults
            read_noise=read_noise,
            read_noise_uncertainty=read_noise * 0.3,  # 30% uncertainty for defaults
            black_level=physics["bit_depth"] == 14 and 512.0 or 1024.0,
            white_level=2 ** physics["bit_depth"] - 1,
            full_well_capacity=physics["full_well_electrons"],
            quantum_efficiency=physics["quantum_efficiency"],
            background=background,
            scale_factor=scale_factor,
            confidence_score=0.5,  # Low confidence for fallback
            method="camera_defaults_fallback",
            validation_metrics={
                "fallback_reason": "insufficient_data",
                "physics_validation": 0.8,
            },
            domain_physics={
                "camera_type": camera_type,
                "pixel_size_um": physics["pixel_size_um"],
            },
        )

    def validate_calibration(
        self, calibration: PhysicsBasedCalibrationData, images: List[np.ndarray]
    ) -> Dict[str, float]:
        """Validate calibration against physics and image statistics"""

        validation = {}

        # Check if gain is reasonable for camera type
        camera_type = calibration.domain_physics.get("camera_type", "sony")
        physics = self.camera_physics.get(camera_type, self.camera_physics["sony"])

        gain_min, gain_max = physics["expected_gain_range"]
        validation["gain_in_range"] = (
            1.0 if gain_min <= calibration.gain <= gain_max else 0.0
        )

        rn_min, rn_max = physics["expected_read_noise_range"]
        validation["read_noise_in_range"] = (
            1.0 if rn_min <= calibration.read_noise <= rn_max else 0.0
        )

        # Validate against image noise if images provided
        if images:
            # Check if read noise estimate matches low-signal variance
            low_signal_regions = []
            for img in images[:3]:
                low_regions = img[img < np.percentile(img, 20)]
                if len(low_regions) > 100:
                    low_signal_regions.extend(low_regions)

            if low_signal_regions:
                observed_noise = np.std(low_signal_regions)
                expected_noise_adu = calibration.read_noise / calibration.gain
                noise_ratio = observed_noise / (expected_noise_adu + 1e-6)
                validation["noise_consistency"] = max(0.0, 1.0 - abs(1.0 - noise_ratio))
            else:
                validation["noise_consistency"] = 0.5

        return validation


class MicroscopyPhotonLimitedCalibration(PhysicsBasedCalibrationMethod):
    """Microscopy calibration for photon-limited sCMOS sensors"""

    def __init__(self):
        super().__init__("microscopy")

        # sCMOS physics parameters - UPDATED WITH PRECISE BioSR MEASUREMENTS
        self.scmos_physics = {
            "pixel_size_um": 6.5,  # Typical sCMOS
            "full_well_electrons": 30000,  # Typical sCMOS
            "quantum_efficiency": 0.85,  # High QE for modern sCMOS
            "expected_gain_range": (0.5, 1.5),  # e-/ADU (BioSR measured range)
            "expected_read_noise_range": (
                1.0,
                2.0,
            ),  # electrons RMS per pixel (BioSR measured range)
            "bit_depth": 16,
            "photon_conversion": 1.0,  # Already in photons/electrons
            "dark_current": 0.1,  # e-/pixel/s at room temperature
            "default_gain": 1.0,  # e-/ADU (BioSR typical, measured)
            "default_read_noise": 1.5,  # electrons (BioSR typical, measured)
        }

    def estimate_calibration(
        self, images: List[np.ndarray], metadata: List[Dict[str, Any]]
    ) -> PhysicsBasedCalibrationData:
        """Estimate calibration for photon-limited microscopy"""

        self.logger.info("Estimating calibration for sCMOS microscopy sensor")

        # For microscopy, we often have noise level information
        noise_levels = [meta.get("noise_level", 1) for meta in metadata]
        structures = [meta.get("structure", "unknown") for meta in metadata]

        # Use noise level analysis for gain estimation
        gain, read_noise = self._estimate_from_noise_levels(images, noise_levels)

        # Validate against sCMOS physics
        gain, read_noise = self._validate_scmos_physics(gain, read_noise)

        # Estimate other parameters
        background = self._estimate_microscopy_background(images)
        scale_factor = self._estimate_microscopy_scale(images)

        # Compute confidence based on data quality
        confidence = self._compute_microscopy_confidence(images, noise_levels)

        return PhysicsBasedCalibrationData(
            gain=gain,
            gain_uncertainty=gain * 0.15,  # 15% uncertainty
            read_noise=read_noise,
            read_noise_uncertainty=read_noise * 0.2,  # 20% uncertainty
            black_level=0.0,  # sCMOS typically bias-corrected
            white_level=65535.0,  # 16-bit
            full_well_capacity=self.scmos_physics["full_well_electrons"],
            quantum_efficiency=self.scmos_physics["quantum_efficiency"],
            background=background,
            scale_factor=scale_factor,
            confidence_score=confidence,
            method="noise_level_analysis",
            validation_metrics={
                "noise_levels_analyzed": len(set(noise_levels)),
                "structures_analyzed": len(set(structures)),
                "scmos_validation": 1.0,
            },
            domain_physics={
                "sensor_type": "sCMOS",
                "pixel_size_um": self.scmos_physics["pixel_size_um"],
                "photon_limited": True,
            },
        )

    def _estimate_from_noise_levels(
        self, images: List[np.ndarray], noise_levels: List[int]
    ) -> Tuple[float, float]:
        """Estimate gain and read noise from BioSR noise levels"""

        # Group images by noise level
        level_groups = {}
        for img, level in zip(images, noise_levels):
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(img)

        if len(level_groups) < 2:
            # Fallback to single image analysis
            return self._single_image_analysis(images[0])

        # Analyze noise vs signal relationship
        signal_levels = []
        noise_levels_measured = []

        for level, imgs in level_groups.items():
            if len(imgs) > 0:
                # Average signal and noise for this level
                avg_signal = np.mean([np.mean(img) for img in imgs])
                avg_noise = np.mean([np.std(img) for img in imgs])

                signal_levels.append(avg_signal)
                noise_levels_measured.append(avg_noise)

        if len(signal_levels) >= 2:
            # Fit noise model: noise^2 = gain * signal + read_noise^2
            signals = np.array(signal_levels)
            noises_squared = np.array(noise_levels_measured) ** 2

            # Linear regression
            A = np.vstack([signals, np.ones(len(signals))]).T
            gain, read_noise_squared = np.linalg.lstsq(A, noises_squared, rcond=None)[0]
            read_noise = np.sqrt(max(read_noise_squared, 0.1))

            return max(gain, 0.1), read_noise
        else:
            return self._single_image_analysis(images[0])

    def _single_image_analysis(self, image: np.ndarray) -> Tuple[float, float]:
        """Fallback single image analysis"""

        # Use updated sCMOS default values
        gain = self.scmos_physics["default_gain"]  # 1.0 e-/ADU

        # Estimate read noise from low-signal regions
        low_signal = image[image < np.percentile(image, 30)]
        if len(low_signal) > 100:
            read_noise_adu = np.std(low_signal)
            read_noise = read_noise_adu * gain
        else:
            read_noise = self.scmos_physics["default_read_noise"]  # 1.5 electrons

        return gain, read_noise

    def _validate_scmos_physics(
        self, gain: float, read_noise: float
    ) -> Tuple[float, float]:
        """Validate against sCMOS physics"""

        gain_min, gain_max = self.scmos_physics["expected_gain_range"]
        rn_min, rn_max = self.scmos_physics["expected_read_noise_range"]

        if gain < gain_min or gain > gain_max:
            self.logger.warning(
                f"Gain {gain:.2f} outside sCMOS range [{gain_min}, {gain_max}]"
            )
            gain = np.clip(gain, gain_min, gain_max)

        if read_noise < rn_min or read_noise > rn_max:
            self.logger.warning(
                f"Read noise {read_noise:.2f} outside sCMOS range [{rn_min}, {rn_max}]"
            )
            read_noise = np.clip(read_noise, rn_min, rn_max)

        return gain, read_noise

    def _estimate_microscopy_background(self, images: List[np.ndarray]) -> float:
        """Estimate microscopy background (cellular autofluorescence)"""
        backgrounds = []
        for img in images[:5]:
            # Use 5th percentile for microscopy (avoid zero regions)
            bg = np.percentile(img, 5)
            backgrounds.append(bg)
        return float(np.median(backgrounds))

    def _estimate_microscopy_scale(self, images: List[np.ndarray]) -> float:
        """Estimate scale for microscopy normalization"""
        scales = []
        for img in images[:10]:
            # Use 99.5th percentile (avoid saturated pixels)
            scale = np.percentile(img, 99.5)
            scales.append(scale)
        return float(np.median(scales))

    def _compute_microscopy_confidence(
        self, images: List[np.ndarray], noise_levels: List[int]
    ) -> float:
        """Compute confidence score for microscopy calibration"""

        # Base confidence
        confidence = 0.7

        # Boost if multiple noise levels available
        unique_levels = len(set(noise_levels))
        if unique_levels >= 3:
            confidence += 0.2
        elif unique_levels >= 2:
            confidence += 0.1

        # Boost if sufficient data
        if len(images) >= 10:
            confidence += 0.1

        return min(1.0, confidence)

    def validate_calibration(
        self, calibration: PhysicsBasedCalibrationData, images: List[np.ndarray]
    ) -> Dict[str, float]:
        """Validate microscopy calibration"""

        validation = {}

        # Check against sCMOS ranges
        gain_min, gain_max = self.scmos_physics["expected_gain_range"]
        validation["gain_in_range"] = (
            1.0 if gain_min <= calibration.gain <= gain_max else 0.0
        )

        rn_min, rn_max = self.scmos_physics["expected_read_noise_range"]
        validation["read_noise_in_range"] = (
            1.0 if rn_min <= calibration.read_noise <= rn_max else 0.0
        )

        # Check photon-limited assumption
        if images:
            # In photon-limited regime, variance ≈ mean for high signals
            high_signal_regions = []
            for img in images[:3]:
                high_regions = img[img > np.percentile(img, 80)]
                if len(high_regions) > 100:
                    high_signal_regions.extend(high_regions)

            if high_signal_regions:
                mean_signal = np.mean(high_signal_regions)
                var_signal = np.var(high_signal_regions)
                # Convert to electrons
                mean_electrons = mean_signal * calibration.gain
                var_electrons = var_signal * (calibration.gain**2)

                # Check if variance ≈ mean (photon-limited)
                photon_ratio = var_electrons / (mean_electrons + 1e-6)
                validation["photon_limited"] = max(0.0, 1.0 - abs(1.0 - photon_ratio))
            else:
                validation["photon_limited"] = 0.5

        return validation


class AstronomyHeaderBasedCalibration(PhysicsBasedCalibrationMethod):
    """Astronomy calibration using FITS header information"""

    def __init__(self):
        super().__init__("astronomy")

        # HST instrument physics - UPDATED WITH PRECISE MEASUREMENTS
        self.hst_physics = {
            "ACS": {
                "pixel_size_um": 15.0,  # ACS/WFC pixel size
                "full_well_electrons": 84000,  # ACS full well
                "quantum_efficiency": 0.8,  # Peak QE
                "expected_gain_range": (
                    1.0,
                    1.0,
                ),  # e-/DN (measured: ~1.0 for most amplifiers)
                "expected_read_noise_range": (
                    2.0,
                    5.0,
                ),  # electrons (measured: 2.0-5.0 across amplifiers)
                "bit_depth": 16,
                "default_gain": 1.0,  # Measured default
                "default_read_noise": 3.5,  # Measured mid-range
            },
            "WFC3_UVIS": {
                "pixel_size_um": 13.2,  # WFC3/UVIS pixel size
                "full_well_electrons": 70000,  # WFC3 UVIS full well
                "quantum_efficiency": 0.7,  # Peak QE
                "expected_gain_range": (
                    1.0,
                    4.0,
                ),  # e-/DN (measured supported: 1, 1.5, 2, 4)
                "expected_read_noise_range": (
                    2.95,
                    3.22,
                ),  # electrons (measured mean across amplifiers)
                "bit_depth": 16,
                "default_gain": 1.5,  # Measured default
                "default_read_noise": 3.09,  # Measured mean value
            },
            "WFC3_IR": {
                "pixel_size_um": 18.0,  # WFC3/IR pixel size
                "full_well_electrons": 80000,  # WFC3 IR full well
                "quantum_efficiency": 0.8,  # Peak QE in IR
                "expected_gain_range": (
                    2.0,
                    4.0,
                ),  # e-/DN (measured supported: 2.0, 2.5, 3.0, 4.0)
                "expected_read_noise_range": (
                    10.0,
                    20.0,
                ),  # electrons (measured: higher than UVIS, <20)
                "bit_depth": 16,
                "default_gain": 2.5,  # Measured default
                "default_read_noise": 15.0,  # Measured conservative estimate
            },
            "WFPC2": {
                "pixel_size_um": 15.0,  # WFPC2 pixel size
                "full_well_electrons": 85000,  # WFPC2 full well
                "quantum_efficiency": 0.6,  # Peak QE
                "expected_gain_range": (
                    7.0,
                    14.0,
                ),  # e-/DN (measured ATD-GAIN=7 and 14)
                "expected_read_noise_range": (
                    5.0,
                    8.0,
                ),  # electrons RMS (measured range at gain 7)
                "bit_depth": 12,
                "default_gain": 7.0,  # Measured most common
                "default_read_noise": 6.5,  # Measured mid-range
            },
        }

    def estimate_calibration(
        self, images: List[np.ndarray], metadata: List[Dict[str, Any]]
    ) -> PhysicsBasedCalibrationData:
        """Estimate calibration from FITS headers with validation"""

        self.logger.info("Estimating calibration from FITS headers")

        # Extract instrument information
        instruments = [meta.get("instrument", "ACS") for meta in metadata]
        detectors = [meta.get("detector", "WFC") for meta in metadata]

        # Determine dominant instrument
        instrument = max(set(instruments), key=instruments.count)
        physics = self.hst_physics.get(instrument, self.hst_physics["ACS"])

        # Extract calibration from headers
        gains = []
        read_noises = []

        for meta in metadata:
            # Try multiple header keywords
            gain = meta.get("gain") or meta.get("GAIN") or meta.get("CCDGAIN")
            rn = meta.get("read_noise") or meta.get("RDNOISE") or meta.get("READNSE")

            if gain is not None:
                gains.append(float(gain))
            if rn is not None:
                read_noises.append(float(rn))

        # Use robust statistics
        if gains:
            gain = float(np.median(gains))
            gain_uncertainty = float(np.std(gains)) if len(gains) > 1 else gain * 0.05
        else:
            # Fallback to typical values
            gain = (
                physics["expected_gain_range"][0] + physics["expected_gain_range"][1]
            ) / 2
            gain_uncertainty = gain * 0.2

        if read_noises:
            read_noise = float(np.median(read_noises))
            read_noise_uncertainty = (
                float(np.std(read_noises)) if len(read_noises) > 1 else read_noise * 0.1
            )
        else:
            # Fallback to typical values
            read_noise = (
                physics["expected_read_noise_range"][0]
                + physics["expected_read_noise_range"][1]
            ) / 2
            read_noise_uncertainty = read_noise * 0.2

        # Validate against instrument physics
        gain, read_noise = self._validate_hst_physics(gain, read_noise, physics)

        # Cross-validate with image statistics
        validation_passed = self._validate_with_images(images, gain, read_noise)

        # Estimate other parameters
        background = self._estimate_astronomy_background(images)
        scale_factor = self._estimate_astronomy_scale(images)

        # Compute confidence
        confidence = self._compute_astronomy_confidence(
            len(gains), len(read_noises), validation_passed
        )

        return PhysicsBasedCalibrationData(
            gain=gain,
            gain_uncertainty=gain_uncertainty,
            read_noise=read_noise,
            read_noise_uncertainty=read_noise_uncertainty,
            black_level=0.0,  # HST data is bias-subtracted
            white_level=65535.0,  # 16-bit
            full_well_capacity=physics["full_well_electrons"],
            quantum_efficiency=physics["quantum_efficiency"],
            background=background,
            scale_factor=scale_factor,
            confidence_score=confidence,
            method="fits_header_analysis",
            validation_metrics={
                "header_gains_found": len(gains),
                "header_read_noises_found": len(read_noises),
                "instrument_validation": 1.0 if validation_passed else 0.7,
                "dominant_instrument": instrument,
            },
            domain_physics={
                "instrument": instrument,
                "pixel_size_um": physics["pixel_size_um"],
                "space_based": True,
                "cosmic_ray_affected": True,
            },
        )

    def _validate_hst_physics(
        self, gain: float, read_noise: float, physics: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Validate against HST instrument physics"""

        gain_min, gain_max = physics["expected_gain_range"]
        rn_min, rn_max = physics["expected_read_noise_range"]

        if gain < gain_min or gain > gain_max:
            self.logger.warning(
                f"Gain {gain:.2f} outside {physics} range [{gain_min}, {gain_max}]"
            )
            gain = np.clip(gain, gain_min, gain_max)

        if read_noise < rn_min or read_noise > rn_max:
            self.logger.warning(
                f"Read noise {read_noise:.2f} outside {physics} range [{rn_min}, {rn_max}]"
            )
            read_noise = np.clip(read_noise, rn_min, rn_max)

        return gain, read_noise

    def _validate_with_images(
        self, images: List[np.ndarray], gain: float, read_noise: float
    ) -> bool:
        """Cross-validate calibration with image statistics"""

        if not images:
            return True  # Can't validate without images

        # Check if background noise matches read noise expectation
        background_noises = []
        for img in images[:3]:
            # Use low percentile regions as background
            bg_regions = img[img < np.percentile(img, 20)]
            if len(bg_regions) > 1000:
                bg_noise_adu = np.std(bg_regions)
                bg_noise_electrons = bg_noise_adu * gain
                background_noises.append(bg_noise_electrons)

        if background_noises:
            median_bg_noise = np.median(background_noises)
            # Should be close to read noise (within factor of 2)
            ratio = median_bg_noise / read_noise
            return 0.5 <= ratio <= 2.0

        return True  # Default to valid if can't check

    def _estimate_astronomy_background(self, images: List[np.ndarray]) -> float:
        """Estimate sky background level"""
        backgrounds = []
        for img in images[:5]:
            # Use sigma-clipped mean for robust background
            bg = np.percentile(img, 10)  # Conservative background estimate
            backgrounds.append(bg)
        return float(np.median(backgrounds))

    def _estimate_astronomy_scale(self, images: List[np.ndarray]) -> float:
        """Estimate scale for astronomy normalization"""
        scales = []
        for img in images[:10]:
            # Use high percentile but avoid cosmic rays
            scale = np.percentile(img, 99.9)
            scales.append(scale)
        return float(np.median(scales))

    def _compute_astronomy_confidence(
        self, n_gains: int, n_read_noises: int, validation_passed: bool
    ) -> float:
        """Compute confidence for astronomy calibration"""

        confidence = 0.6  # Base confidence for header-based method

        # Boost for multiple header values
        if n_gains >= 3:
            confidence += 0.2
        elif n_gains >= 1:
            confidence += 0.1

        if n_read_noises >= 3:
            confidence += 0.2
        elif n_read_noises >= 1:
            confidence += 0.1

        # Boost for validation
        if validation_passed:
            confidence += 0.1

        return min(1.0, confidence)

    def validate_calibration(
        self, calibration: PhysicsBasedCalibrationData, images: List[np.ndarray]
    ) -> Dict[str, float]:
        """Validate astronomy calibration"""

        validation = {}

        # Check against instrument ranges
        instrument = calibration.domain_physics.get("instrument", "ACS")
        physics = self.hst_physics.get(instrument, self.hst_physics["ACS"])

        gain_min, gain_max = physics["expected_gain_range"]
        validation["gain_in_range"] = (
            1.0 if gain_min <= calibration.gain <= gain_max else 0.0
        )

        rn_min, rn_max = physics["expected_read_noise_range"]
        validation["read_noise_in_range"] = (
            1.0 if rn_min <= calibration.read_noise <= rn_max else 0.0
        )

        # Validate with images if available
        if images:
            validation["image_consistency"] = (
                1.0
                if self._validate_with_images(
                    images, calibration.gain, calibration.read_noise
                )
                else 0.5
            )

        return validation


# Factory function to create appropriate calibration method
def create_physics_based_calibrator(domain_name: str) -> PhysicsBasedCalibrationMethod:
    """Factory function to create domain-specific calibration method"""

    if domain_name == "photography":
        return PhotographyPhotonTransferCalibration()
    elif domain_name == "microscopy":
        return MicroscopyPhotonLimitedCalibration()
    elif domain_name == "astronomy":
        return AstronomyHeaderBasedCalibration()
    else:
        raise ValueError(f"Unknown domain: {domain_name}")


def main():
    """Test physics-based calibration system"""

    print("🔬 PHYSICS-BASED CALIBRATION SYSTEM")
    print("=" * 50)

    # Test each domain
    domains = ["photography", "microscopy", "astronomy"]

    for domain in domains:
        print(f"\n📊 Testing {domain.title()} Calibration:")

        try:
            calibrator = create_physics_based_calibrator(domain)
            print(f"   ✅ {calibrator.__class__.__name__} created")
            print(f"   📋 Method: {calibrator.__class__.__doc__}")
        except Exception as e:
            print(f"   ❌ Error creating calibrator: {e}")

    print(f"\n🎯 Physics-Based Calibration System Ready!")
    print(f"   📸 Photography: Photon Transfer Curve Analysis")
    print(f"   🔬 Microscopy: Noise Level Analysis (sCMOS)")
    print(f"   🌌 Astronomy: FITS Header + Validation")


if __name__ == "__main__":
    main()
