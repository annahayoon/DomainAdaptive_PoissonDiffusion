#!/usr/bin/env python3
"""
Astronomy Data Preprocessor for Hubble Legacy Field
Handles negative values and channel consistency for Poisson-Gaussian denoising
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class AstronomyDataPreprocessor:
    """
    Preprocessor for astronomy data with proper handling of:
    1. Negative values from background subtraction
    2. Channel consistency (multi-band to single channel)
    3. Extreme dynamic range of astronomical data
    """

    def __init__(
        self,
        offset_method: str = "adaptive",  # "fixed", "percentile", or "adaptive"
        target_channels: int = 1,  # Astronomy typically uses single channel
        preserve_noise_statistics: bool = True,
        min_offset: float = 100.0,  # Minimum offset to ensure positivity
    ):
        self.offset_method = offset_method
        self.target_channels = target_channels
        self.preserve_noise_statistics = preserve_noise_statistics
        self.min_offset = min_offset

    def handle_negative_values(
        self, data: torch.Tensor, metadata: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Handle negative values in astronomy data while preserving noise characteristics.

        Scientific Rationale:
        - Negative values are artifacts of calibration, not physical
        - We need to shift data to positive domain for Poisson modeling
        - But we must preserve the relative noise structure

        Args:
            data: Input tensor with potential negative values
            metadata: Optional metadata dictionary

        Returns:
            Tuple of (processed_data, processing_info)
        """
        if metadata is None:
            metadata = {}

        # Compute statistics before processing
        data_min = data.min().item()
        data_max = data.max().item()
        data_mean = data.mean().item()
        data_std = data.std().item()

        processing_info = {
            "original_min": data_min,
            "original_max": data_max,
            "original_mean": data_mean,
            "original_std": data_std,
        }

        if data_min < 0:
            logger.info(f"Handling negative values: min={data_min:.2f}")

            if self.offset_method == "fixed":
                # Simple fixed offset - preserves relative differences
                offset = max(abs(data_min) + self.min_offset, self.min_offset)

            elif self.offset_method == "percentile":
                # Use percentile-based offset for robustness to outliers
                p5 = torch.quantile(data, 0.05).item()
                offset = max(abs(p5) + self.min_offset, self.min_offset)

            elif self.offset_method == "adaptive":
                # Adaptive offset based on noise statistics
                # This is most physically motivated for astronomy

                # Estimate background noise (MAD estimator)
                median = torch.median(data).item()
                mad = torch.median(torch.abs(data - median)).item()
                noise_std = 1.4826 * mad  # MAD to std conversion

                # Offset should be at least 3-5 sigma above the negative tail
                # This ensures >99.7% of noise fluctuations are positive
                offset = max(
                    abs(data_min) + 3 * noise_std, 5 * noise_std, self.min_offset
                )

                processing_info["estimated_noise_std"] = noise_std

            else:
                raise ValueError(f"Unknown offset method: {self.offset_method}")

            # Apply offset
            data = data + offset
            processing_info["applied_offset"] = offset

            # Verify positivity
            assert (
                data.min() > 0
            ), f"Data still has negative values after offset: {data.min()}"

            # Store offset in metadata for inverse transform
            metadata["astronomy_offset"] = offset

        else:
            processing_info["applied_offset"] = 0.0

        # Update statistics after processing
        processing_info["processed_min"] = data.min().item()
        processing_info["processed_max"] = data.max().item()
        processing_info["processed_mean"] = data.mean().item()
        processing_info["processed_std"] = data.std().item()

        return data, processing_info

    def handle_channel_consistency(
        self, data: torch.Tensor, metadata: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Ensure consistent channel dimensions for astronomy data.

        Scientific Approach:
        - Multi-band astronomy data (e.g., griz filters) -> weighted combination
        - RGB/RGBA data -> luminance extraction
        - Already single channel -> pass through

        Args:
            data: Input tensor with shape [C, H, W] or [H, W]
            metadata: Optional metadata

        Returns:
            Tuple of (processed_data, channel_info)
        """
        channel_info = {}

        # Handle different input shapes
        if data.ndim == 2:
            # Already 2D, add channel dimension
            data = data.unsqueeze(0)
            channel_info["original_channels"] = 1

        elif data.ndim == 3:
            num_channels = data.shape[0]
            channel_info["original_channels"] = num_channels

            if num_channels == self.target_channels:
                # Already correct number of channels
                pass

            elif num_channels == 3 or num_channels == 4:
                # RGB or RGBA - convert to luminance
                # Use astronomy-appropriate weights (not standard RGB)
                if num_channels == 4:
                    # Remove alpha channel if present
                    data = data[:3]

                # Weighted combination optimized for astronomical sources
                # These weights emphasize red/NIR where many astronomical sources are bright
                weights = torch.tensor([0.2, 0.3, 0.5]).reshape(3, 1, 1).to(data.device)
                data = (data[:3] * weights).sum(dim=0, keepdim=True)

                channel_info["conversion"] = "multi_to_single"
                channel_info["method"] = "weighted_combination"

            elif num_channels > 4:
                # Multi-band astronomy data (e.g., SDSS ugriz)
                # Use median combination for robustness
                data = torch.median(data, dim=0, keepdim=True)[0]
                channel_info["conversion"] = "multiband_to_single"
                channel_info["method"] = "median_combination"

            else:
                # Unexpected number of channels
                logger.warning(
                    f"Unexpected channel count: {num_channels}, using first channel"
                )
                data = data[0:1]

        else:
            raise ValueError(f"Unexpected data dimensions: {data.shape}")

        # Ensure we have the target number of channels
        if data.shape[0] != self.target_channels:
            logger.warning(
                f"Channel mismatch after processing: {data.shape[0]} != {self.target_channels}"
            )
            if data.shape[0] > self.target_channels:
                data = data[: self.target_channels]
            else:
                # Repeat channels if needed
                data = data.repeat(self.target_channels, 1, 1)

        channel_info["final_channels"] = data.shape[0]
        return data, channel_info

    def preprocess_astronomy_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess a batch of astronomy data.

        Args:
            batch: Dictionary containing 'clean' and 'noisy' tensors

        Returns:
            Processed batch safe for Poisson-Gaussian modeling
        """
        processed_batch = {}

        for key in ["clean", "noisy"]:
            if key in batch:
                data = batch[key]
                metadata = batch.get(f"{key}_metadata", {})

                # Handle channel consistency first
                data, channel_info = self.handle_channel_consistency(data, metadata)

                # Then handle negative values
                data, processing_info = self.handle_negative_values(data, metadata)

                processed_batch[key] = data
                processed_batch[f"{key}_metadata"] = {
                    **metadata,
                    **channel_info,
                    **processing_info,
                }

        return processed_batch


class AstronomyCollateFunction:
    """
    Pickle-able collate function for astronomy data preprocessing.

    This class-based approach avoids the pickle issues with closures
    while maintaining the same functionality as the original closure-based approach.
    """

    def __init__(self, preprocessor: AstronomyDataPreprocessor):
        """
        Initialize the astronomy collate function.

        Args:
            preprocessor: AstronomyDataPreprocessor instance
        """
        self.preprocessor = preprocessor

    def __call__(self, batch):
        """Collate function that preprocesses astronomy data before batching."""
        # First, preprocess each item individually
        processed_items = []

        for item in batch:
            if isinstance(item, dict):
                # Handle dictionary format
                processed_item = {}

                for key, value in item.items():
                    if key in ["clean", "noisy"] and isinstance(value, torch.Tensor):
                        # Preprocess image data with extra safety checks
                        data, _ = self.preprocessor.handle_channel_consistency(value)

                        # Double-check for negative values (defensive programming)
                        if data.min() < 0:
                            logger.debug(
                                f"Still found negative values after channel processing for {key}, applying emergency offset"
                            )
                            offset = abs(data.min().item())
                            data = data + offset

                            # Store offset in metadata for potential inverse transform
                            if "metadata" not in processed_item:
                                processed_item["metadata"] = {}
                            processed_item["metadata"]["emergency_offset"] = offset

                        data, _ = self.preprocessor.handle_negative_values(data)
                        processed_item[key] = data
                    else:
                        processed_item[key] = value

                processed_items.append(processed_item)
            else:
                # Handle tuple or other formats
                processed_items.append(item)

        # Now stack the preprocessed items
        if processed_items and isinstance(processed_items[0], dict):
            collated = {}
            for key in processed_items[0].keys():
                values = [item[key] for item in processed_items]

                if isinstance(values[0], torch.Tensor):
                    # Ensure all tensors have the same shape before stacking
                    shapes = [v.shape for v in values]
                    if len(set(shapes)) > 1:
                        logger.warning(f"Inconsistent shapes for {key}: {shapes}")
                        # Pad or crop to match first item's shape
                        target_shape = shapes[0]
                        for i in range(len(values)):
                            if values[i].shape != target_shape:
                                # Simple crop/pad strategy
                                values[i] = torch.nn.functional.pad(
                                    values[i],
                                    [
                                        0,
                                        max(0, target_shape[-1] - values[i].shape[-1]),
                                        0,
                                        max(0, target_shape[-2] - values[i].shape[-2]),
                                        0,
                                        max(0, target_shape[-3] - values[i].shape[-3]),
                                    ],
                                )
                                if values[i].shape != target_shape:
                                    values[i] = values[i][
                                        : target_shape[0],
                                        : target_shape[1],
                                        : target_shape[2],
                                    ]

                    collated[key] = torch.stack(values)
                else:
                    collated[key] = values

            return collated
        else:
            # Fallback to default collate
            return torch.utils.data.default_collate(processed_items)


def create_astronomy_collate_fn(preprocessor: AstronomyDataPreprocessor):
    """
    Create a collate function that handles astronomy data preprocessing.

    Args:
        preprocessor: AstronomyDataPreprocessor instance

    Returns:
        AstronomyCollateFunction instance (pickle-able)
    """
    return AstronomyCollateFunction(preprocessor)


def reverse_astronomy_offset(data: torch.Tensor, metadata: Dict) -> torch.Tensor:
    """
    Reverse the offset applied to astronomy data for evaluation purposes.

    This function can be used to restore the original scale of processed
    astronomy images when computing evaluation metrics.

    Args:
        data: Processed data tensor
        metadata: Metadata dictionary containing offset information

    Returns:
        Data tensor with offset reversed
    """
    if "astronomy_offset" in metadata:
        offset = metadata["astronomy_offset"]
        return data - offset
    elif "emergency_offset" in metadata:
        offset = metadata["emergency_offset"]
        return data - offset
    else:
        # No offset was applied
        return data


# Example usage in training script
if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = AstronomyDataPreprocessor(
        offset_method="adaptive",
        target_channels=1,
        preserve_noise_statistics=True,
        min_offset=100.0,
    )

    # Test with sample data containing negative values
    test_data = torch.randn(4, 128, 128) - 0.5  # Some negative values
    print(f"Original data range: [{test_data.min():.2f}, {test_data.max():.2f}]")

    processed_data, info = preprocessor.handle_channel_consistency(test_data)
    print(f"After channel processing: shape={processed_data.shape}")

    processed_data, info = preprocessor.handle_negative_values(processed_data)
    print(
        f"After negative value handling: [{processed_data.min():.2f}, {processed_data.max():.2f}]"
    )
    print(f"Processing info: {info}")
