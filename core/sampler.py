"""EDM Sampler with native and solver-based modes."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from config.config import SamplingConfig
from config.sample_config import DEFAULT_SENSOR_RANGES
from core.edm_wrapper import EDMModelWrapper
from core.error_handlers import ConfigurationError, SamplingError
from core.guidance import GaussianGuidance, PoissonGaussianGuidance
from core.normalization import convert_range
from core.utils.data_utils import apply_exposure_scaling, load_tensor
from core.utils.sampling_utils import (
    create_edm_noise_schedule,
    prepare_conditioning,
    should_apply_guidance,
)
from core.utils.tensor_utils import get_device


class EDMSampler:
    """EDM sampler supporting both native and solver-based modes."""

    def __init__(
        self,
        model: Optional[EDMModelWrapper] = None,
        guidance: Optional[Union[PoissonGaussianGuidance, GaussianGuidance]] = None,
        config: Optional[SamplingConfig] = None,
        model_path: Optional[str] = None,
        device: str = "cuda",
        sensor_ranges: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """
        Initialize EDM sampler.

        Args:
            model: EDM model wrapper (required if model_path not provided)
            guidance: Guidance module (optional)
            config: Sampling configuration (optional)
            model_path: Path to checkpoint file (alternative to model parameter)
            device: Device to use ("cuda" or "cpu")
            sensor_ranges: Sensor-specific parameter ranges (optional)
        """
        self.config = config or SamplingConfig()
        self.device = get_device(prefer_cuda=(device == "cuda"))
        self.exposure_ratio = self.config.exposure_ratio
        self.sensor_ranges = DEFAULT_SENSOR_RANGES.copy()
        if sensor_ranges:
            self.sensor_ranges.update(sensor_ranges)

        # Load model from checkpoint if model_path provided
        if model_path is not None:
            checkpoint = load_tensor(
                Path(model_path), device=None, map_location="cpu", weights_only=False
            )
            # Extract model from checkpoint
            if "ema" in checkpoint:
                self.model = checkpoint["ema"].to(self.device)
            elif "model" in checkpoint:
                self.model = checkpoint["model"].to(self.device)
            else:
                raise ConfigurationError(
                    f"Checkpoint must contain 'ema' or 'model' key. Found keys: {list(checkpoint.keys())}"
                )
            self.model.eval()
        elif model is not None:
            self.model = model
        else:
            raise ConfigurationError(
                "Must provide either 'model' or 'model_path' parameter"
            )

        self.guidance = guidance

        if model is not None and guidance is not None:
            self._validate_components()

        self.diagnostics = {}
        self.intermediates = []

    def _validate_components(self):
        """Validate model and guidance compatibility."""
        if not isinstance(self.model, EDMModelWrapper):
            if not callable(self.model):
                raise ConfigurationError(
                    "Model must be EDMModelWrapper or callable with compatible signature"
                )

        if self.guidance is not None:
            if not isinstance(
                self.guidance, (PoissonGaussianGuidance, GaussianGuidance)
            ):
                if not hasattr(self.guidance, "compute") or not callable(
                    self.guidance.compute
                ):
                    raise ConfigurationError(
                        "Guidance must be PoissonGaussianGuidance, GaussianGuidance, "
                        "or have a compute() method"
                    )

    def create_noise_schedule(self, num_steps: Optional[int] = None) -> torch.Tensor:
        """Create EDM noise schedule."""
        if num_steps is None:
            num_steps = self.config.num_steps

        return create_edm_noise_schedule(
            num_steps=num_steps,
            sigma_min=self.config.sigma_min,
            sigma_max=self.config.sigma_max,
            rho=self.config.rho,
        )

    def sample(
        self,
        y_observed: torch.Tensor,
        domain: Optional[Union[str, List[str], torch.Tensor]] = None,
        scale: Optional[Union[float, torch.Tensor]] = None,
        read_noise: Optional[Union[float, torch.Tensor]] = None,
        background: Optional[Union[float, torch.Tensor]] = None,
        condition: Optional[torch.Tensor] = None,
        conditioning_type: str = "dapgd",
        x_init: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        exposure_ratio: Optional[float] = None,
        apply_exposure_scaling: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample from posterior distribution.

        Args:
            y_observed: Observed noisy image [B, C, H, W] (electrons)
            domain: Domain specification (if condition not provided)
            scale: Scale parameter (if condition not provided)
            read_noise: Read noise parameter (if condition not provided)
            background: Background parameter (if condition not provided)
            condition: Pre-computed conditioning vector [B, 6]
            conditioning_type: Type of conditioning ('dapgd' or 'l2')
            x_init: Initial sample (uses random if None)
            mask: Valid pixel mask [B, C, H, W]
            exposure_ratio: Exposure ratio for scaling (overrides config)
            apply_exposure_scaling: Whether to apply exposure scaling to x_init
            **kwargs: Additional arguments

        Returns:
            Dictionary with sampling results
        """
        # Update exposure ratio if provided
        if exposure_ratio is not None:
            self.exposure_ratio = exposure_ratio
            if self.guidance is not None and hasattr(self.guidance, "alpha"):
                self.guidance.alpha = exposure_ratio

        # Prepare conditioning
        condition = prepare_conditioning(
            model=self.model,
            condition=condition,
            scale=scale,
            read_noise=read_noise,
            background=background,
            device=y_observed.device,
            conditioning_type=conditioning_type,
        )

        # Handle exposure scaling for x_init
        if x_init is None and apply_exposure_scaling and self.exposure_ratio != 1.0:
            x_init = (
                apply_exposure_scaling(y_observed, self.exposure_ratio)
                .to(torch.float64)
                .to(y_observed.device)
            )

        # Choose sampling method based on solver type
        if self.config.solver == "native":
            result = self._sample_native(
                y_observed=y_observed,
                condition=condition,
                x_init=x_init,
                mask=mask,
                **kwargs,
            )
        else:
            result = self._sample_solver_based(
                y_observed=y_observed,
                condition=condition,
                x_init=x_init,
                mask=mask,
                **kwargs,
            )

        return result

    def _sample_native(
        self,
        y_observed: torch.Tensor,
        condition: torch.Tensor,
        x_init: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Sample using native EDM algorithm."""
        batch_size, channels, height, width = y_observed.shape
        device = y_observed.device

        if x_init is None:
            latents = torch.randn(batch_size, channels, height, width, device=device)
        else:
            latents = x_init

        # Reset diagnostics
        if self.config.collect_diagnostics:
            self.diagnostics = {
                "sigma_values": [],
                "guidance_norms": [],
                "denoised_norms": [],
                "step_sizes": [],
            }

        if self.config.save_intermediates:
            self.intermediates = []

        result = self._guided_edm_sampler(
            net=self.model,
            latents=latents,
            class_labels=condition,
            y_observed=y_observed,
            mask=mask,
            num_steps=self.config.num_steps,
            sigma_min=self.config.sigma_min,
            sigma_max=self.config.sigma_max,
            rho=self.config.rho,
            S_churn=self.config.S_churn,
            S_min=self.config.S_min,
            S_max=self.config.S_max,
            S_noise=self.config.S_noise,
            **kwargs,
        )

        if self.config.clip_denoised:
            result = torch.clamp(result, *self.config.clip_range)

        results = {
            "x_final": result,
            "condition": condition,
        }

        if self.config.collect_diagnostics:
            results["diagnostics"] = self.diagnostics

        if self.config.save_intermediates:
            results["intermediates"] = self.intermediates

        return results

    def _sample_solver_based(
        self,
        y_observed: torch.Tensor,
        condition: torch.Tensor,
        x_init: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Sample using flexible solver-based approach."""
        batch_size, channels, height, width = y_observed.shape
        device = y_observed.device

        if x_init is None:
            x_init = torch.randn(batch_size, channels, height, width, device=device)

        # Create noise schedule
        sigma_schedule = self.create_noise_schedule().to(device)

        # Reset diagnostics
        if self.config.collect_diagnostics:
            self.diagnostics = {
                "sigma_values": [],
                "guidance_norms": [],
                "denoised_norms": [],
                "step_sizes": [],
            }

        if self.config.save_intermediates:
            self.intermediates = []

        x = x_init * sigma_schedule[0]

        with torch.no_grad():
            for i in tqdm(range(len(sigma_schedule) - 1), desc="Sampling"):
                sigma_curr = sigma_schedule[i]
                sigma_next = sigma_schedule[i + 1]

                # Compute denoised prediction
                x_denoised = self._denoise_step(
                    x, sigma_curr, condition, y_observed, mask, i
                )

                if self._should_apply_guidance(i):
                    x_denoised_before_guidance = x_denoised.clone()
                    x_denoised = self._apply_guidance(
                        x_denoised, y_observed, sigma_curr, mask
                    )

                    if self.config.collect_diagnostics:
                        guidance_correction = x_denoised - x_denoised_before_guidance
                        self.diagnostics["guidance_norms"].append(
                            torch.norm(guidance_correction).item()
                        )
                else:
                    if self.config.collect_diagnostics:
                        self.diagnostics["guidance_norms"].append(0.0)

                d = (x - x_denoised) / sigma_curr

                # Apply score-level guidance if needed
                if (
                    self.config.guidance_level == "score"
                    and self._should_apply_guidance(i)
                    and self.guidance is not None
                ):
                    guidance_contribution = self._apply_score_guidance(
                        x_denoised, y_observed, sigma_curr, mask
                    )
                    if guidance_contribution is not None:
                        d = d - sigma_curr * guidance_contribution

                # Take sampling step
                x = self._take_step(x, d, sigma_curr, sigma_next, i)

                # Collect diagnostics
                if self.config.collect_diagnostics:
                    self.diagnostics["sigma_values"].append(sigma_curr.item())
                    self.diagnostics["denoised_norms"].append(
                        torch.norm(x_denoised).item()
                    )
                    self.diagnostics["step_sizes"].append(
                        torch.norm(x - (x - d * (sigma_next - sigma_curr))).item()
                    )

                # Save intermediate
                if self.config.save_intermediates:
                    self.intermediates.append(
                        {
                            "step": i,
                            "sigma": sigma_curr.item(),
                            "x": x.clone(),
                            "x_denoised": x_denoised.clone(),
                        }
                    )

        x_final = self._denoise_step(
            x, sigma_schedule[-1], condition, y_observed, mask, len(sigma_schedule) - 1
        )

        # Apply final guidance if needed
        if self._should_apply_guidance(len(sigma_schedule) - 1):
            x_final = self._apply_guidance(
                x_final, y_observed, sigma_schedule[-1], mask
            )

        if self.config.clip_denoised:
            x_final = torch.clamp(x_final, *self.config.clip_range)

        results = {
            "x_final": x_final,
            "x_init": x_init,
            "sigma_schedule": sigma_schedule,
            "condition": condition,
        }

        if self.config.collect_diagnostics:
            results["diagnostics"] = self.diagnostics

        if self.config.save_intermediates:
            results["intermediates"] = self.intermediates

        return results

    def _guided_edm_sampler(
        self,
        net,
        latents,
        class_labels=None,
        y_observed=None,
        mask=None,
        randn_like=torch.randn_like,
        num_steps=18,
        sigma_min=0.002,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
    ):
        """EDM native sampler with integrated guidance."""
        sigma_min = max(sigma_min, net.sigma_min)
        sigma_max = min(sigma_max, net.sigma_max)

        step_indices = torch.arange(
            num_steps, dtype=torch.float64, device=latents.device
        )
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        x_next = latents.to(torch.float64) * t_steps[0]

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next

            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= t_cur <= S_max
                else 0
            )
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(
                x_cur
            )

            denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
            if self._should_apply_guidance(i) and y_observed is not None:
                denoised_before_guidance = denoised.clone()
                denoised = self._apply_guidance(denoised, y_observed, t_hat, mask)

                if self.config.collect_diagnostics:
                    guidance_correction = denoised - denoised_before_guidance
                    self.diagnostics["guidance_norms"].append(
                        torch.norm(guidance_correction).item()
                    )
            else:
                if self.config.collect_diagnostics:
                    self.diagnostics["guidance_norms"].append(0.0)

            if self.config.collect_diagnostics:
                self.diagnostics["sigma_values"].append(t_hat.item())
                self.diagnostics["denoised_norms"].append(torch.norm(denoised).item())

            if self.config.save_intermediates:
                self.intermediates.append(
                    {
                        "step": i,
                        "sigma": t_hat.item(),
                        "x": x_hat.clone(),
                        "x_denoised": denoised.clone(),
                    }
                )

            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            if not self.config.no_heun and i < num_steps - 1:
                denoised = net(x_next, t_next, class_labels).to(torch.float64)
                if self._should_apply_guidance(i) and y_observed is not None:
                    denoised = self._apply_guidance(denoised, y_observed, t_next, mask)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    def _denoise_step(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        condition: torch.Tensor,
        y_observed: torch.Tensor,
        mask: Optional[torch.Tensor],
        step: int,
    ) -> torch.Tensor:
        """Perform denoising step using the model."""
        if sigma.dim() == 0:
            sigma = sigma.expand(x.shape[0])
        return self.model(x, sigma, condition=condition)

    def _should_apply_guidance(self, step: int) -> bool:
        """Check if guidance should be applied at current step."""
        if self.guidance is None:
            return False

        return should_apply_guidance(
            step=step,
            guidance_start_step=self.config.guidance_start_step,
            guidance_end_step=self.config.guidance_end_step,
        )

    def _apply_guidance(
        self,
        denoised: torch.Tensor,
        y_observed: torch.Tensor,
        sigma: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply physics-aware guidance to denoised prediction."""
        if self.guidance is None:
            return denoised

        # Handle guidance level: "x0" modifies denoised estimate, "score" modifies score
        if self.config.guidance_level == "x0":
            # Convert to [0,1] range for guidance computation
            denoised_01 = torch.clamp(
                convert_range(denoised, "[-1,1]", "[0,1]"), 0.0, 1.0
            )

            if (
                hasattr(self.guidance, "guidance_level")
                and self.guidance.guidance_level == "x0"
            ):
                # Use guidance's forward method which handles x0-level guidance
                guided_01 = self.guidance(denoised_01, y_observed, sigma.item())
                return torch.clamp(
                    convert_range(guided_01, "[0,1]", "[-1,1]"), -1.0, 1.0
                )
            elif self.config.guidance_type == "pg":
                guidance_correction = self.guidance.compute(
                    denoised_01, y_observed, sigma, mask=mask
                )
                guidance_correction = guidance_correction * self.config.guidance_scale
                guided_01 = denoised_01 + guidance_correction
                return torch.clamp(
                    convert_range(guided_01, "[0,1]", "[-1,1]"), -1.0, 1.0
                )
            elif self.config.guidance_type == "l2":
                guidance_correction = y_observed - denoised_01
                guidance_correction = guidance_correction * self.config.guidance_scale
                guided_01 = denoised_01 + guidance_correction
                return torch.clamp(
                    convert_range(guided_01, "[0,1]", "[-1,1]"), -1.0, 1.0
                )
            else:
                raise ConfigurationError(
                    f"Unknown guidance type: {self.config.guidance_type}"
                )
        else:  # guidance_level == "score"
            # Score-level guidance is handled in _sample_solver_based
            return denoised

    def _apply_score_guidance(
        self,
        denoised: torch.Tensor,
        y_observed: torch.Tensor,
        sigma: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Apply score-level guidance (returns guidance contribution to score)."""
        if self.guidance is None:
            return None

        if (
            hasattr(self.guidance, "guidance_level")
            and self.guidance.guidance_level == "score"
        ):
            # Convert to [0,1] range for guidance computation
            denoised_01 = torch.clamp(
                convert_range(denoised, "[-1,1]", "[0,1]"), 0.0, 1.0
            )
            return self.guidance.kappa * self.guidance.compute_likelihood_gradient(
                denoised_01, y_observed
            )
        else:
            # For guidance without score-level support, return None
            return None

    def _take_step(
        self,
        x: torch.Tensor,
        d: torch.Tensor,
        sigma_curr: torch.Tensor,
        sigma_next: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """Take sampling step using specified solver."""
        dt = sigma_next - sigma_curr

        if self.config.solver == "euler":
            x_next = x + d * dt
        elif self.config.solver == "heun":
            if self.config.no_heun:
                # Skip Heun correction step
                x_next = x + d * dt
            else:
                x_euler = x + d * dt
                if sigma_next > 0:
                    d_next = d
                    x_next = x + (d + d_next) * dt / 2
                else:
                    x_next = x_euler
        elif self.config.solver == "dpm":
            x_next = x + d * dt
        else:
            raise SamplingError(f"Unknown solver: {self.config.solver}")

        if (
            self.config.S_churn > 0
            and self.config.S_min <= sigma_curr <= self.config.S_max
        ):
            gamma = min(self.config.S_churn / self.config.num_steps, np.sqrt(2) - 1)
            sigma_hat = sigma_curr * (1 + gamma)
            if sigma_hat > sigma_curr:
                noise = torch.randn_like(x) * self.config.S_noise
                x_next = x_next + noise * np.sqrt(sigma_hat**2 - sigma_curr**2)

        return x_next

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get sampling diagnostics."""
        if not self.config.collect_diagnostics:
            return {"diagnostics_disabled": True}

        if not self.diagnostics:
            return {"no_sampling_performed": True}

        # Compute summary statistics
        diagnostics = {}

        for key, values in self.diagnostics.items():
            if values:
                diagnostics[f"{key}_mean"] = float(np.mean(values))
                diagnostics[f"{key}_std"] = float(np.std(values))
                diagnostics[f"{key}_max"] = float(np.max(values))
                diagnostics[f"{key}_min"] = float(np.min(values))

        # Add configuration info
        diagnostics["config"] = {
            "num_steps": self.config.num_steps,
            "solver": self.config.solver,
            "guidance_type": self.config.guidance_type,
            "guidance_scale": self.config.guidance_scale,
            "sigma_min": self.config.sigma_min,
            "sigma_max": self.config.sigma_max,
        }

        return diagnostics

    def estimate_sampling_time(
        self, batch_size: int = 1, image_size: int = 128
    ) -> Dict[str, float]:
        """Estimate sampling time for given parameters."""
        base_time_per_step = 0.1
        size_factor = (image_size / 128) ** 2
        batch_factor = batch_size
        model_time = (
            base_time_per_step * size_factor * batch_factor * self.config.num_steps
        )
        guidance_time = model_time * 0.1
        total_time = model_time + guidance_time

        return {
            "model_time_sec": model_time,
            "guidance_time_sec": guidance_time,
            "total_time_sec": total_time,
            "time_per_step_sec": total_time / self.config.num_steps,
            "batch_size": batch_size,
            "image_size": image_size,
        }


# Factory functions


def create_edm_sampler(
    model: EDMModelWrapper,
    guidance: Optional[Union[PoissonGaussianGuidance, GaussianGuidance]] = None,
    num_steps: int = 18,
    guidance_scale: float = 1.0,
    solver: str = "heun",
    guidance_type: str = "pg",
    **config_overrides,
) -> EDMSampler:
    """Create EDM sampler with specified configuration."""
    config = SamplingConfig(
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        solver=solver,
        guidance_type=guidance_type,
        **config_overrides,
    )
    return EDMSampler(model=model, guidance=guidance, config=config)


def create_fast_sampler(
    model: EDMModelWrapper,
    guidance: Optional[Union[PoissonGaussianGuidance, GaussianGuidance]] = None,
    guidance_type: str = "pg",
    **config_overrides,
) -> EDMSampler:
    """Create fast sampler with reduced steps."""
    config = SamplingConfig(
        num_steps=10,
        solver="euler",
        guidance_scale=0.8,
        guidance_type=guidance_type,
        collect_diagnostics=False,
        **config_overrides,
    )
    return EDMSampler(model=model, guidance=guidance, config=config)


def create_high_quality_sampler(
    model: EDMModelWrapper,
    guidance: Optional[Union[PoissonGaussianGuidance, GaussianGuidance]] = None,
    guidance_type: str = "pg",
    **config_overrides,
) -> EDMSampler:
    """Create high-quality sampler with more steps and diagnostics."""
    config = SamplingConfig(
        num_steps=50,
        solver="heun",
        guidance_scale=1.2,
        guidance_type=guidance_type,
        collect_diagnostics=True,
        save_intermediates=True,
        **config_overrides,
    )
    return EDMSampler(model=model, guidance=guidance, config=config)


def create_native_sampler(
    model: EDMModelWrapper,
    guidance: Optional[Union[PoissonGaussianGuidance, GaussianGuidance]] = None,
    num_steps: int = 18,
    guidance_scale: float = 1.0,
    guidance_type: str = "pg",
    **config_overrides,
) -> EDMSampler:
    """Create native EDM sampler (exact EDM algorithm)."""
    config = SamplingConfig(
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        solver="native",
        guidance_type=guidance_type,
        **config_overrides,
    )
    return EDMSampler(model=model, guidance=guidance, config=config)


def create_fast_native_sampler(
    model: EDMModelWrapper,
    guidance: Optional[Union[PoissonGaussianGuidance, GaussianGuidance]] = None,
    guidance_type: str = "pg",
    **config_overrides,
) -> EDMSampler:
    """Create fast native EDM sampler."""
    config = SamplingConfig(
        num_steps=10,
        solver="native",
        S_churn=0.0,
        guidance_scale=0.8,
        guidance_type=guidance_type,
        **config_overrides,
    )
    return EDMSampler(model=model, guidance=guidance, config=config)


def create_high_quality_native_sampler(
    model: EDMModelWrapper,
    guidance: Optional[Union[PoissonGaussianGuidance, GaussianGuidance]] = None,
    guidance_type: str = "pg",
    **config_overrides,
) -> EDMSampler:
    """Create high-quality native EDM sampler."""
    config = SamplingConfig(
        num_steps=50,
        solver="native",
        S_churn=0.0,
        guidance_scale=1.2,
        guidance_type=guidance_type,
        **config_overrides,
    )
    return EDMSampler(model=model, guidance=guidance, config=config)


# Utility functions


def sample_batch(
    sampler: EDMSampler,
    y_observed_batch: torch.Tensor,
    domain_batch: List[str],
    scale_batch: List[float],
    read_noise_batch: List[float],
    background_batch: List[float],
    **kwargs,
) -> List[Dict[str, torch.Tensor]]:
    """Sample a batch of images with different parameters."""
    results = []
    for i in range(y_observed_batch.shape[0]):
        result = sampler.sample(
            y_observed=y_observed_batch[i : i + 1],
            domain=domain_batch[i],
            scale=scale_batch[i],
            read_noise=read_noise_batch[i],
            background=background_batch[i],
            **kwargs,
        )
        results.append(result)
    return results
