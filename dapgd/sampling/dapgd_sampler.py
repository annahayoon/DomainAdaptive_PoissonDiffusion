"""
DAPGD Guided Sampler

Implements Algorithm 1 from the DAPGD paper by extending EDM's sampling loop
with Poisson-Gaussian guidance.

ARCHITECTURE:
EDM Sampler (untouched) + PG Guidance (our contribution)

Key Features:
- Physics-informed guidance injection
- Domain-adaptive conditioning
- Proper noise schedule handling
- Gradient-based guidance application
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..guidance.pg_guidance import PoissonGaussianGuidance, create_guidance_from_domain
from .edm_wrapper import EDMModelWrapper

logger = logging.getLogger(__name__)


class DAPGDSampler:
    """
    Domain-Adaptive Poisson-Gaussian Diffusion Sampler

    Implements guided sampling for photon-limited imaging by injecting
    physics-informed guidance into EDM's proven sampling infrastructure.

    Args:
        edm_wrapper: EDM model wrapper with domain conditioning
        guidance: PG guidance module (if None, runs vanilla EDM)
        num_steps: Number of diffusion steps
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
        rho: EDM schedule parameter
        S_churn: Stochasticity (0 = deterministic)
        device: 'cuda' or 'cpu'

    Example:
        >>> sampler = DAPGDSampler(
        ...     edm_wrapper=model_wrapper,
        ...     guidance_config={'s': 1000, 'sigma_r': 5.0, 'kappa': 0.5}
        ... )
        >>> restored = sampler.sample(noisy_observation)
    """

    def __init__(
        self,
        edm_wrapper: EDMModelWrapper,
        guidance: Optional[PoissonGaussianGuidance] = None,
        num_steps: int = 50,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        S_churn: float = 0.0,
        device: str = "cuda",
    ):
        self.edm_wrapper = edm_wrapper
        self.guidance = guidance
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.S_churn = S_churn
        self.device = device

        # Pre-compute noise schedule
        self.sigmas = self._compute_sigmas()

        # Setup logging
        if guidance is not None:
            logger.info(f"Initialized DAPGD sampler with PG guidance: {guidance}")
        else:
            logger.info("Initialized DAPGD sampler without guidance (vanilla EDM)")

    def _compute_sigmas(self) -> torch.Tensor:
        """
        Compute EDM's geometric noise schedule

        Formula: σ_i = (σ_max^(1/ρ) + i/(N-1) * (σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ
        """

        ramp = np.linspace(0, 1, self.num_steps)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        sigmas = np.append(sigmas, 0)  # Append σ_0 = 0

        return torch.from_numpy(sigmas).float().to(self.device)

    @torch.no_grad()
    def sample(
        self,
        y_e: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        image_size: Tuple[int, int] = (256, 256),
        channels: int = 3,
        conditioning: Optional[Dict] = None,
        return_trajectory: bool = False,
        show_progress: bool = True,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample clean image(s) using guided diffusion

        Args:
            y_e: Noisy observation [B,C,H,W] in electrons
            batch_size: Number of samples (used if y_e is None)
            image_size: (H, W) if y_e is None
            channels: Number of channels if y_e is None
            conditioning: Domain conditioning dict (for domain-adaptive prior)
            return_trajectory: If True, return all intermediate x_t
            show_progress: Show progress bar
            seed: Random seed for reproducibility

        Returns:
            x_0: Restored/generated image(s) [B,C,H,W] in [0,1] range
            (optionally) trajectory: List of intermediate states
        """

        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        # Determine shape and validate guidance
        if y_e is not None:
            shape = y_e.shape
            batch_size = shape[0]
            y_e = y_e.to(self.device)

            # Validate that guidance is available if y_e is provided
            if self.guidance is None:
                logger.warning(
                    "Observation y_e provided but guidance is disabled. "
                    "Running unconditional generation instead."
                )
        else:
            shape = (batch_size, channels, image_size[0], image_size[1])

        logger.info(f"Starting DAPGD sampling with shape {shape}")

        # Initialize from noise
        x_t = torch.randn(shape, device=self.device) * self.sigmas[0]

        # Storage for trajectory
        trajectory = [x_t.cpu()] if return_trajectory else None

        # Prepare progress bar
        iterator = range(self.num_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="DAPGD Sampling", unit="step")

        # Main sampling loop (Algorithm 1 from paper)
        for i in iterator:
            t_cur = self.sigmas[i].item()
            t_next = self.sigmas[i + 1].item()

            # Take one EDM step with guidance
            x_t = self.sample_step(
                x_t, t_cur, t_next, y_e=y_e, conditioning=conditioning
            )

            # Store trajectory if requested
            if return_trajectory:
                trajectory.append(x_t.cpu())

            # Update progress bar with metrics
            if show_progress and isinstance(iterator, tqdm):
                iterator.set_postfix(
                    {
                        "sigma": f"{t_cur:.3f}",
                        "min": f"{x_t.min():.2f}",
                        "max": f"{x_t.max():.2f}",
                    }
                )

        # Final output
        x_0 = x_t

        logger.info("DAPGD sampling complete")

        if return_trajectory:
            return x_0, trajectory
        else:
            return x_0

    def sample_step(
        self,
        x_cur: torch.Tensor,
        t_cur: float,
        t_next: float,
        y_e: Optional[torch.Tensor] = None,
        conditioning: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Single sampling step with optional guidance

        Args:
            x_cur: Current sample [B,C,H,W]
            t_cur: Current noise level (sigma)
            t_next: Next noise level
            y_e: Noisy observation for guidance
            conditioning: Domain conditioning parameters

        Returns:
            x_next: Sample at next step
        """

        # This implements EDM's Heun sampler (2nd order) with guidance injection

        # Extract domain parameters from conditioning
        domain_params = self._extract_domain_params(conditioning)

        # Add stochasticity if S_churn > 0
        gamma = (
            min(self.S_churn / self.num_steps, np.sqrt(2) - 1)
            if self.S_churn > 0
            else 0
        )
        t_hat = t_cur + gamma * t_cur

        if gamma > 0:
            epsilon = torch.randn_like(x_cur)
            x_hat = x_cur + np.sqrt(t_hat**2 - t_cur**2) * epsilon
        else:
            x_hat = x_cur

        # Get denoised prediction from EDM
        denoised = self._denoise_with_guidance(x_hat, t_hat, y_e, domain_params)

        # Euler step
        d_cur = (x_hat - denoised) / t_hat if t_hat > 0 else torch.zeros_like(x_hat)
        x_next = x_hat + (t_next - t_hat) * d_cur

        # 2nd order correction (Heun)
        if t_next > 0:
            denoised_next = self._denoise_with_guidance(
                x_next, t_next, y_e, domain_params
            )
            d_next = (x_next - denoised_next) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_next)

        return x_next

    def _denoise_with_guidance(
        self,
        x: torch.Tensor,
        sigma: float,
        y_e: Optional[torch.Tensor],
        domain_params: Optional[Dict],
    ) -> torch.Tensor:
        """
        Get denoised prediction with optional guidance

        Args:
            x: Noisy sample
            sigma: Current noise level
            y_e: Noisy observation (for guidance)
            domain_params: Domain parameters for conditioning

        Returns:
            Denoised prediction (possibly guided)
        """

        # Get EDM prediction
        if domain_params:
            # Use domain conditioning
            denoised = self.edm_wrapper(
                x,
                torch.full((x.shape[0],), sigma, device=x.device),
                domain=domain_params["domain"],
                scale=domain_params["s"],
                read_noise=domain_params["sigma_r"],
                background=domain_params["background"],
            )
        else:
            # No domain conditioning
            denoised = self.edm_wrapper(
                x, torch.full((x.shape[0],), sigma, device=x.device)
            )

        # Apply PG guidance if available and y_e provided
        if self.guidance is not None and y_e is not None:
            # Apply guidance to denoised prediction
            denoised = self.guidance(denoised, y_e, sigma)

        return denoised

    def _extract_domain_params(self, conditioning: Optional[Dict]) -> Optional[Dict]:
        """
        Extract domain parameters from conditioning dict

        Args:
            conditioning: Dict with keys {domain_type, s, sigma_r, b}

        Returns:
            Domain parameters dict or None
        """

        if conditioning is None:
            return None

        # Extract parameters with defaults
        domain_params = {
            "domain": conditioning.get("domain_type", "photo"),
            "s": conditioning.get("s", 1000.0),
            "sigma_r": conditioning.get("sigma_r", 5.0),
            "background": conditioning.get("background", 0.0),
        }

        return domain_params

    def compute_chi_squared(
        self,
        x_restored: torch.Tensor,
        y_observed: torch.Tensor,
        domain_params: Optional[Dict] = None,
    ) -> float:
        """
        Compute reduced chi-squared statistic for physical validation

        Args:
            x_restored: Restored image [B,C,H,W] in [0,1] range
            y_observed: Noisy observation [B,C,H,W] in electrons
            domain_params: Domain parameters for variance computation

        Returns:
            chi2_reduced: Reduced chi-squared value
        """

        if self.guidance is None:
            logger.warning("Chi-squared requires guidance configuration")
            return float("nan")

        # Use domain parameters if provided, otherwise use guidance defaults
        if domain_params:
            s = domain_params["s"]
            sigma_r = domain_params["sigma_r"]
        else:
            s = self.guidance.s.item()
            sigma_r = self.guidance.sigma_r.item()

        # Compute chi-squared using guidance utility
        from ..guidance.pg_guidance import compute_chi_squared as compute_chi2

        chi2 = compute_chi2(x_restored, y_observed, s, sigma_r)

        return chi2

    def analyze_residuals(
        self,
        x_restored: torch.Tensor,
        y_observed: torch.Tensor,
        domain_params: Optional[Dict] = None,
    ) -> Dict:
        """
        Analyze residual statistics for diagnostics

        Args:
            x_restored: Restored image [B,C,H,W]
            y_observed: Noisy observation [B,C,H,W]
            domain_params: Domain parameters

        Returns:
            Dictionary with residual statistics
        """

        if self.guidance is None:
            logger.warning("Residual analysis requires guidance configuration")
            return {}

        # Use domain parameters if provided
        if domain_params:
            s = domain_params["s"]
            sigma_r = domain_params["sigma_r"]
        else:
            s = self.guidance.s.item()
            sigma_r = self.guidance.sigma_r.item()

        # Compute residual statistics
        from ..guidance.pg_guidance import analyze_residual_statistics

        stats = analyze_residual_statistics(x_restored, y_observed, s, sigma_r)

        return stats


# Factory function for easy sampler creation


def create_dapgd_sampler(
    edm_config: Dict[str, any],
    guidance_config: Optional[Dict] = None,
    domain: Optional[str] = None,
    device: str = "cuda",
    **kwargs,
) -> DAPGDSampler:
    """
    Factory function to create DAPGD sampler from configuration

    Args:
        edm_config: EDM model configuration
        guidance_config: PG guidance configuration
        domain: Domain type for automatic guidance creation
        device: Target device
        **kwargs: Additional sampler parameters

    Returns:
        Configured DAPGDSampler instance

    Example:
        >>> sampler = create_dapgd_sampler(
        ...     edm_config={'img_resolution': 256, 'img_channels': 3},
        ...     guidance_config={'s': 1000, 'sigma_r': 5.0, 'kappa': 0.5},
        ...     domain='photo'
        ... )
    """

    # Create EDM wrapper
    edm_wrapper = EDMModelWrapper(edm_config=edm_config, device=device)

    # Create guidance
    guidance = None
    if guidance_config is not None:
        if domain is not None:
            # Create guidance with domain-appropriate parameters
            guidance = create_guidance_from_domain(domain, **guidance_config)
        else:
            # Create guidance with provided parameters
            guidance = PoissonGaussianGuidance(**guidance_config)

    # Create sampler
    sampler = DAPGDSampler(
        edm_wrapper=edm_wrapper, guidance=guidance, device=device, **kwargs
    )

    return sampler


# Test function for integration


def test_dapgd_sampler():
    """Test DAPGD sampler functionality"""

    logger.info("Testing DAPGD sampler...")

    try:
        # Create minimal test setup
        edm_config = {
            "img_resolution": 32,  # Very small for testing
            "img_channels": 1,
            "model_channels": 32,
            "channel_mult": (1, 2),
            "num_blocks": 2,
            "label_dim": 6,
        }

        guidance_config = {"s": 1000.0, "sigma_r": 5.0, "kappa": 0.5, "tau": 0.01}

        # Create sampler
        sampler = create_dapgd_sampler(
            edm_config=edm_config,
            guidance_config=guidance_config,
            num_steps=5,  # Very few steps for testing
            device="cpu",
        )

        logger.info("✓ DAPGD sampler created successfully")

        # Test unconditional sampling
        samples = sampler.sample(
            batch_size=1, image_size=(32, 32), channels=1, show_progress=False
        )

        logger.info(f"✓ Unconditional sampling: {samples.shape}")
        assert samples.shape == (1, 1, 32, 32)
        assert not torch.isnan(samples).any()

        # Test guided sampling with synthetic data
        clean = torch.rand(1, 1, 32, 32)
        noisy = torch.poisson(clean * 1000) + 5.0 * torch.randn_like(clean)

        restored = sampler.sample(y_e=noisy, show_progress=False)

        logger.info(f"✓ Guided sampling: {restored.shape}")
        assert restored.shape == clean.shape
        assert not torch.isnan(restored).any()

        # Test chi-squared computation
        chi2 = sampler.compute_chi_squared(restored, noisy)
        logger.info(f"✓ Chi-squared computation: {chi2:.4f}")
        assert not np.isnan(chi2)
        assert chi2 > 0

        logger.info("✅ All DAPGD sampler tests passed!")
        return True

    except Exception as e:
        logger.error(f"❌ DAPGD sampler test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run integration test
    success = test_dapgd_sampler()
    if success:
        print("\n🎉 DAPGD sampler integration test PASSED!")
        print("Core sampling pipeline is working correctly.")
    else:
        print("\n❌ DAPGD sampler integration test FAILED!")
        print("Check logs above for details.")
