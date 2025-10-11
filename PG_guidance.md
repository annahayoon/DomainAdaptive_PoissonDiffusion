Poisson-Gaussian Guidance for Diffusion Models

This module implements Equation 3 from the paper:

The score (gradient) of the Poisson-Gaussian log-likelihood.

KEY INSIGHT: The variance in photon-limited imaging is signal-dependent:

Var[y|x] = s·x + σ_r²


This heteroscedasticity requires adaptive weighting - we cannot use uniform L2 loss.

"""

import torch

import torch.nn as nn

from typing import Literal, Optional

import logging

logger = logging.getLogger(__name__)

class PoissonGaussianGuidance(nn.Module):

"""

Physics-informed guidance for photon-limited imaging


Implements the score of the Poisson-Gaussian likelihood:

∇_x log p(y_e|x)


This tells the diffusion model how to adjust predictions to match

observed noisy measurements while respecting physical noise properties.


Args:

s: Scale factor (max photon count, typically full-well capacity)

sigma_r: Read noise standard deviation (in electrons)

kappa: Guidance strength multiplier (typically 0.3-1.0)

tau: Guidance threshold - only apply when σ_t > tau

mode: 'wls' for weighted least squares, 'full' for complete gradient

epsilon: Small constant for numerical stability


Example:

>>> guidance = PoissonGaussianGuidance(s=1000, sigma_r=5.0, kappa=0.5)

>>> x_guided = guidance(x_pred, y_observed, sigma_t=0.1)
