Excellent point. A good lab notebook should be reproducible and self-contained. Vague references like "the bug" are a liability.

I have revised the to-do list to be explicit, detailing the exact nature of the numerical instability we discovered and framing it as a deliberate, crucial experiment. This version will stand on its own for future reference.

-----

## **`evaluation_enhancement_todos.md`**

**Objective:** This document outlines the next set of targeted experiments following the resolution of the critical numerical instability. Our goal is to transform our debugging insights into a compelling, evidence-based narrative for the ICLR paper, demonstrating the necessity of a physically-principled approach to guided diffusion.

### 1\. The "Coordinate System Mismatch" Ablation Study

This experiment will formally document the catastrophic failure that occurs when the model's normalized prediction space is naively mixed with the physical measurement space of the guidance term. This is no longer a "bug," but our most critical ablation study, proving the core thesis of our paper.

  * **Action:** Implement and run a "naive" sampler that contains the specific flaw we discovered. In this sampler, the guidance term is calculated in **physical space** using the real observation `y_e`, but it is then *incorrectly* added directly to the model's predicted `x_hat_0`, which exists in a **normalized [0, 1] space**.

    ```python
    # Pseudocode of the INCORRECT "naive" sampling logic to reproduce:

    # 1. Model predicts x_hat_0 in normalized [0,1] space
    x_hat_0_normalized = model(x_t, sigma_t, condition)

    # 2. Guidance is calculated in PHYSICAL space, using un-normalized y_e
    #    and a normalized x_hat_0. This creates a massive scale mismatch.
    gradient = scale * (y_observed - x_hat_0_normalized) / (x_hat_0_normalized + σ_r²) # <-- THE FLAW

    # 3. The exploding gradient is applied, causing numerical instability
    x_hat_0_guided = x_hat_0_normalized + κ*σ_t² * gradient 
    ```

  * **Goal:** Generate quantitative results (e.g., exploding loss values, non-physical metrics) and qualitative visualizations (images that diverge to NaN/infinity) that decisively demonstrate the failure.

  * **Paper Location:** **Main Paper, Experimental Section.** This will be the first experiment we present. It serves to motivate the problem that our proposed method solves.

-----

### 2\. Characterize Sensitivity to Guidance Strength ($\\kappa$)

With a now-stable sampler, we can properly analyze the impact of guidance strength. This demonstrates the controllability and robustness of our physics-correct approach.

  * **Action:** Using the correct **"Un-normalize, Guide, Re-normalize"** sampler, run a full evaluation across a sweep of `kappa` values (e.g., `[0.0, 0.25, 0.5, 1.0, 1.5, 2.0]`).
  * **Goal:** Produce plots of key image quality metrics (e.g., PSNR, SSIM, LPIPS) as a function of `kappa` for each domain. This will allow us to identify the optimal guidance strength and discuss the trade-offs between fidelity and potential artifacts.
  * **Paper Location:** **Main Paper, Experimental Section.**

-----

### 3\. Generate High-Impact Qualitative Comparisons

We need a single, powerful figure that visually encapsulates our contribution.

  * **Action:** For a few compelling examples from both Photography and Microscopy, generate a multi-panel comparison figure showing:
    1.  **Noisy Observation (Input `y_e`)**
    2.  **Unguided Denoising (Prior only, `κ=0`)**
    3.  **L2 Baseline (Comparison Method)**
    4.  **Ours: DAPGD (at optimal `κ`)**
    5.  **Ground Truth (`x`)**
  * **Goal:** To create the centerpiece figure for our results section, offering an intuitive and immediate demonstration of our method's superiority.
  * **Paper Location:** **Main Paper, Results Section.**

-----

### 4\. Appendix: "A Note on Coordinate Systems in Physics-Informed Diffusion"

The detailed technical explanation of the solution belongs in the appendix. This provides a valuable, practical guide for other researchers in the field.

  * **Action:** Author a new appendix section dedicated to this topic.
  * **Goal:** Clearly articulate the **"Two-Space Problem"** (Normalized Prior Space vs. Physical Likelihood Space). Provide the explicit Python-like pseudo-code for our correct **"Un-normalize, Guide, Re-normalize"** sampling loop. Frame this as a critical implementation detail for achieving stable and physically-correct results.
  * **Paper Location:** **Appendix.**
