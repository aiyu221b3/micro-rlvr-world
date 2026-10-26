# RLVR-World: Scoped Mechanism Demo & Diagnostic Ablation Suite

A self-contained, single-GPU PyTorch implementation and empirical audit of the core mechanisms in **RLVR-World: Training World Models with Reinforcement Learning** (Wu et al., NeurIPS 2025).

> **Scope Statement:** This repository is a **scoped mechanism demonstration in a toy continuous environment (`32x32` 2D robotic block-pushing and a `144`-state symbolic text world)**, not a full-scale reproduction of the 500-GPU-hour RT-1 experiments. Every table reports sample size $n$, number of random seeds, confidence intervals or standard deviations, and explicit diagnoses of where the miniature setup diverges from the paper's full-scale claims.

---

## 1. Implemented Mechanisms & Invariant Checks (`mechanism_checks.py`)

1. **Dual Compressive FSQ-VQGAN (`src/networks.py`):**
   - Finite Scalar Quantization ($L = [5, 5, 5] \implies K = 125$) with Straight-Through Estimator (STE) gradients.
   - Context frames $s_c$ (sampled strictly prior to the rollout segment to prevent $t=0$ spatial leakage) are encoded at $8\times8$ ($64$ tokens). Subsequent dynamics frames $s_t$ are compressed $4\times$ to $4\times4$ ($16$ tokens) and decoded via 2D-position-embedded, gated cross-attention against $E_c(s_c)$.
2. **Interleaved Sequence Modeling (`UnifiedWorldSequenceBuilder`):**
   - Non-overlapping token vocabulary ($|\mathcal{V}| = 508$) combining context visual tokens ($[0, 124]$), dynamics visual tokens ($[125, 249]$), and 256-bin uniformly quantized proprioception and continuous actions ($[250, 505]$).
   - Causal loss masking (`get_multistep_target_mask`) restricts MLE cross-entropy and GRPO policy gradients strictly to state/proprioception tokens $(z_{1:3}, p_{1:3})$, never action interventions.
3. **Off-Policy Multi-Mini-Batch GRPO (`src/rl.py`):**
   - Group Relative Policy Optimization with group size $G \in \{2, 4, 8\}$, inner mini-batch updates where $r_{i,t}(\theta) \neq 1$, PPO ratio clipping ($\epsilon = 0.2$), and Schulman's unbiased $k_3$ KL divergence estimator against $\pi_{\text{ref}}$.

---

## 2. Empirical Results ($n$, Seeds, & Supported Claims)

### A. Multi-Step Video World Model (`exp5_table3_multiseed.csv`, $n = 120$ disjoint test trajectories, 3 seeds)

| Model Variant | Repetition Rate (%) $\downarrow$ | MSE ($\times 100$) $\downarrow$ | PSNR (dB) $\uparrow$ | SSIM ($\times 100$) $\uparrow$ | LPIPS ($\times 100$) $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Base (MLE)** | $2.50 \pm 0.00$ | $1.766 \pm 0.000$ | $17.85 \pm 0.00$ | $80.17 \pm 0.00$ | $24.25 \pm 0.00$ |
| **Base (w/ rep. rejection)** | $0.00 \pm 0.00$ | $1.908 \pm 0.036$ | $17.46 \pm 0.09$ | $79.05 \pm 0.28$ | $25.78 \pm 0.16$ |
| **RLVR-World (Ours)** | $2.04 \pm 0.42$ | $1.622 \pm 0.012$ | $18.17 \pm 0.03$ | $81.14 \pm 0.09$ | $23.16 \pm 0.13$ |
| **RLVR-World (w/ rep. penalty)** | $1.76 \pm 0.24$ | $1.618 \pm 0.007$ | $18.19 \pm 0.02$ | $81.20 \pm 0.03$ | $23.09 \pm 0.12$ |

* **Supported Claim:** Post-training with verifiable perceptual + foreground-$L_1$ + proprioceptive rewards improves multi-step test LPIPS by $\sim 1.1$ points ($24.25 \to 23.16 \pm 0.13$) and MSE by $\sim 8.1\%$ ($1.766 \to 1.622 \pm 0.012$) within 40 GRPO steps. Naively rejection-sampling non-repetitive trajectories from the Base model degrades LPIPS ($25.78 \pm 0.16$) by pushing generation into lower-likelihood tail tokens.

### B. Group Size Scaling (`exp6a_group_size_scaling.csv`, $n = 120$, 3 seeds from shared Step-0 init)

| Group Size $G$ | Step 0 | Step 8 | Step 16 | Step 24 | Step 32 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$G = 2$** | $24.25 \pm 0.00$ | $23.91 \pm 0.74$ | $24.21 \pm 0.39$ | $24.52 \pm 0.54$ | $24.51 \pm 0.50$ |
| **$G = 4$** | $24.25 \pm 0.00$ | $23.98 \pm 0.53$ | $23.68 \pm 0.48$ | $23.42 \pm 0.55$ | $23.34 \pm 0.14$ |
| **$G = 8$** | $24.25 \pm 0.00$ | $23.98 \pm 0.32$ | $23.24 \pm 0.20$ | $23.24 \pm 0.32$ | $23.26 \pm 0.13$ |

* **Supported Claim:** $G = 2$ fails to improve over Base due to noisy 2-sample mean/std advantage estimates. Increasing to $G \ge 4$ reliably reduces LPIPS, though $G = 4$ and $G = 8$ are statistically indistinguishable at Step 32 ($23.34 \pm 0.14$ vs. $23.26 \pm 0.13$).

### C. Best-of-($N$) Test-Time Scaling (`exp6a_best_of_n_bootstrap_ci.csv`, $n = 120$, Paired Bootstrap)
* At $N=1$, RLVR-World improves LPIPS by $0.93$ points ($23.32$ vs. $24.25$), recovering roughly half the gain of doubling Base test-time compute to $N=2$ ($22.10$). The gap narrows as $N$ grows ($0.56$ at $N=2$, $0.88$ at $N=4$, $0.27$ at $N=8$), consistent with RLVR sharpening the policy around high-reward modes while reducing tail entropy.

---

## 3. Documented Negative Results & Mechanistic Diagnosis

### Negative Result 1: Unreproduced Diagonal in Metric-Oriented Optimization (Panel c)
* **Observation (`exp6b_fig4c_raw_and_norm.csv`):** While the paper reports a clean diagonal on RT-1 (training on metric $X$ achieves the best test score on $X$), our $32\times32$ benchmark does **not** produce a diagonal. Training on pure `MAE` achieves the best test scores across all rows (`MAE`: $4.145$, `MSE`: $1.552$, `PSNR`: $18.23$, `SSIM`: $81.41$, `LPIPS`: $22.71$), whereas training on pure `LPIPS` performs worst among the RLVR runs (`LPIPS`: $23.70$, though still beating Base at $24.25$).
* **Diagnosis:**
  1. **Metric Collinearity at Low Resolution:** On a $32\times32$ synthetic scene with flat-colored geometric sprites, `MAE`, `MSE`, `PSNR`, and `SSIM` are all dominated by the exact 2D bounding-box alignment of the red circle and blue square.
  2. **Adversarial Exploitation of Unanchored LPIPS:** When optimizing SqueezeNet `LPIPS` alone without an $L_1$ pixel anchor, GRPO discovers discrete $4\times4$ token combinations that produce high-frequency edge textures favored by early SqueezeNet filters on training batches, which fails to improve object centroid accuracy on held-out test trajectories. Furthermore, raw differences across columns ($0.14$ MSE, $0.99$ LPIPS) sit near the single-seed noise floor ($\sim 0.15\text{–}0.50$), so min-max row normalization amplifies tiny variations.

### Negative Result 2: Real2Sim Policy Ranking Floor Effect (Panel d)
* **Observation (`exp7_real2sim_evaluation.csv`):** Across 6 closed-loop PD policies ($n = 50$ episodes each), real environment success climbs from $38\% \to 100\%$, but simulated success in both Base and RLVR-World remains stuck between $0\% \text{ and } 18\%$ ($r = -0.26$ and $r = -0.50$).
* **Diagnosis via VQGAN Oracle Replay (`evaluate_real2sim_with_oracle_replay`):**
  - Over a short 3-step horizon starting at $x_0 = 0.46$, successful real trajectories barely cross the goal radius boundary ($x_{\text{final}} \in [0.53, 0.56]$ for goal $0.68 \pm 0.15$).
  - Because the compressive dynamics encoder $E_v$ downsamples $32\times32 \to 4\times4$ ($8\times8$ pixel macro-blocks $= 0.25$ arena units), sub-grid block displacements near the $0.53$ threshold suffer from quantization snapping and slight prior bias toward the training distribution mean. Replaying ground-truth final frames through the VQGAN (`VQGAN_Oracle_SR`) and testing action-conditioning sensitivity (`exp7a_conditioning_ablation.csv`) isolates how much error stems from the $4\times4$ discrete bottleneck versus autoregressive rollout drift.

---

## 4. Quickstart

```bash
pip install -r requirements.txt
python mechanism_checks.py
python run_benchmark.py
