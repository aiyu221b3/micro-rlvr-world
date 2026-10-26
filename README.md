# micro-rlvr-world: mechanistic study of reinforcement learning with verifiable rewards for world models

A single-GPU mechanistic study of **RLVR-World**, the framework introduced by Wu et al. at *NeurIPS* (2025), in a scaled **32×32 continuous robotic manipulation** environment and a **144-state symbolic text-game** simulator. The original paper is titled [“RLVR-World: Training World Models with Reinforcement Learning”](https://arxiv.org/abs/2505.13934) ([project page](https://thuml.github.io/RLVR-World)).

The paper reports that its full-scale robotic manipulation experiments on RT-1 used an 87,212-trajectory dataset at 256×320 resolution, requiring 360 to 480 A100 GPU hours for visual tokenizer pre-training, 500 to 530 A100 GPU hours for autoregressive Transformer pre-training, and 4 to 8 A100 GPUs for RLVR post-training. Language world model experiments used 4 to 8 A100/H100 GPUs for 17 to 25 hours. This project does not attempt to match that compute scale; instead, it implements and audits the core algorithmic mechanisms in a deliberately reduced environment.

The visual environment uses a 32×32 RGB tabletop arena with a continuous 2D robotic end-effector (`red`), a movable object block (`blue`), and a target goal zone (`green`). The implementation retains the dual compressive Finite Scalar Quantization (FSQ) visual tokenizer with cross-attention context conditioning, unified sequence modeling with 256-bin action and proprioception quantization, decoder-only LLaMA world model pre-training (RoPE, SwiGLU, RMSNorm), and off-policy Group Relative Policy Optimization (GRPO) driven by decoded perceptual and state-transition metrics.

**The goal is mechanistic fidelity and diagnostic transparency, not scale replication.** The smaller environment makes the tokenization bottleneck, advantage normalization, and rollout dynamics easier to inspect, test, and ablate on a single GPU, while documenting both where the mechanisms hold and where low-resolution simplifications break down.

---

## Mechanistic Fidelity and Documented Simplifications

The implementation follows the paper's core formulation while making explicit reductions to the environment resolution, sequence horizon, and model size. The table distinguishes reproduced mechanisms from changes in scale.

| Component | Paper specification | Implementation | Fidelity and scope |
|---|---|---|---|
| **Compressive visual tokenization** | Dual VQGAN $(E_c, D_c)$ and $(E_v, D_v)$ with Finite Scalar Quantization (FSQ) and cross-attention to a preceding context frame $s_c$ | `DualCompressiveVQGAN` and `FSQ` (`L = [5, 5, 5]`, `K = 125`) in `src/networks.py` with 2D spatial positional embeddings and gated cross-attention | Core architecture reproduced at reduced spatial resolution (8×8 context, 4×4 dynamics) |
| **Unified multimodal vocabulary** | Offset vocabulary combining context visual tokens, dynamics tokens, and 256-bin quantized continuous vectors | `UnifiedWorldSequenceBuilder(\|V\|= 508)` in `src/networks.py` | Tokenization and non-overlapping offset scheme reproduced |
| **Autoregressive world model** | 12-layer LLaMA (138M parameters) with RoPE, RMSNorm, and SwiGLU trained with masked next-token cross-entropy on state tokens only | 4-layer `MicroLLaMA` in `src/networks.py` with causal target mask `get_multistep_target_mask` | Architecture and state-only loss masking reproduced at reduced parameter count |
| **Off-policy GRPO post-training** | Group sampling ${o_i}_{i=1}^G ~ p_{theta_old}$, group-normalized advantages $A_i$, inner mini-batch ratio clipping, and KL penalty to `p_ref` | `train_video_rlvr_grpo` in `src/rl.py` | Update equations and multi-mini-batch off-policy loop reproduced |
| **Verifiable decoded rewards** | `-sum [L1(s_hat, s) + LPIPS(s_hat, s)]` plus proprioceptive MSE and repetition penalty (video); Eq. (4) state-diff reward (text) | `compute_video_verifiable_rewards` and `compute_text_reward` in `src/rl.py` | Reward functions and non-differentiable decode-then-score pipeline reproduced |
| **Environment and horizon** | RT-1 / PushT (256×320 or 224×224, 7-step multi-step rollout) and ByteSized32 text games | 32×32 continuous 2D block-pushing (3-step rollout) and 144-state symbolic kitchen game | Deliberate scale and horizon reduction |

### Group Relative Policy Optimization (GRPO) objective

Given an input history sequence $q(s_{<=t}, a_{<=t})$, the behavior policy $p_{theta_old}$ samples a group of $G$ candidate future token trajectories ${o_i}_{i=1}^G$, decodes each into its native modality $s'_i$, and scores it against ground truth $s'$ via a verifiable metric `R_i = sign(D) * D(s'_i, s')`.

The group-relative advantage and clipped objective are:

$
A_i = (R_i - mean({R_1, ..., R_G})) / (std({R_1, ..., R_G}) + eps)$

$J_GRPO(theta) = (1 / G) * sum_{i=1}^G (1 / |o_i|) * sum_{t=1}^{|o_i|} [
    min(r_{i,t}(theta) * A_i, clip(r_{i,t}(theta), 1 - eps_clip, 1 + eps_clip) * A_i)
    - beta * D_KL(p_theta || p_ref)
]$


The corresponding PyTorch update over inner mini-batches is implemented in `src/rl.py`:

```python
ratio = torch.exp(new_logp - old_logp[mb])
s1 = ratio * adv[mb]
s2 = torch.clamp(ratio, 0.8, 1.2) * adv[mb]
log_r_ref = ref_logp[mb] - new_logp
kl = torch.exp(log_r_ref) - log_r_ref - 1.0
loss = -(torch.min(s1, s2) - 1e-3 * kl).mean()
```

---

## Experimental Questions

The study focuses on five mechanism-level questions:

1. Does RLVR post-training improve multi-step visual prediction metrics (LPIPS, MSE, PSNR, SSIM) over the MLE base model on held-out trajectories?
2. How does the GRPO group size `G in {2, 4, 8}` affect optimization stability and test-set LPIPS across seeds?
3. How does test-time Best-of-`N` sampling compare between the MLE base model and the RLVR-tuned model?
4. How do sparse binary rewards compare with structured task-specific rewards (Equation 4) on state-changing vs. unchanged transitions in symbolic language world modeling?
5. Where do low-resolution spatial quantization and single-metric reward optimization fail to replicate full-scale benchmarks (specifically in metric-specific specialization and Real2Sim policy ranking)?

---

## Results

The experimental suite evaluates three independent training seeds (`42`, `43`, and `44`) on held-out test sets (`n = 120` disjoint manipulation trajectories for video world modeling; `n = 400` transitions for symbolic language world modeling; `n = 50` episodes per policy checkpoint for Real2Sim evaluation).

![RLVR-World Scoped Mechanism Diagnostic Dashboard](results/figures/rlvr_world_diagnostic_dashboard.jpg)

### 1. Multi-step video world model performance (`results/exp5_table3_multiseed.csv`)

Across `n = 120` held-out multi-step trajectories and 3 seeds, post-training the MLE base model for 40 GRPO steps with combined foreground-L1, pretrained SqueezeNet LPIPS, and proprioceptive MSE rewards improves all perceptual and pixel reconstruction metrics:

| Model variant (`n = 120`, 3 seeds) | Repetition rate (%) ↓ | MSE (×100) ↓ | PSNR (dB) ↑ | SSIM (×100) ↑ | LPIPS (×100) ↓ |
|---|---:|---:|---:|---:|---:|
| **Base (MLE)** | 5.56 ± 0.00 | 1.773 ± 0.000 | 17.86 ± 0.00 | 80.50 ± 0.00 | 24.25 ± 0.00 |
| **Base (w/ repetition rejection)** | 0.00 ± 0.00 | 1.802 ± 0.004 | 17.77 ± 0.02 | 79.96 ± 0.08 | 25.20 ± 0.19 |
| **RLVR-World (Ours)** | 9.26 ± 1.33 | 1.727 ± 0.023 | 17.97 ± 0.05 | 80.92 ± 0.18 | 23.15 ± 0.27 |
| **RLVR-World (w/ repetition penalty)** | 5.00 ± 0.39 | 1.711 ± 0.011 | 18.03 ± 0.03 | 81.12 ± 0.05 | 23.00 ± 0.09 |

- **Perceptual and pixel gains:** `RLVR-World (Ours)` reduces test LPIPS (×100) from **24.25** to **23.15 ± 0.27** and MSE (×100) from **1.773** to **1.727 ± 0.023**. Adding the explicit non-differentiable repetition penalty (`RLVR-World w/ repetition penalty`) reduces frame repetition to **5.00 ± 0.39%** while achieving the best overall LPIPS (**23.00 ± 0.09**) and SSIM (**81.12 ± 0.05**).
- **Rejection sampling baseline:** Naively rejection-sampling non-repetitive rollouts from the MLE base model (`Base w/ repetition rejection`) eliminates repetition (0.0%) but degrades LPIPS to **25.20 ± 0.19**, confirming that filtering alone pushes generation into lower-quality tail samples.

### 2. Group size scaling and test-time Best-of-`N` scaling (`results/exp6a_group_size_scaling.csv`)

- **Group size scaling (Panel b):** All three group sizes start from the exact same Step-0 base checkpoint (`24.25` test LPIPS). At `G = 2`, two-sample advantage normalization is too noisy to improve test performance (`24.51 ± 0.49` at Step 32). Increasing the group size to `G = 4` (`23.34 ± 0.14`) and `G = 8` (`23.26 ± 0.14`) reliably reduces test LPIPS and tightens inter-seed variance. Because `G = 4` and `G = 8` overlap within one standard deviation at convergence, the supported conclusion in this environment is that `G >= 4` is required for stable variance reduction over `G = 2`.
- **Test-time Best-of-`N` scaling (Panel a):** Across `n = 120` test trajectories, RLVR-World outperforms the MLE base model at every sample budget `N in {1, 2, 4, 8}`, but the margin narrows from **0.93** LPIPS points at `N = 1` (`23.32` vs. `24.25`) to **0.27** points at `N = 8` (`18.31` vs. `18.58`). At `N = 1`, RLVR recovers roughly half the perceptual gain of doubling test-time compute to `N = 2` on the base model (`22.10`), consistent with mode-seeking policy sharpening.

### 3. Symbolic language world model (`results/exp1_language_world_model.csv`)

Evaluated across `n = 400` symbolic text-game transitions and 3 seeds:

| Model (`n = 400`, 3 seeds) | Unchanged accuracy (%) | Changed accuracy (%) | Overall accuracy (%) |
|---|---:|---:|---:|
| **Base (Untrained)** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **SFT (CoT fine-tuned)** | 95.99 ± 5.68 | 37.04 ± 8.36 | 77.42 ± 1.53 |
| **RLVR (Binary reward)** | 78.35 ± 7.77 | 88.89 ± 1.12 | 81.67 ± 4.98 |
| **RLVR (Task-specific Eq. 4)** | 86.13 ± 19.61 | 65.61 ± 17.58 | 79.67 ± 7.90 |

In this compact 144-state symbolic domain, supervised fine-tuning (SFT) learns the dominant unchanged ("no-op") structure first (**95.99 ± 5.68%** unchanged vs. **37.04 ± 8.36%** changed). Subsequent GRPO post-training with unchanged-case downsampling substantially lifts accuracy on state-changing transitions (**88.89 ± 1.12%** with binary reward; **65.61 ± 17.58%** with Equation 4 reward) at the cost of higher variance on unchanged transitions.

---

## Documented Negative Results and Mechanistic Diagnosis

A primary objective of this repository is to report where reduced-scale toy environments fail to replicate full-scale paper claims and to isolate the underlying causes.

### 1. Absence of a diagonal in metric-oriented optimization (Panel c, `results/exp6b_fig4c_raw_and_norm.csv`)

In the original paper's Figure 4c, post-training a video world model on a single metric `X in {MAE, MSE, PSNR, SSIM, LPIPS}` yields a diagonal heatmap where each model performs best on its own training metric. **That claim is not reproduced in this 32×32 environment.**

| Test metric (`n = 120`) | Base (MLE) | MAE-trained | MSE-trained | PSNR-trained | SSIM-trained | LPIPS-trained |
|---|---:|---:|---:|---:|---:|---:|
| **MAE (×100) ↓** | 4.340 | **4.118** | 4.211 | 4.131 | 4.268 | 4.297 |
| **MSE (×100) ↓** | 1.773 | **1.639** | 1.658 | 1.646 | 1.719 | 1.743 |
| **PSNR (dB) ↑** | 17.861 | **18.227** | 18.123 | 18.169 | 18.024 | 17.949 |
| **SSIM (×100) ↑** | 80.496 | **81.370** | 80.834 | 81.302 | 80.915 | 80.640 |
| **LPIPS (×100) ↓** | 24.246 | **22.712** | 23.769 | 22.891 | 23.613 | 23.683 |

- **What happens:** Every single-metric RLVR run improves over the `Base (MLE)` column across all five evaluation metrics. However, the `MAE`-trained model ranks first (`1.00` normalized) on every row—including LPIPS (`22.712`)—whereas the pure `LPIPS`-trained model ranks last among the five RLVR runs (`0.00`–`0.08` normalized, `23.683` LPIPS).
- **Mechanistic diagnosis:**
  1. **Low-resolution metric collinearity:** On a 32×32 canvas with flat-colored geometric sprites, MAE, MSE, PSNR, and SSIM are all governed by the exact 2D bounding-box placement of the effector and block.
  2. **Unanchored perceptual hacking vs. noise floor:** The raw spread across all five RLVR columns is only `0.18` on MAE (`4.118` to `4.297`), `0.10` on MSE (`1.639` to `1.743`), and `0.97` on LPIPS (`22.712` to `23.683`). Because those spreads are comparable to the inter-seed standard deviation observed in Panel (b) (`~0.14–0.50`), min-max row normalization magnifies small variations into stark `0.00` vs. `1.00` contrasts. Moreover, optimizing SqueezeNet LPIPS alone without an L1 pixel anchor allows discrete 4×4 tokens to exploit local high-frequency edge activations without tightening global object centroids.

### 2. Real2Sim policy evaluation floor effect (Panel d, `results/exp7_real2sim_evaluation.csv`)

In Panel (d), six closed-loop proportional-derivative (PD) policy checkpoints (`P1` through `P6`, `n = 50` episodes each) span real environment success rates from **0.38** to **1.00**. However, when rolled out inside both the Base and RLVR world models, evaluated success rates remain flat between **0.00** and **0.18** (`r = -0.26` for Base, `r = -0.50` for RLVR).

- **Mechanistic diagnosis:**
  1. **Spatial quantization vs. short-horizon boundary margin:** Over a 3-step rollout starting at `x_0 = 0.46`, successful real pushes barely cross the goal radius boundary (`x_final in [0.53, 0.56]` for goal `0.68 +- 0.15`). Because the compressive dynamics encoder `E_v` downsamples `32×32 -> 4×4` (`8×8` pixel macro-blocks `= 0.25` arena units), sub-grid displacements near the goal boundary snap to coarse grid centers.
  2. **Prior bias in rollout dynamics (Panel e):** While the VQGAN Oracle row in Panel (e) accurately reconstructs both the red effector and blue block from ground-truth tokens, both Base and RLVR autoregressive rollouts exhibit a bias toward a canonical downward-right push trajectory. When the evaluated policy executes subtle closed-loop corrections, the world model's prior under-shoots the goal threshold.

---

## Repository Structure

```text
micro-rlvr-world/
├── README.md
├── requirements.txt
├── run_reproduction.py
├── src/
│   ├── __init__.py
│   ├── env.py
│   ├── networks.py
│   ├── rl.py
│   ├── eval.py
│   └── mechanism_checks.py
└── results/
    ├── exp1_language_world_model.csv
    ├── exp5_table3_multiseed.csv
    ├── exp6a_group_size_scaling.csv
    ├── exp6b_fig4c_raw_and_norm.csv
    ├── exp7_real2sim_evaluation.csv
    └── figures/
        └── rlvr_world_diagnostic_dashboard.jpg
```

### Core modules

| File | Purpose |
|---|---|
| `src/env.py` | Continuous 2D robotic pushing simulator, closed-loop PD controller, and 144-state symbolic text game |
| `src/networks.py` | Dual compressive FSQ-VQGAN, unified sequence builder, LLaMA world model, and pretrained SqueezeNet LPIPS |
| `src/rl.py` | Multi-mini-batch off-policy GRPO engine, verifiable decoded rewards, and autoregressive rollout loop |
| `src/eval.py` | Multi-step evaluation suite, repetition-rejection baseline, paired bootstrap CIs, and Real2Sim evaluator |
| `src/mechanism_checks.py` | Fast implementation-level invariant checks (FSQ STE gradients, vocab bounds, GRPO scale invariance, KL >= 0) |
| `run_reproduction.py` | Main entry point for full training/evaluation and standalone CSV-to-figure plotting |

---

## Reproducing the Experiments

### Installation

Clone the repository and install dependencies:

```bash
git clone [https://github.com/aiyu221b3/micro-rlvr-world.git](https://github.com/aiyu221b3/micro-rlvr-world.git)
cd micro-rlvr-world
pip install -r requirements.txt
```

### 1. Run fast mechanistic invariant checks (~5 seconds)

Verify FSQ straight-through gradient flow, vocabulary offset bounds, causal target masking, GRPO advantage scale invariance, and non-negative Schulman `k3` KL divergence:

```bash
python -m src.mechanism_checks
```

### 2. Regenerate summary plots from saved CSVs (no GPU or retraining required)

To inspect the recorded multi-seed CSV tables in `results/` and regenerate summary figures directly from the saved data without re-running training:

```bash
python -c "from run_reproduction import plot_from_csv; plot_from_csv()"
```

### 3. Run the full training and evaluation suite from scratch (~8–10 minutes on a single T4 GPU)

To re-run all three seeds across Experiments 1 through 7 and overwrite the CSVs and figures under `results/`:

```bash
python run_reproduction.py --train
```

---

## Limitations

This repository is a reduced-scale mechanistic investigation, not a full-scale reproduction of RLVR-World.

The 32×32 image resolution, 4×4 compressive token grid, 3-step rollout horizon, 4-layer Transformer backbone, and synthetic 2D physics substantially alter the visual complexity and control dynamics compared to 256×320 real-robot RT-1 trajectories. In particular, low spatial resolution compresses the numerical spread across pixel and perceptual metrics (Panel c) and introduces macro-block quantization floors during short-horizon Real2Sim evaluation (Panel d).

The experiments use three independent seeds (`42`, `43`, and `44`). Observed standard deviations are reported descriptively to distinguish mechanism-level signal from batch and initialization noise.

---

## Reference

Wu, J., Yin, S., Feng, N., & Long, M. (2025). “RLVR-World: Training World Models with Reinforcement Learning.” *Advances in Neural Information Processing Systems (NeurIPS)*.

https://arxiv.org/abs/2505.13934

This repository is an independent implementation and mechanistic study of the published method. Please cite the original paper when referring to RLVR-World itself.
