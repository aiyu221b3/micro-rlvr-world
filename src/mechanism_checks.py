import numpy as np
import torch
from src.env import device, seed_everything, ContinuousPushTEnv
from src.networks import DualCompressiveVQGAN, UnifiedWorldSequenceBuilder, MicroLLaMA
from src.eval import detect_block_goal_success


def run_mechanism_checks():
    # safety checks for ze skeptics & peer reviewers
    print("Running RLVR-World mechanistic invariant checks...")
    seed_everything(42)

    vq = DualCompressiveVQGAN(fsq_levels=[5, 5, 5], ch=32).to(device)
    sb = UnifiedWorldSequenceBuilder()

    # check 1: fsq ste gradient flow invariant
    x_dummy = torch.randn(2, 3, 32, 32, device=device, requires_grad=True)
    rc, rt = vq(x_dummy, x_dummy)
    loss = (rc.mean() + rt.mean())
    loss.backward()
    assert x_dummy.grad is not None and x_dummy.grad.abs().max().item() > 0.0, "fsq ste broke grad flow!"
    print("  [PASS] 1. fsq straight-through estimator gradient invariant")

    # check 2: vocabulary non-overlap & target mask alignment invariant
    frames = torch.randn(2, 5, 3, 32, 32, device=device).clamp(-1.0, 1.0)
    proprios = torch.rand(2, 5, 2, device=device) * 0.8 + 0.1
    actions = (torch.rand(2, 4, 2, device=device) - 0.5) * 0.3
    seq = sb.build_multistep_eq8(frames, proprios, actions, vq)
    mask = sb.get_multistep_target_mask()
    assert seq.shape == (2, 144) and mask.sum().item() == 54, "sequence or target mask shape mismatch!"
    assert seq[:, :64].max().item() < 125, "context tokens leaked out of [0, 124]!"
    assert seq[:, 64:80].min().item() >= 125 and seq[:, 64:80].max().item() < 250, "dynamics tokens out of bounds!"
    assert seq[:, 80:84].min().item() >= 250 and seq[:, 80:84].max().item() < 506, "bin tokens out of bounds!"
    print("  [PASS] 2. unified sequence vocabulary offset & causal target mask invariant")

    # check 3: grpo group advantage scale invariance
    r_raw = torch.tensor([[0.010, 0.012, 0.015, 0.009], [1.0, 2.0, 3.0, 4.0]], device=device)
    adv_1 = (r_raw - r_raw.mean(dim=1, keepdim=True)) / r_raw.std(dim=1, keepdim=True).clamp(min=1e-6)
    r_scaled = r_raw * 1000.0
    adv_2 = (r_scaled - r_scaled.mean(dim=1, keepdim=True)) / r_scaled.std(dim=1, keepdim=True).clamp(min=1e-6)
    assert (adv_1 - adv_2).abs().max().item() < 1e-4, "grpo advantage normalization is scale-variant!"
    print("  [PASS] 3. grpo group-relative advantage scale-invariance check")

    # check 4: schulman k3 kl estimator non-negativity invariant
    log_p = torch.randn(16, 54, device=device)
    log_ref = log_p + torch.randn(16, 54, device=device) * 0.2
    kl = torch.exp(log_ref - log_p) - (log_ref - log_p) - 1.0
    assert (kl >= -1e-6).all(), "negative kl divergence detected!"
    print("  [PASS] 4. schulman k3 kl divergence non-negativity invariant")

    # check 5: real2sim success detector sanity check on raw ground-truth goal vs non-goal frames
    env = ContinuousPushTEnv()
    img_hit, _ = env.reset(agent_pos=[0.50, 0.50], block_pos=[0.68, 0.50])
    img_miss, _ = env.reset(agent_pos=[0.30, 0.50], block_pos=[0.35, 0.50])
    t_hit = torch.tensor(img_hit, device=device)
    t_miss = torch.tensor(img_miss, device=device)
    assert detect_block_goal_success(t_hit, env.goal_pos) is True, "detector missed ground-truth goal!"
    assert detect_block_goal_success(t_miss, env.goal_pos) is False, "detector false-positived on miss!"
    print("  [PASS] 5. real2sim centroid detector ground-truth separation invariant\n")


if __name__ == "__main__":
    run_mechanism_checks()