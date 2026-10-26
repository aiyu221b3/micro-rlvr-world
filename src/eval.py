from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.env import device, seed_everything, ContinuousPushTEnv
from src.networks import (
    MicroLLaMA,
    DualCompressiveVQGAN,
    UnifiedWorldSequenceBuilder,
    PretrainedLPIPS,
    compute_ssim,
)
from src.rl import rollout_multistep_world_model


@torch.no_grad()
def eval_text_model(model: MicroLLaMA, dataset) -> Dict[str, float]:
    model.eval()
    u_ok, u_tot, c_ok, c_tot = 0, 0, 0, 0
    prompts = torch.tensor([x[0] for x in dataset], dtype=torch.long, device=device)
    curr = prompts.clone()
    for _ in range(13):
        logits = model(curr)[:, -1, :]
        curr = torch.cat([curr, torch.argmax(logits, dim=-1, keepdim=True)], dim=1)
    preds = curr[:, 7:][:, [4, 6, 8, 10, 12]].cpu().tolist()
    for idx, (_, _, gt_v, ch) in enumerate(dataset):
        ok = int(preds[idx] == gt_v)
        if ch:
            c_ok += ok
            c_tot += 1
        else:
            u_ok += ok
            u_tot += 1
    return {
        "Unchanged": 100.0 * u_ok / max(1, u_tot),
        "Changed": 100.0 * c_ok / max(1, c_tot),
        "Overall": 100.0 * (u_ok + c_ok) / (u_tot + c_tot),
    }


@torch.no_grad()
def evaluate_video_model_table3(
    model: MicroLLaMA,
    loader: DataLoader,
    vqgan: DualCompressiveVQGAN,
    seq_builder: UnifiedWorldSequenceBuilder,
    lpips_fn: PretrainedLPIPS,
    rejection_sample_repetitions: bool = False,
    best_of_n: int = 1,
    return_per_sample: bool = False,
) -> Dict:
    model.eval()
    agg = {"rep": [], "mae": [], "mse": [], "psnr": [], "ssim": [], "lpips": []}

    for frames, proprios, actions in loader:
        frames, proprios, actions = frames.to(device), proprios.to(device), actions.to(device)
        B = frames.size(0)
        gt_seq = seq_builder.build_multistep_eq8(frames, proprios, actions, vqgan)
        feat_c, _ = vqgan.encode_context(frames[:, 0])
        prefix_84 = gt_seq[:, :84]
        z0_idx = gt_seq[:, 64:80] - seq_builder.off_dyn
        fut_acts = torch.stack([gt_seq[:, 102:104], gt_seq[:, 122:124], gt_seq[:, 142:144]], dim=1)

        best_lpips = torch.full((B,), float("inf"), device=device)
        best_stats = {k: torch.zeros(B, device=device) for k in agg}

        for _ in range(best_of_n):
            _, pred_vis, _ = rollout_multistep_world_model(
                model, prefix_84, fut_acts, seq_builder,
                temperature=0.85 if (best_of_n > 1 or rejection_sample_repetitions) else 1.0,
                greedy=(best_of_n == 1 and not rejection_sample_repetitions),
            )

            if rejection_sample_repetitions:
                for _ in range(8):
                    rep_check = torch.zeros(B, dtype=torch.bool, device=device)
                    prv = z0_idx
                    for t in range(3):
                        rep_check |= ((pred_vis[:, t, :] == prv).float().mean(dim=-1) >= 0.75)
                        prv = pred_vis[:, t, :]
                    if not rep_check.any():
                        break
                    _, new_vis, _ = rollout_multistep_world_model(
                        model, prefix_84[rep_check], fut_acts[rep_check], seq_builder, temperature=1.0, greedy=False
                    )
                    pred_vis[rep_check] = new_vis

            s_mae = torch.zeros(B, device=device)
            s_mse = torch.zeros(B, device=device)
            s_psnr = torch.zeros(B, device=device)
            s_ssim = torch.zeros(B, device=device)
            s_lpips = torch.zeros(B, device=device)
            s_rep = torch.zeros(B, device=device)

            prv = z0_idx
            for t in range(3):
                idx_t = pred_vis[:, t, :]
                s_rep += ((idx_t == prv).float().mean(dim=-1) >= 0.75).float() / 3.0
                prv = idx_t

                dec_rgb = vqgan.decode_dynamics_indices(idx_t, feat_c)
                gt_rgb = frames[:, t + 2]
                dec_01 = torch.clamp((dec_rgb + 1.0) * 0.5, 0.0, 1.0)
                gt_01 = torch.clamp((gt_rgb + 1.0) * 0.5, 0.0, 1.0)

                mse_t = (dec_01 - gt_01).pow(2).mean(dim=[1, 2, 3])
                s_mae += (dec_01 - gt_01).abs().mean(dim=[1, 2, 3]) / 3.0
                s_mse += mse_t / 3.0
                s_psnr += (10.0 * torch.log10(1.0 / (mse_t + 1e-6))) / 3.0
                s_ssim += compute_ssim(dec_rgb, gt_rgb) / 3.0
                s_lpips += lpips_fn(dec_rgb, gt_rgb) / 3.0

            imp = s_lpips < best_lpips
            best_lpips = torch.where(imp, s_lpips, best_lpips)
            for k, t_val in zip(
                ["rep", "mae", "mse", "psnr", "ssim", "lpips"],
                [s_rep, s_mae, s_mse, s_psnr, s_ssim, s_lpips],
            ):
                best_stats[k] = torch.where(imp, t_val, best_stats[k])

        for k in agg:
            scale = 100.0 if k in ["rep", "mae", "mse", "ssim", "lpips"] else 1.0
            agg[k].extend((best_stats[k] * scale).cpu().tolist())

    if return_per_sample:
        return {k: np.array(v, dtype=np.float64) for k, v in agg.items()}
    return {k: float(np.mean(v)) for k, v in agg.items()}


def paired_bootstrap_ci(
    base_samples: np.ndarray, rlvr_samples: np.ndarray, num_boot: int = 1000, seed: int = 42
) -> Dict[str, float]:
    # paired bootstrap across the n=120 test trajectories for 95% cis
    rng = np.random.default_rng(seed)
    n = len(base_samples)
    idx = rng.integers(0, n, size=(num_boot, n))
    b_means = base_samples[idx].mean(axis=1)
    r_means = rlvr_samples[idx].mean(axis=1)
    diff_means = b_means - r_means

    return {
        "base_mean": float(base_samples.mean()),
        "base_ci_low": float(np.percentile(b_means, 2.5)),
        "base_ci_high": float(np.percentile(b_means, 97.5)),
        "rlvr_mean": float(rlvr_samples.mean()),
        "rlvr_ci_low": float(np.percentile(r_means, 2.5)),
        "rlvr_ci_high": float(np.percentile(r_means, 97.5)),
        "diff_mean": float(diff_means.mean()),
        "diff_ci_low": float(np.percentile(diff_means, 2.5)),
        "diff_ci_high": float(np.percentile(diff_means, 97.5)),
    }


@torch.no_grad()
def run_conditioning_ablation(
    model: MicroLLaMA,
    loader: DataLoader,
    vqgan: DualCompressiveVQGAN,
    seq_builder: UnifiedWorldSequenceBuilder,
    lpips_fn: PretrainedLPIPS,
) -> Dict[str, Dict[str, float]]:
    # doth the model heed our actions, or merely hallucinate a generic push?
    model.eval()
    modes = ["normal", "shuffled_actions", "zeroed_conditioning"]
    results = {m: {"lpips": [], "mse": [], "token_change_rate": []} for m in modes}

    for frames, proprios, actions in loader:
        frames, proprios, actions = frames.to(device), proprios.to(device), actions.to(device)
        B = frames.size(0)
        gt_seq = seq_builder.build_multistep_eq8(frames, proprios, actions, vqgan)
        feat_c, _ = vqgan.encode_context(frames[:, 0])

        prefix_normal = gt_seq[:, :84].clone()
        fut_normal = torch.stack([gt_seq[:, 102:104], gt_seq[:, 122:124], gt_seq[:, 142:144]], dim=1)
        _, vis_norm, _ = rollout_multistep_world_model(model, prefix_normal, fut_normal, seq_builder, greedy=True)

        for mode in modes:
            pref = prefix_normal.clone()
            fut = fut_normal.clone()
            if mode == "shuffled_actions":
                perm = torch.randperm(B, device=device)
                pref[:, 80:84] = pref[perm, 80:84]
                fut = fut[perm]
            elif mode == "zeroed_conditioning":
                zero_act = seq_builder.quantize_vec(torch.zeros((B, 2), device=device), -0.15, 0.15)
                mid_prop = seq_builder.quantize_vec(torch.full((B, 2), 0.50, device=device), 0.10, 0.90)
                pref[:, 80:82] = mid_prop
                pref[:, 82:84] = zero_act
                fut = zero_act.unsqueeze(1).repeat(1, 3, 1)

            if mode == "normal":
                pred_vis = vis_norm
            else:
                _, pred_vis, _ = rollout_multistep_world_model(model, pref, fut, seq_builder, greedy=True)

            tok_diff = (pred_vis != vis_norm).float().mean(dim=[1, 2]) * 100.0
            s_lpips = torch.zeros(B, device=device)
            s_mse = torch.zeros(B, device=device)
            for t in range(3):
                dec_rgb = vqgan.decode_dynamics_indices(pred_vis[:, t, :], feat_c)
                gt_rgb = frames[:, t + 2]
                dec_01 = torch.clamp((dec_rgb + 1.0) * 0.5, 0.0, 1.0)
                gt_01 = torch.clamp((gt_rgb + 1.0) * 0.5, 0.0, 1.0)
                s_mse += (dec_01 - gt_01).pow(2).mean(dim=[1, 2, 3]) / 3.0
                s_lpips += lpips_fn(dec_rgb, gt_rgb) / 3.0

            results[mode]["lpips"].extend((s_lpips * 100.0).cpu().tolist())
            results[mode]["mse"].extend((s_mse * 100.0).cpu().tolist())
            results[mode]["token_change_rate"].extend(tok_diff.cpu().tolist())

    return {
        m: {k: float(np.mean(v)) for k, v in results[m].items()}
        for m in modes
    }


def detect_block_goal_success(rgb_tensor: torch.Tensor, goal_pos: np.ndarray, radius: float = 0.15) -> bool:
    rgb_01 = np.clip(((rgb_tensor.cpu().numpy() + 1.0) * 0.5).transpose(1, 2, 0), 0.0, 1.0)
    blue_mask = (rgb_01[:, :, 2] - rgb_01[:, :, 0]) > 0.25
    if blue_mask.sum() < 3:
        return False
    ys, xs = np.where(blue_mask)
    sim_pos = np.array([(xs.mean() + 0.5) / 32.0, (ys.mean() + 0.5) / 32.0])
    return bool(np.linalg.norm(sim_pos - goal_pos) < radius)


@torch.no_grad()
def evaluate_real2sim_with_oracle_replay(
    base_wm: MicroLLaMA,
    rlvr_wm: MicroLLaMA,
    vqgan: DualCompressiveVQGAN,
    seq_builder: UnifiedWorldSequenceBuilder,
    num_trials: int = 50,
) -> Tuple[List[float], List[float], List[float], List[float], List[List]]:
    # oracle replay diagnostic: passeth real final frame through vqgan + detector to isolate quantization floor
    checkpoints = [
        ("Policy-Begin", 1.00),
        ("Policy-15%", 0.75),
        ("Policy-35%", 0.55),
        ("Policy-60%", 0.35),
        ("Policy-85%", 0.18),
        ("Policy-Conv", 0.04),
    ]
    real_srs, oracle_srs, base_srs, rlvr_srs = [], [], [], []
    r2s_rows = []

    for ckpt_name, noise in checkpoints:
        seed_everything(200)
        r_wins, o_wins, b_wins, rl_wins = 0, 0, 0, 0
        for _ in range(num_trials):
            env = ContinuousPushTEnv()
            y_jit = np.random.uniform(-0.06, 0.06)
            s_c_img, _ = env.reset(agent_pos=[0.30, 0.50 + y_jit], block_pos=[0.46, 0.50 + y_jit * 0.5])
            s_0_img, p_0, _ = env.step(np.array([0.04, 0.0], dtype=np.float32))

            acts = []
            for _ in range(3):
                pd_act = env.expert_pd_action()
                rand_act = np.random.uniform(-0.15, 0.15, size=2).astype(np.float32)
                act = np.clip((1.0 - noise) * pd_act + noise * rand_act, -0.15, 0.15)
                acts.append(act)
                final_real_img, _, info = env.step(act)

            if np.linalg.norm(info["block"] - env.goal_pos) < 0.15:
                r_wins += 1

            sc_t = torch.tensor(s_c_img, dtype=torch.float32, device=device).unsqueeze(0)
            s0_t = torch.tensor(s_0_img, dtype=torch.float32, device=device).unsqueeze(0)
            s_final_t = torch.tensor(final_real_img, dtype=torch.float32, device=device).unsqueeze(0)
            p0_t = torch.tensor(p_0, dtype=torch.float32, device=device).unsqueeze(0)
            acts_t = torch.tensor(np.stack(acts), dtype=torch.float32, device=device).unsqueeze(0)

            feat_c, idx_c = vqgan.encode_context(sc_t)
            _, idx_0 = vqgan.encode_dynamics(s0_t)
            _, idx_oracle_final = vqgan.encode_dynamics(s_final_t)
            oracle_rgb = vqgan.decode_dynamics_indices(idx_oracle_final, feat_c)[0]
            if detect_block_goal_success(oracle_rgb, env.goal_pos):
                o_wins += 1

            p0_tok = seq_builder.quantize_vec(p0_t, 0.10, 0.90)
            b_toks = seq_builder.quantize_vec(acts_t, -0.15, 0.15)
            prefix_84 = torch.cat([idx_c, idx_0 + seq_builder.off_dyn, p0_tok, b_toks[:, 0, :]], dim=1)
            fut_b = torch.cat([b_toks[:, 1:, :], b_toks[:, -1:, :]], dim=1)

            for wm, tag in [(base_wm, "base"), (rlvr_wm, "rlvr")]:
                _, pred_vis, _ = rollout_multistep_world_model(wm, prefix_84, fut_b, seq_builder, greedy=True)
                sim_rgb = vqgan.decode_dynamics_indices(pred_vis[:, -1, :], feat_c)[0]
                if detect_block_goal_success(sim_rgb, env.goal_pos):
                    if tag == "base":
                        b_wins += 1
                    else:
                        rl_wins += 1

        r_sr = r_wins / num_trials
        o_sr = o_wins / num_trials
        b_sr = b_wins / num_trials
        rl_sr = rl_wins / num_trials
        real_srs.append(r_sr)
        oracle_srs.append(o_sr)
        base_srs.append(b_sr)
        rlvr_srs.append(rl_sr)
        r2s_rows.append([ckpt_name, noise, num_trials, r_sr, o_sr, b_sr, rl_sr])

    return real_srs, oracle_srs, base_srs, rlvr_srs, r2s_rows


@torch.no_grad()
def evaluate_cem_mpc(
    model: MicroLLaMA,
    vqgan: DualCompressiveVQGAN,
    seq_builder: UnifiedWorldSequenceBuilder,
    lpips_fn: PretrainedLPIPS,
    episodes: int = 60,
    seed: int = 42,
) -> float:
    seed_everything(seed)
    env = ContinuousPushTEnv()
    wins = 0
    for _ in range(episodes):
        y_off = np.random.uniform(-0.08, 0.08)
        s_c_img, _ = env.reset(agent_pos=[0.26, 0.50 + y_off], block_pos=[0.44, 0.50 + y_off * 0.5])
        s_0_img, p_0, _ = env.step(np.array([0.04, 0.0], dtype=np.float32))

        goal_env = ContinuousPushTEnv()
        goal_img, _ = goal_env.reset(agent_pos=[0.56, 0.50], block_pos=[0.68, 0.50])
        goal_t = torch.tensor(goal_img, dtype=torch.float32, device=device).unsqueeze(0)

        sc_t = torch.tensor(s_c_img, dtype=torch.float32, device=device).unsqueeze(0)
        s0_t = torch.tensor(s_0_img, dtype=torch.float32, device=device).unsqueeze(0)
        p0_t = torch.tensor(p_0, dtype=torch.float32, device=device).unsqueeze(0)

        feat_c, idx_c = vqgan.encode_context(sc_t)
        _, idx_0 = vqgan.encode_dynamics(s0_t)
        p0_tok = seq_builder.quantize_vec(p0_t, 0.10, 0.90)

        N_pop = 32
        mu = np.zeros((3, 2), dtype=np.float32)
        sigma = np.ones((3, 2), dtype=np.float32) * 0.11

        for _ in range(2):
            acts = np.clip(mu[None] + np.random.randn(N_pop, 3, 2) * sigma[None], -0.15, 0.15).astype(np.float32)
            acts_t = torch.tensor(acts, dtype=torch.float32, device=device)
            b_toks = seq_builder.quantize_vec(acts_t, -0.15, 0.15)

            prefix_84 = torch.cat(
                [
                    idx_c.repeat(N_pop, 1),
                    (idx_0 + seq_builder.off_dyn).repeat(N_pop, 1),
                    p0_tok.repeat(N_pop, 1),
                    b_toks[:, 0, :],
                ],
                dim=1,
            )
            fut_b = torch.cat([b_toks[:, 1:, :], b_toks[:, -1:, :]], dim=1)

            _, pred_vis, _ = rollout_multistep_world_model(model, prefix_84, fut_b, seq_builder, greedy=True)
            final_rgb = vqgan.decode_dynamics_indices(pred_vis[:, -1, :], feat_c.repeat(N_pop, 1, 1, 1))
            cost = lpips_fn(final_rgb, goal_t.repeat(N_pop, 1, 1, 1)) + (final_rgb - goal_t).abs().mean(dim=[1, 2, 3])
            elites = torch.topk(-cost, k=6).indices.cpu().numpy()
            mu = acts[elites].mean(axis=0)
            sigma = acts[elites].std(axis=0) + 0.01

        for a_step in mu:
            _, _, info = env.step(a_step)
        if np.linalg.norm(info["block"] - env.goal_pos) < 0.15:
            wins += 1
    return 100.0 * wins / float(episodes)