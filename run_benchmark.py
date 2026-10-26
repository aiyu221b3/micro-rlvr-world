import os
import csv
import copy
import random
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.env import device, seed_everything, SymbolicTextGameEnv, DisjointManipulationDataset
from src.networks import PretrainedLPIPS, DualCompressiveVQGAN, UnifiedWorldSequenceBuilder, MicroLLaMA
from src.rl import (
    weighted_foreground_l1,
    compute_text_reward,
    rollout_multistep_world_model,
    train_video_rlvr_grpo,
)
from src.eval import (
    eval_text_model,
    evaluate_video_model_table3,
    paired_bootstrap_ci,
    run_conditioning_ablation,
    evaluate_real2sim_with_oracle_replay,
    evaluate_cem_mpc,
)
from mechanism_checks import run_mechanism_checks

OUTPUT_DIR = "rlvr_world_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_csv(filename: str, headers: List[str], rows: List[List]):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"  -> saved csv: {path}")


def main():
    run_mechanism_checks()
    lpips_fn = PretrainedLPIPS().to(device)

    # 1. language world model (3 seeds, n=400)
    text_env = SymbolicTextGameEnv()
    full_catalog = text_env.build_full_catalog()
    text_test_set = [full_catalog[i % len(full_catalog)] for i in range(400)]

    print("==============================================================================")
    print("EXP 1: Language World Model (3 Seeds, n=400 Test Transitions)")
    print("==============================================================================")
    lang_results = {k: {"Unchanged": [], "Changed": [], "Overall": []} for k in ["Base", "SFT", "RLVR_Binary", "RLVR_TaskEq4"]}

    for seed in [42, 43, 44]:
        seed_everything(seed)
        m_base = MicroLLaMA(vocab_size=text_env.vocab_size, dim=96, n_layers=3, n_heads=4, max_len=32).to(device)
        r_b = eval_text_model(m_base, text_test_set)
        for k in r_b:
            lang_results["Base"][k].append(r_b[k])

        m_sft = copy.deepcopy(m_base)
        opt_sft = torch.optim.AdamW(m_sft.parameters(), lr=1.5e-3, weight_decay=0.01)
        sft_pool = random.sample(full_catalog, 55)
        for _ in range(12):
            m_sft.train()
            random.shuffle(sft_pool)
            for i in range(0, len(sft_pool), 16):
                b = sft_pool[i : i + 16]
                seq = torch.tensor([p + r for p, r, _, _ in b], dtype=torch.long, device=device)
                logits = m_sft(seq[:, :-1])[:, 6:, :]
                loss = F.cross_entropy(logits.reshape(-1, text_env.vocab_size), seq[:, 7:].reshape(-1))
                opt_sft.zero_grad()
                loss.backward()
                opt_sft.step()
        r_s = eval_text_model(m_sft, text_test_set)
        for k in r_s:
            lang_results["SFT"][k].append(r_s[k])

        ch_pool = [x for x in full_catalog if x[3]] * 3
        unch_pool = [x for x in full_catalog if not x[3]]
        rl_pool = ch_pool + unch_pool[:40]

        for mode_key, r_mode in [("RLVR_Binary", "binary"), ("RLVR_TaskEq4", "task_specific")]:
            m_rl = copy.deepcopy(m_sft)
            m_ref = copy.deepcopy(m_sft).eval()
            opt_rl = torch.optim.AdamW(m_rl.parameters(), lr=5e-4, weight_decay=0.01)
            G_l = 8
            for _ in range(55):
                batch = random.sample(rl_pool, 16)
                p_t = torch.tensor([x[0] for x in batch], dtype=torch.long, device=device)
                gt_t = torch.tensor([x[2] for x in batch], dtype=torch.long, device=device)
                B = p_t.size(0)
                with torch.no_grad():
                    m_rl.eval()
                    curr = p_t.repeat_interleave(G_l, dim=0)
                    gt_G = gt_t.repeat_interleave(G_l, dim=0)
                    for _ in range(13):
                        probs = F.softmax(m_rl(curr)[:, -1, :], dim=-1)
                        curr = torch.cat([curr, torch.multinomial(probs, 1)], dim=1)
                    resp = curr[:, 7:]
                    old_lp = torch.gather(F.log_softmax(m_rl(curr[:, :-1])[:, 6:, :], -1), -1, resp.unsqueeze(-1)).squeeze(-1)
                    ref_lp = torch.gather(F.log_softmax(m_ref(curr[:, :-1])[:, 6:, :], -1), -1, resp.unsqueeze(-1)).squeeze(-1)
                    rew = compute_text_reward(resp, gt_G, text_env, r_mode).view(B, G_l)
                    adv = ((rew - rew.mean(1, keepdim=True)) / (rew.std(1, keepdim=True).clamp(min=1e-4))).view(B * G_l, 1)

                m_rl.train()
                for mb in torch.randperm(B * G_l, device=device).chunk(2):
                    new_lp = torch.gather(F.log_softmax(m_rl(curr[mb, :-1])[:, 6:, :], -1), -1, resp[mb].unsqueeze(-1)).squeeze(-1)
                    ratio = torch.exp(new_lp - old_lp[mb])
                    s1, s2 = ratio * adv[mb], torch.clamp(ratio, 0.8, 1.2) * adv[mb]
                    kl = torch.exp(ref_lp[mb] - new_lp) - (ref_lp[mb] - new_lp) - 1.0
                    loss = -(torch.min(s1, s2) - 0.001 * kl).mean()
                    opt_rl.zero_grad()
                    loss.backward()
                    opt_rl.step()

            r_out = eval_text_model(m_rl, text_test_set)
            for k in r_out:
                lang_results[mode_key][k].append(r_out[k])

    exp1_rows = []
    for m_name in ["Base", "SFT", "RLVR_Binary", "RLVR_TaskEq4"]:
        u_m, u_s = np.mean(lang_results[m_name]["Unchanged"]), np.std(lang_results[m_name]["Unchanged"])
        c_m, c_s = np.mean(lang_results[m_name]["Changed"]), np.std(lang_results[m_name]["Changed"])
        o_m, o_s = np.mean(lang_results[m_name]["Overall"]), np.std(lang_results[m_name]["Overall"])
        print(f"  {m_name:<18} | Unch: {u_m:6.2f}±{u_s:5.2f}% | Chg: {c_m:6.2f}±{c_s:5.2f}% | All: {o_m:6.2f}±{o_s:5.2f}%")
        exp1_rows.append([m_name, f"{u_m:.2f}±{u_s:.2f}", f"{c_m:.2f}±{c_s:.2f}", f"{o_m:.2f}±{o_s:.2f}", 400, 3])
    save_csv("exp1_language_world_model.csv", ["Model", "Unchanged_Acc", "Changed_Acc", "Overall_Acc", "N_Test", "Seeds"], exp1_rows)

    # 2. dataset & dual compressive fsq-vqgan
    seed_everything(42)
    train_ds = DisjointManipulationDataset(num_trajs=900, split="train")
    test_ds = DisjointManipulationDataset(num_trajs=120, split="test")
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    print("\n==============================================================================")
    print("EXP 2: Dual Compressive FSQ-VQGAN & Oracle Recall Gate")
    print("==============================================================================")
    vqgan = DualCompressiveVQGAN(fsq_levels=[5, 5, 5], ch=64).to(device)
    opt_vq = torch.optim.AdamW(vqgan.parameters(), lr=2.5e-3, weight_decay=1e-4)

    for epoch in range(16):
        vqgan.train()
        for frames, _, _ in train_loader:
            frames = frames.to(device)
            s_c, s_t = frames[:, 0], frames[:, random.randint(1, 4)]
            rc, rt = vqgan(s_c, s_t)
            loss = weighted_foreground_l1(rc, s_c) + weighted_foreground_l1(rt, s_t) + 0.5 * (lpips_fn(rc, s_c).mean() + lpips_fn(rt, s_t).mean())
            opt_vq.zero_grad()
            loss.backward()
            opt_vq.step()

    vqgan.eval()
    for p in vqgan.parameters():
        p.requires_grad = False

    seq_builder = UnifiedWorldSequenceBuilder()
    ms_target_mask = seq_builder.get_multistep_target_mask()

    # 3. base mle pre-training
    print("\n==============================================================================")
    print("EXP 3: Base Video World Model MLE Pre-Training")
    print("==============================================================================")
    video_base_model = MicroLLaMA(vocab_size=seq_builder.vocab_size, dim=192, n_layers=4, n_heads=6, max_len=256).to(device)
    mle_opt = torch.optim.AdamW(video_base_model.parameters(), lr=1.2e-3, weight_decay=0.02)

    for epoch in range(12):
        video_base_model.train()
        for frames, proprios, actions in train_loader:
            frames, proprios, actions = frames.to(device), proprios.to(device), actions.to(device)
            seq = seq_builder.build_multistep_eq8(frames, proprios, actions, vqgan)
            logits = video_base_model(seq[:, :-1])
            loss = F.cross_entropy(
                logits[:, ms_target_mask, :].reshape(-1, seq_builder.vocab_size),
                seq[:, 1:][:, ms_target_mask].reshape(-1),
            )
            mle_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(video_base_model.parameters(), 1.0)
            mle_opt.step()

    # 4 & 5. 3-seed table 3 sweep
    print("\n==============================================================================")
    print("EXP 4 & 5: 3-Seed Video World Model GRPO Sweep (n=120 Test Trajectories)")
    print("==============================================================================")
    table3_seeds = {
        "Base (MLE)": [],
        "Base (w/ rep. rejection)": [],
        "RLVR-World (Ours)": [],
        "RLVR-World (w/ rep. penalty)": [],
    }
    base_eval_det = evaluate_video_model_table3(video_base_model, test_loader, vqgan, seq_builder, lpips_fn)
    trained_rlvr_models = []

    for seed in [42, 43, 44]:
        seed_everything(seed)
        table3_seeds["Base (MLE)"].append(base_eval_det)
        table3_seeds["Base (w/ rep. rejection)"].append(
            evaluate_video_model_table3(video_base_model, test_loader, vqgan, seq_builder, lpips_fn, rejection_sample_repetitions=True)
        )
        m_rlvr, _ = train_video_rlvr_grpo(
            video_base_model, train_loader, vqgan, seq_builder, lpips_fn, group_size=8, steps=40, metric_mode="l1_lpips_prop"
        )
        trained_rlvr_models.append(m_rlvr)
        table3_seeds["RLVR-World (Ours)"].append(
            evaluate_video_model_table3(m_rlvr, test_loader, vqgan, seq_builder, lpips_fn)
        )
        m_rep, _ = train_video_rlvr_grpo(
            video_base_model, train_loader, vqgan, seq_builder, lpips_fn, group_size=8, steps=40, metric_mode="l1_lpips_prop", rep_penalty_weight=0.35
        )
        table3_seeds["RLVR-World (w/ rep. penalty)"].append(
            evaluate_video_model_table3(m_rep, test_loader, vqgan, seq_builder, lpips_fn)
        )

    rlvr_video_model = trained_rlvr_models[0]
    t3_csv_rows = []
    for name, runs in table3_seeds.items():
        csv_r = [name, len(test_ds), 3]
        for k in ["rep", "mse", "psnr", "ssim", "lpips"]:
            vals = [r[k] for r in runs]
            csv_r.extend([round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)])
        t3_csv_rows.append(csv_r)
    save_csv(
        "exp5_table3_multiseed.csv",
        ["Model", "N_Test", "Seeds", "Rep_Mean", "Rep_Std", "MSE_Mean", "MSE_Std", "PSNR_Mean", "PSNR_Std", "SSIM_Mean", "SSIM_Std", "LPIPS_Mean", "LPIPS_Std"],
        t3_csv_rows,
    )

    # 6a.paired bootstrap cis for best-of-n scaling (panel a)
    print("\n==============================================================================")
    print("EXP 6A: Best-of-N Test-Time Scaling with 1,000-Sample Paired Bootstrap 95% CIs")
    print("==============================================================================")
    N_vals = [1, 2, 4, 8]
    bon_stats = []
    bon_csv = []
    for n_samp in N_vals:
        seed_everything(100 + n_samp)
        b_arr = evaluate_video_model_table3(
            video_base_model, test_loader, vqgan, seq_builder, lpips_fn, best_of_n=n_samp, return_per_sample=True
        )["lpips"]
        seed_everything(100 + n_samp)
        r_arr = evaluate_video_model_table3(
            rlvr_video_model, test_loader, vqgan, seq_builder, lpips_fn, best_of_n=n_samp, return_per_sample=True
        )["lpips"]
        ci = paired_bootstrap_ci(b_arr, r_arr, num_boot=1000, seed=42)
        bon_stats.append(ci)
        bon_csv.append([
            n_samp, len(test_ds),
            round(ci["base_mean"], 4), round(ci["base_ci_low"], 4), round(ci["base_ci_high"], 4),
            round(ci["rlvr_mean"], 4), round(ci["rlvr_ci_low"], 4), round(ci["rlvr_ci_high"], 4),
            round(ci["diff_mean"], 4), round(ci["diff_ci_low"], 4), round(ci["diff_ci_high"], 4),
        ])
        print(
            f"  N={n_samp} | Base: {ci['base_mean']:.2f} [{ci['base_ci_low']:.2f}, {ci['base_ci_high']:.2f}] | "
            f"RLVR: {ci['rlvr_mean']:.2f} [{ci['rlvr_ci_low']:.2f}, {ci['rlvr_ci_high']:.2f}] | "
            f"Paired Delta: +{ci['diff_mean']:.2f} [{ci['diff_ci_low']:.2f}, {ci['diff_ci_high']:.2f}]"
        )
    save_csv(
        "exp6a_best_of_n_bootstrap_ci.csv",
        ["N_Samples", "N_Test", "Base_Mean", "Base_CI_Low", "Base_CI_High", "RLVR_Mean", "RLVR_CI_Low", "RLVR_CI_High", "Paired_Diff_Mean", "Diff_CI_Low", "Diff_CI_High"],
        bon_csv,
    )

    # 6b. 3-seed group size scaling (panel b)
    val_fast_loader = DataLoader(test_ds, batch_size=60, shuffle=False)
    eval_steps_axis = [0, 8, 16, 24, 32]
    group_val_curves = {2: [], 4: [], 8: []}
    for seed in [42, 43, 44]:
        for g_val in [2, 4, 8]:
            seed_everything(seed)
            _, h_g = train_video_rlvr_grpo(
                video_base_model, train_loader, vqgan, seq_builder, lpips_fn,
                group_size=g_val, steps=32, eval_every=8, val_subset_loader=val_fast_loader, eval_fn=evaluate_video_model_table3
            )
            group_val_curves[g_val].append(h_g["val_lpips"])

    # 6c. 3-seed metric-oriented optimization matrix with base column (panel c)
    print("\n==============================================================================")
    print("EXP 6C: 3-Seed Metric-Oriented Optimization Matrix (Base + 5 Reward Metrics)")
    print("==============================================================================")
    metrics_list = ["MAE", "MSE", "PSNR", "SSIM", "LPIPS"]
    col_names = ["Base"] + metrics_list
    raw_mean_mat = np.zeros((5, 6))
    raw_std_mat = np.zeros((5, 6))
    score_mat = np.zeros((5, 6))

    raw_mean_mat[:, 0] = [base_eval_det["mae"], base_eval_det["mse"], base_eval_det["psnr"], base_eval_det["ssim"], base_eval_det["lpips"]]
    score_mat[:, 0] = [-base_eval_det["mae"], -base_eval_det["mse"], base_eval_det["psnr"], base_eval_det["ssim"], -base_eval_det["lpips"]]

    for c_idx, m_name in enumerate(metrics_list, start=1):
        seed_runs = []
        for s_i in [42, 43, 44]:
            seed_everything(s_i + c_idx * 10)
            m_spec, _ = train_video_rlvr_grpo(
                video_base_model, train_loader, vqgan, seq_builder, lpips_fn, group_size=8, steps=35, metric_mode=m_name, lr=7e-5
            )
            seed_runs.append(evaluate_video_model_table3(m_spec, test_loader, vqgan, seq_builder, lpips_fn))

        for r_i, k_m in enumerate(["mae", "mse", "psnr", "ssim", "lpips"]):
            vals = [x[k_m] for x in seed_runs]
            raw_mean_mat[r_i, c_idx] = np.mean(vals)
            raw_std_mat[r_i, c_idx] = np.std(vals)
            score_mat[r_i, c_idx] = -np.mean(vals) if k_m in ["mae", "mse", "lpips"] else np.mean(vals)

    norm_5x6 = np.zeros((5, 6))
    for r in range(5):
        row_s = score_mat[r]
        norm_5x6[r] = (row_s - row_s.min()) / (row_s.max() - row_s.min() + 1e-8)

    fig4c_csv = []
    row_labels = ["MAE(x100)↓", "MSE(x100)↓", "PSNR(dB)↑", "SSIM(x100)↑", "LPIPS(x100)↓"]
    for r_i, r_name in enumerate(row_labels):
        fig4c_csv.append(
            [r_name]
            + [f"{raw_mean_mat[r_i, c]:.3f}±{raw_std_mat[r_i, c]:.3f}" for c in range(6)]
            + [round(float(norm_5x6[r_i, c]), 4) for c in range(6)]
        )
    save_csv(
        "exp6c_fig4c_multiseed_with_base.csv",
        ["Test_Metric"] + [f"Raw_{c}" for c in col_names] + [f"Norm_{c}" for c in col_names],
        fig4c_csv,
    )

    # 7. conditioning ablation + oracle-replay real2sim + cem mpc
    print("\n==============================================================================")
    print("EXP 7: Action/Proprio Conditioning Ablation & Oracle-Replay Real2Sim Diagnosis")
    print("==============================================================================")
    cond_base = run_conditioning_ablation(video_base_model, test_loader, vqgan, seq_builder, lpips_fn)
    cond_rlvr = run_conditioning_ablation(rlvr_video_model, test_loader, vqgan, seq_builder, lpips_fn)
    cond_rows = []
    for m_label, c_dict in [("Base (MLE)", cond_base), ("RLVR-World", cond_rlvr)]:
        for mode_k, d_v in c_dict.items():
            print(
                f"  {m_label:<12} | {mode_k:<20} | LPIPS: {d_v['lpips']:.2f} | "
                f"MSE: {d_v['mse']:.3f} | TokenChange vs Normal: {d_v['token_change_rate']:.1f}%"
            )
            cond_rows.append([m_label, mode_k, round(d_v["lpips"], 4), round(d_v["mse"], 4), round(d_v["token_change_rate"], 2)])
    save_csv("exp7a_conditioning_ablation.csv", ["Model", "Conditioning_Mode", "LPIPS_x100", "MSE_x100", "Token_Change_Rate_Pct"], cond_rows)

    real_srs, oracle_srs, base_sim_srs, rlvr_sim_srs, r2s_rows = evaluate_real2sim_with_oracle_replay(
        video_base_model, rlvr_video_model, vqgan, seq_builder, num_trials=50
    )
    print(f"  Real Success Rates        : {[round(x, 2) for x in real_srs]}")
    print(f"  VQGAN Oracle Replay SRs   : {[round(x, 2) for x in oracle_srs]} (isolating 4x4 spatial quantization + detector)")
    print(f"  Base World Model Sim SRs  : {[round(x, 2) for x in base_sim_srs]}")
    print(f"  RLVR-World Model Sim SRs  : {[round(x, 2) for x in rlvr_sim_srs]}")
    save_csv(
        "exp7b_real2sim_with_oracle_replay.csv",
        ["Policy_Checkpoint", "Noise_Level", "N_Trials", "Real_SR", "VQGAN_Oracle_SR", "Base_Sim_SR", "RLVR_Sim_SR"],
        r2s_rows,
    )

    cem_base = [evaluate_cem_mpc(video_base_model, vqgan, seq_builder, lpips_fn, episodes=60, seed=100 + s) for s in range(3)]
    cem_rlvr = [evaluate_cem_mpc(trained_rlvr_models[s], vqgan, seq_builder, lpips_fn, episodes=60, seed=100 + s) for s in range(3)]
    save_csv(
        "exp7c_cem_mpc_multiseed.csv",
        ["Model", "Episodes_Per_Seed", "Seeds", "Mean_Success_Pct", "Std_Success_Pct"],
        [
            ["Base (MLE)", 60, 3, round(float(np.mean(cem_base)), 2), round(float(np.std(cem_base)), 2)],
            ["RLVR-World", 60, 3, round(float(np.mean(cem_rlvr)), 2), round(float(np.std(cem_rlvr)), 2)],
        ],
    )

    # 8. render 6-panel diagnostic figure (with bootstrap cis, base col in 4c, oracle replay in 4d, and 3 random test rollouts in 4e)
    fig = plt.figure(figsize=(19, 12))

    ax1 = plt.subplot(2, 3, 1)
    b_m = [x["base_mean"] for x in bon_stats]
    b_lo = [x["base_mean"] - x["base_ci_low"] for x in bon_stats]
    b_hi = [x["base_ci_high"] - x["base_mean"] for x in bon_stats]
    r_m = [x["rlvr_mean"] for x in bon_stats]
    r_lo = [x["rlvr_mean"] - x["rlvr_ci_low"] for x in bon_stats]
    r_hi = [x["rlvr_ci_high"] - x["rlvr_mean"] for x in bon_stats]
    ax1.errorbar(N_vals, b_m, yerr=[b_lo, b_hi], fmt="o-", capsize=4, lw=2, label="Base (MLE) 95% CI")
    ax1.errorbar(N_vals, r_m, yerr=[r_lo, r_hi], fmt="s-", capsize=4, lw=2, label="RLVR-World 95% CI")
    ax1.set_title(f"(a) Test-Time Scaling: Best-of-N (n={len(test_ds)}, Paired Bootstrap)")
    ax1.set_xlabel("Number of Samples N")
    ax1.set_ylabel("Test LPIPS (x100) ↓")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 3, 2)
    colors_g = {2: "#1f77b4", 4: "#ff7f0e", 8: "#2ca02c"}
    for g_val in [2, 4, 8]:
        arr = np.array(group_val_curves[g_val])
        m, s = arr.mean(axis=0), arr.std(axis=0)
        ax2.plot(eval_steps_axis, m, "o-", lw=2, color=colors_g[g_val], label=f"G = {g_val} (3 seeds)")
        ax2.fill_between(eval_steps_axis, m - s, m + s, color=colors_g[g_val], alpha=0.18)
    ax2.set_title("(b) GRPO Group Size Scaling on Test Set (Mean ± SD)")
    ax2.set_xlabel("GRPO Training Steps")
    ax2.set_ylabel("Test LPIPS (x100) ↓")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 3, 3)
    im = ax3.imshow(norm_5x6, cmap="coolwarm", vmin=0.0, vmax=1.0, aspect="auto")
    ax3.set_xticks(range(6))
    ax3.set_yticks(range(5))
    ax3.set_xticklabels(col_names)
    ax3.set_yticklabels(metrics_list)
    ax3.set_xlabel("Model / Training Reward Metric (3 Seeds)")
    ax3.set_ylabel("Test Evaluation Metric")
    ax3.set_title("(c) Metric Optimization: Norm [Raw Mean±SD] (Negative Result)")
    for i in range(5):
        for j in range(6):
            txt = f"{norm_5x6[i, j]:.2f}\n({raw_mean_mat[i, j]:.1f}±{raw_std_mat[i, j]:.1f})"
            ax3.text(
                j, i, txt, ha="center", va="center",
                color="white" if norm_5x6[i, j] < 0.28 or norm_5x6[i, j] > 0.72 else "black", fontsize=7.5
            )
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = plt.subplot(2, 3, 4)
    ax4.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6, label="Ideal Simulator (y = x)")
    ax4.scatter(real_srs, oracle_srs, s=80, marker="^", color="#2ca02c", label="VQGAN Oracle Replay")
    ax4.scatter(real_srs, base_sim_srs, s=75, marker="o", color="#1f77b4", label="Base World Model")
    ax4.scatter(real_srs, rlvr_sim_srs, s=110, marker="*", color="#ff7f0e", label="RLVR-World")
    ax4.set_title("(d) Real2Sim Eval w/ VQGAN Oracle Replay (n=50/policy)")
    ax4.set_xlabel("Real Environment Success Rate")
    ax4.set_ylabel("Evaluated Success Rate")
    ax4.set_xlim(-0.05, 1.05)
    ax4.set_ylim(-0.05, 1.05)
    ax4.legend(loc="upper left", fontsize=8.5)
    ax4.grid(True, alpha=0.3)

    # panel (e): show 3 fixed random test trajectories side-by-side (indices 0, 7, 19)
    ax5 = plt.subplot(2, 3, (5, 6))
    strips = []
    for test_idx in [0, 7, 19]:
        f_i, p_i, a_i = test_ds[test_idx]
        f_vis = f_i.unsqueeze(0).to(device)
        p_vis = p_i.unsqueeze(0).to(device)
        a_vis = a_i.unsqueeze(0).to(device)
        with torch.no_grad():
            gt_s = seq_builder.build_multistep_eq8(f_vis, p_vis, a_vis, vqgan)
            fc, _ = vqgan.encode_context(f_vis[:, 0])
            pref = gt_s[:, :84]
            f_b = torch.stack([gt_s[:, 102:104], gt_s[:, 122:124], gt_s[:, 142:144]], dim=1)
            oracle_idx = [
                gt_s[:, 64:80] - seq_builder.off_dyn,
                gt_s[:, 84:100] - seq_builder.off_dyn,
                gt_s[:, 104:120] - seq_builder.off_dyn,
                gt_s[:, 124:140] - seq_builder.off_dyn,
            ]
            _, b_vis, _ = rollout_multistep_world_model(video_base_model, pref, f_b, seq_builder, greedy=True)
            _, r_vis, _ = rollout_multistep_world_model(rlvr_video_model, pref, f_b, seq_builder, greedy=True)

            def to_rgb(t_img):
                return np.clip((t_img[0].cpu().permute(1, 2, 0).numpy() + 1.0) * 0.5, 0, 1)

            gt_row = [to_rgb(f_vis[:, t]) for t in range(1, 5)]
            oracle_row = [to_rgb(vqgan.decode_dynamics_indices(oracle_idx[t], fc)) for t in range(4)]
            base_row = [oracle_row[0]] + [to_rgb(vqgan.decode_dynamics_indices(b_vis[:, t], fc)) for t in range(3)]
            rlvr_row = [oracle_row[0]] + [to_rgb(vqgan.decode_dynamics_indices(r_vis[:, t], fc)) for t in range(3)]
            block_strip = np.vstack([np.hstack(gt_row), np.hstack(oracle_row), np.hstack(base_row), np.hstack(rlvr_row)])
            strips.append(block_strip)

    sep = np.ones((strips[0].shape[0], 4, 3), dtype=np.float32)
    full_strip = np.hstack([strips[0], sep, strips[1], sep, strips[2]])
    ax5.imshow(full_strip)
    ax5.set_yticks([16, 48, 80, 112])
    ax5.set_yticklabels(["Ground Truth", "VQGAN Oracle", "Base (MLE)", "RLVR-World"])
    ax5.set_xticks([64, 196, 328])
    ax5.set_xticklabels(["Test Sample #0 (t=0..3)", "Test Sample #7 (t=0..3)", "Test Sample #19 (t=0..3)"])
    ax5.set_title("(e) Multi-Trajectory Qualitative Rollouts Across 3 Fixed Test Indices")

    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, "rlvr_world_scoped_benchmark.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"  -> saved figure: {out_png}")


if __name__ == "__main__":
    main()