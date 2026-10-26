import os
import sys
import csv
import copy
import random
from typing import List
import numpy as np
import pandas as pd
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
from src.mechanism_checks import run_mechanism_checks

RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def save_csv(filename: str, headers: List[str], rows: List[List]):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"  -> saved csv: {path}")


def plot_from_csv():
    # plot-only path: reads existing csvs in results/ without retraining a single weight
    df_t3 = pd.read_csv(os.path.join(RESULTS_DIR, "exp5_table3_multiseed.csv"))
    df_g = pd.read_csv(os.path.join(RESULTS_DIR, "exp6a_group_size_scaling.csv"))
    df_4c = pd.read_csv(os.path.join(RESULTS_DIR, "exp6b_fig4c_raw_and_norm.csv"))
    df_r2s = pd.read_csv(os.path.join(RESULTS_DIR, "exp7_real2sim_evaluation.csv"))

    fig = plt.figure(figsize=(16, 9))

    ax1 = plt.subplot(2, 2, 1)
    x_pos = np.arange(len(df_t3))
    ax1.bar(
        x_pos,
        df_t3["LPIPS_Mean"],
        yerr=df_t3["LPIPS_Std"],
        capsize=5,
        color=["#1f77b4", "#7f7f7f", "#ff7f0e", "#2ca02c"],
    )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(df_t3["Model"], rotation=15, ha="right")
    ax1.set_ylabel("Test LPIPS (x100) ↓")
    ax1.set_title("(a) Multi-Step Video Prediction (n=120, 3 Seeds)")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = plt.subplot(2, 2, 2)
    colors_g = {2: "#1f77b4", 4: "#ff7f0e", 8: "#2ca02c"}
    for g_val in [2, 4, 8]:
        sub = df_g[df_g["Group_Size"] == g_val]
        steps = sub["Step"].values
        m = sub["Test_LPIPS_Mean"].values
        s = sub["Test_LPIPS_Std"].values
        ax2.plot(steps, m, "o-", lw=2, color=colors_g[g_val], label=f"G = {g_val} (3 seeds)")
        ax2.fill_between(steps, m - s, m + s, color=colors_g[g_val], alpha=0.18)
    ax2.set_xlabel("GRPO Training Steps")
    ax2.set_ylabel("Test LPIPS (x100) ↓")
    ax2.set_title("(b) GRPO Group Size Scaling on Test Set (Mean ± SD)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 2, 3)
    metrics_list = ["MAE", "MSE", "PSNR", "SSIM", "LPIPS"]
    norm_cols = [f"Norm_{m}" for m in metrics_list]
    raw_cols = [f"Raw_{m}" for m in metrics_list]
    norm_mat = df_4c[norm_cols].values
    raw_mat = df_4c[raw_cols].values
    im = ax3.imshow(norm_mat, cmap="coolwarm", vmin=0.0, vmax=1.0, aspect="auto")
    ax3.set_xticks(range(5))
    ax3.set_yticks(range(5))
    ax3.set_xticklabels(metrics_list)
    ax3.set_yticklabels(metrics_list)
    ax3.set_xlabel("Training Reward Metric")
    ax3.set_ylabel("Test Evaluation Metric")
    ax3.set_title("(c) Metric-Oriented Optimization (Norm [Raw])")
    for i in range(5):
        for j in range(5):
            txt = f"{norm_mat[i, j]:.2f}\n({raw_mat[i, j]:.1f})"
            ax3.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color="white" if norm_mat[i, j] < 0.28 or norm_mat[i, j] > 0.72 else "black",
                fontsize=8,
            )
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = plt.subplot(2, 2, 4)
    real_srs = df_r2s["Real_SR"].values
    base_srs = df_r2s["Base_Sim_SR"].values
    rlvr_srs = df_r2s["RLVR_Sim_SR"].values
    ax4.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6, label="Ideal Simulator (y = x)")
    ax4.scatter(real_srs, base_srs, s=75, marker="o", color="#1f77b4", label="Base World Model")
    ax4.scatter(real_srs, rlvr_srs, s=110, marker="*", color="#ff7f0e", label="RLVR-World (Ours)")
    ax4.set_xlabel("Real Environment Success Rate")
    ax4.set_ylabel("Simulated Success Rate")
    ax4.set_title("(d) Real2Sim Policy Evaluation (6 Policies, n=50 each)")
    ax4.set_xlim(-0.05, 1.05)
    ax4.set_ylim(-0.05, 1.05)
    ax4.legend(loc="upper left")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "csv_summary_plots.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"  -> regenerated summary figure from csvs: {out_path}")


def train_and_evaluate_all():
    run_mechanism_checks()
    lpips_fn = PretrainedLPIPS().to(device)

    text_env = SymbolicTextGameEnv()
    full_catalog = text_env.build_full_catalog()
    text_test_set = [full_catalog[i % len(full_catalog)] for i in range(400)]

    lang_results = {
        k: {"Unchanged": [], "Changed": [], "Overall": []}
        for k in ["Base", "SFT", "RLVR_Binary", "RLVR_TaskEq4"]
    }

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
        exp1_rows.append([m_name, f"{u_m:.2f}±{u_s:.2f}", f"{c_m:.2f}±{c_s:.2f}", f"{o_m:.2f}±{o_s:.2f}", 400, 3])
    save_csv("exp1_language_world_model.csv", ["Model", "Unchanged_Acc", "Changed_Acc", "Overall_Acc", "N_Test", "Seeds"], exp1_rows)

    seed_everything(42)
    train_ds = DisjointManipulationDataset(num_trajs=900, split="train")
    test_ds = DisjointManipulationDataset(num_trajs=120, split="test")
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    vqgan = DualCompressiveVQGAN(fsq_levels=[5, 5, 5], ch=64).to(device)
    opt_vq = torch.optim.AdamW(vqgan.parameters(), lr=2.5e-3, weight_decay=1e-4)

    for _ in range(16):
        vqgan.train()
        for frames, _, _ in train_loader:
            frames = frames.to(device)
            s_c, s_t = frames[:, 0], frames[:, random.randint(1, 4)]
            rc, rt = vqgan(s_c, s_t)
            loss = (
                weighted_foreground_l1(rc, s_c)
                + weighted_foreground_l1(rt, s_t)
                + 0.5 * (lpips_fn(rc, s_c).mean() + lpips_fn(rt, s_t).mean())
            )
            opt_vq.zero_grad()
            loss.backward()
            opt_vq.step()

    vqgan.eval()
    for p in vqgan.parameters():
        p.requires_grad = False

    seq_builder = UnifiedWorldSequenceBuilder()
    ms_target_mask = seq_builder.get_multistep_target_mask()

    video_base_model = MicroLLaMA(vocab_size=seq_builder.vocab_size, dim=192, n_layers=4, n_heads=6, max_len=256).to(device)
    mle_opt = torch.optim.AdamW(video_base_model.parameters(), lr=1.2e-3, weight_decay=0.02)

    for _ in range(12):
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

    g_csv_rows = []
    for g_val in [2, 4, 8]:
        arr = np.array(group_val_curves[g_val])
        for st, m, s in zip(eval_steps_axis, arr.mean(axis=0), arr.std(axis=0)):
            g_csv_rows.append([g_val, st, round(float(m), 4), round(float(s), 4)])
    save_csv("exp6a_group_size_scaling.csv", ["Group_Size", "Step", "Test_LPIPS_Mean", "Test_LPIPS_Std"], g_csv_rows)

    metrics_list = ["MAE", "MSE", "PSNR", "SSIM", "LPIPS"]
    col_names = ["Base (MLE)"] + metrics_list
    raw_display_matrix = np.zeros((5, 6))
    score_matrix = np.zeros((5, 6))

    raw_display_matrix[:, 0] = [base_eval_det["mae"], base_eval_det["mse"], base_eval_det["psnr"], base_eval_det["ssim"], base_eval_det["lpips"]]
    score_matrix[:, 0] = [-base_eval_det["mae"], -base_eval_det["mse"], base_eval_det["psnr"], base_eval_det["ssim"], -base_eval_det["lpips"]]

    for c_idx, m_name in enumerate(metrics_list, start=1):
        seed_everything(42 + c_idx)
        m_spec, _ = train_video_rlvr_grpo(video_base_model, train_loader, vqgan, seq_builder, lpips_fn, group_size=8, steps=35, metric_mode=m_name, lr=8e-5)
        ev_m = evaluate_video_model_table3(m_spec, test_loader, vqgan, seq_builder, lpips_fn)
        raw_display_matrix[:, c_idx] = [ev_m["mae"], ev_m["mse"], ev_m["psnr"], ev_m["ssim"], ev_m["lpips"]]
        score_matrix[:, c_idx] = [-ev_m["mae"], -ev_m["mse"], ev_m["psnr"], ev_m["ssim"], -ev_m["lpips"]]

    norm_5x5 = np.zeros((5, 5))
    for r in range(5):
        sub = score_matrix[r, 1:]
        norm_5x5[r] = (sub - sub.min()) / (sub.max() - sub.min() + 1e-8)

    fig4c_csv = []
    row_labels = ["MAE(x100)↓", "MSE(x100)↓", "PSNR(dB)↑", "SSIM(x100)↑", "LPIPS(x100)↓"]
    for r_idx, r_name in enumerate(row_labels):
        fig4c_csv.append([r_name] + [round(float(x), 4) for x in raw_display_matrix[r_idx]] + [round(float(x), 4) for x in norm_5x5[r_idx]])
    save_csv("exp6b_fig4c_raw_and_norm.csv", ["Test_Metric"] + [f"Raw_{c}" for c in col_names] + [f"Norm_{c}" for c in metrics_list], fig4c_csv)

    real_srs, oracle_srs, base_sim_srs, rlvr_sim_srs, r2s_rows = evaluate_real2sim_with_oracle_replay(
        video_base_model, rlvr_video_model, vqgan, seq_builder, num_trials=50
    )
    save_csv(
        "exp7_real2sim_evaluation.csv",
        ["Policy_Checkpoint", "Noise_Level", "N_Trials", "Real_SR", "VQGAN_Oracle_SR", "Base_Sim_SR", "RLVR_Sim_SR"],
        r2s_rows,
    )
    plot_from_csv()


def main():
    run_mechanism_checks()
    if "--train" in sys.argv:
        train_and_evaluate_all()
    else:
        plot_from_csv()


if __name__ == "__main__":
    main()
