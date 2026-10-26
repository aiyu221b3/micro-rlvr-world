import copy
import random
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.env import device, SymbolicTextGameEnv
from src.networks import (
    MicroLLaMA,
    DualCompressiveVQGAN,
    UnifiedWorldSequenceBuilder,
    PretrainedLPIPS,
    compute_ssim,
)


def weighted_foreground_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    fg_mask = ((target[:, 0:1] - target[:, 1:2]) > 0.45) | ((target[:, 2:3] - target[:, 0:1]) > 0.45)
    weights = torch.where(fg_mask, 6.0, 1.0)
    return ((pred - target).abs() * weights).mean()


def compute_text_reward(pred_resp: torch.Tensor, gt_vals: torch.Tensor, text_env: SymbolicTextGameEnv, mode: str) -> torch.Tensor:
    pred_vals = pred_resp[:, [4, 6, 8, 10, 12]]
    none_id = text_env.tok2id["v:none"]
    matches = (pred_vals == gt_vals).float()
    acc_all = matches.mean(dim=-1)
    exact = (matches.sum(dim=-1) == 5).float()

    if mode == "binary":
        return exact

    changed_mask = (gt_vals != none_id).float()
    num_ch = changed_mask.sum(dim=-1)
    pred_ch_mask = (pred_vals != none_id).float()

    acc_ch = torch.zeros_like(acc_all)
    for i in range(pred_vals.size(0)):
        if num_ch[i] == 0:
            acc_ch[i] = 1.0 if pred_ch_mask[i].sum() == 0 else 0.0
        else:
            acc_ch[i] = (matches[i] * changed_mask[i]).sum() / num_ch[i]
    return 0.1 * acc_all + 1.0 * acc_ch + 0.2 * exact


@torch.no_grad()
def rollout_multistep_world_model(
    model: MicroLLaMA,
    prefix_84: torch.Tensor,
    future_actions_b1_b2_b3: torch.Tensor,
    seq_builder: UnifiedWorldSequenceBuilder,
    temperature: float = 1.0,
    top_k: int = 50,
    greedy: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # ip -> prefix_84 [z_c, z_0, p_0, b_0] + future acts; op -> full seq + vis tokens + decoded proprio
    model.eval()
    curr = prefix_84.clone()
    vis_list, prop_list = [], []

    for step in range(3):
        v_toks = []
        for _ in range(16):
            logits = model(curr)[:, -1, :] / max(temperature, 1e-5)
            masked = torch.full_like(logits, float("-inf"))
            masked[:, 125:250] = logits[:, 125:250]
            if greedy:
                nxt = torch.argmax(masked, dim=-1, keepdim=True)
            else:
                if top_k > 0:
                    v, _ = torch.topk(masked, min(top_k, 125))
                    masked[masked < v[:, [-1]]] = float("-inf")
                nxt = torch.multinomial(F.softmax(masked, dim=-1), num_samples=1)
            curr = torch.cat([curr, nxt], dim=1)
            v_toks.append(nxt - 125)
        vis_list.append(torch.cat(v_toks, dim=1))

        p_toks = []
        for _ in range(2):
            logits = model(curr)[:, -1, :] / max(temperature, 1e-5)
            masked = torch.full_like(logits, float("-inf"))
            masked[:, 250:506] = logits[:, 250:506]
            if greedy:
                nxt = torch.argmax(masked, dim=-1, keepdim=True)
            else:
                if top_k > 0:
                    v, _ = torch.topk(masked, min(top_k, 256))
                    masked[masked < v[:, [-1]]] = float("-inf")
                nxt = torch.multinomial(F.softmax(masked, dim=-1), num_samples=1)
            curr = torch.cat([curr, nxt], dim=1)
            p_toks.append(nxt)
        p_cat = torch.cat(p_toks, dim=1)
        prop_list.append(seq_builder.dequantize_vec(p_cat, 0.10, 0.90))
        curr = torch.cat([curr, future_actions_b1_b2_b3[:, step, :]], dim=1)

    return curr, torch.stack(vis_list, dim=1), torch.stack(prop_list, dim=1)


@torch.no_grad()
def compute_video_verifiable_rewards(
    pred_vis_idx: torch.Tensor,
    pred_prop: torch.Tensor,
    gt_frames_123: torch.Tensor,
    gt_prop_123: torch.Tensor,
    feat_c: torch.Tensor,
    z0_idx: torch.Tensor,
    vqgan: DualCompressiveVQGAN,
    lpips_fn: PretrainedLPIPS,
    metric_mode: str = "l1_lpips_prop",
    rep_penalty_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    BG = pred_vis_idx.size(0)
    mae_acc = torch.zeros(BG, device=device)
    mse_acc = torch.zeros(BG, device=device)
    psnr_acc = torch.zeros(BG, device=device)
    ssim_acc = torch.zeros(BG, device=device)
    lpips_acc = torch.zeros(BG, device=device)
    fg_l1_acc = torch.zeros(BG, device=device)
    rep_acc = torch.zeros(BG, device=device)

    prev_idx = z0_idx
    for t in range(3):
        idx_t = pred_vis_idx[:, t, :]
        rep_acc += ((idx_t == prev_idx).float().mean(dim=-1) >= 0.75).float() / 3.0
        prev_idx = idx_t

        dec_rgb = vqgan.decode_dynamics_indices(idx_t, feat_c)
        gt_rgb = gt_frames_123[:, t]
        dec_01 = torch.clamp((dec_rgb + 1.0) * 0.5, 0.0, 1.0)
        gt_01 = torch.clamp((gt_rgb + 1.0) * 0.5, 0.0, 1.0)

        fg_mask = ((gt_rgb[:, 0:1] - gt_rgb[:, 1:2]) > 0.45) | ((gt_rgb[:, 2:3] - gt_rgb[:, 0:1]) > 0.45)
        fg_weights = torch.where(fg_mask, 5.0, 1.0)
        fg_l1_t = ((dec_01 - gt_01).abs() * fg_weights).mean(dim=[1, 2, 3])

        mae_t = (dec_01 - gt_01).abs().mean(dim=[1, 2, 3])
        mse_t = (dec_01 - gt_01).pow(2).mean(dim=[1, 2, 3])
        psnr_t = 10.0 * torch.log10(1.0 / (mse_t + 1e-6))
        ssim_t = compute_ssim(dec_rgb, gt_rgb)
        lpips_t = lpips_fn(dec_rgb, gt_rgb)

        fg_l1_acc += fg_l1_t / 3.0
        mae_acc += mae_t / 3.0
        mse_acc += mse_t / 3.0
        psnr_acc += psnr_t / 3.0
        ssim_acc += ssim_t / 3.0
        lpips_acc += lpips_t / 3.0

    prop_mse = (pred_prop - gt_prop_123).pow(2).mean(dim=[1, 2])

    if metric_mode == "l1_lpips_prop":
        rewards = -(2.0 * fg_l1_acc + lpips_acc + 0.5 * prop_mse) - rep_penalty_weight * rep_acc
    elif metric_mode == "MAE":
        rewards = -mae_acc * 100.0
    elif metric_mode == "MSE":
        rewards = -mse_acc * 100.0
    elif metric_mode == "PSNR":
        rewards = psnr_acc
    elif metric_mode == "SSIM":
        rewards = ssim_acc * 100.0
    elif metric_mode == "LPIPS":
        rewards = -lpips_acc * 100.0
    else:
        raise ValueError(f"unknown metric_mode: {metric_mode}")

    stats = {
        "reward": rewards.mean().item(),
        "mae": mae_acc.mean().item(),
        "mse": mse_acc.mean().item(),
        "psnr": psnr_acc.mean().item(),
        "ssim": ssim_acc.mean().item(),
        "lpips": lpips_acc.mean().item(),
        "rep_rate": rep_acc.mean().item(),
    }
    return rewards, stats


def get_masked_logprobs(model: MicroLLaMA, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits = model(seq[:, :-1])
    logp = F.log_softmax(logits[:, mask, :], dim=-1)
    toks = seq[:, 1:][:, mask]
    return torch.gather(logp, dim=-1, index=toks.unsqueeze(-1)).squeeze(-1)


def train_video_rlvr_grpo(
    base_init: MicroLLaMA,
    train_loader: DataLoader,
    vqgan: DualCompressiveVQGAN,
    seq_builder: UnifiedWorldSequenceBuilder,
    lpips_fn: PretrainedLPIPS,
    group_size: int = 8,
    steps: int = 40,
    metric_mode: str = "l1_lpips_prop",
    rep_penalty_weight: float = 0.0,
    eval_every: int = 0,
    val_subset_loader: DataLoader = None,
    eval_fn=None,
    lr: float = 7e-5,
) -> Tuple[MicroLLaMA, Dict[str, List[float]]]:
    model = copy.deepcopy(base_init).to(device)
    ref_model = copy.deepcopy(base_init).eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    ms_target_mask = seq_builder.get_multistep_target_mask()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history = {"reward": [], "mae": [], "lpips": [], "rep_rate": [], "val_lpips": []}
    loader_iter = iter(train_loader)

    if eval_every > 0 and val_subset_loader is not None and eval_fn is not None:
        ev0 = eval_fn(model, val_subset_loader, vqgan, seq_builder, lpips_fn)
        history["val_lpips"].append(ev0["lpips"])

    for step in range(1, steps + 1):
        try:
            frames, proprios, actions = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            frames, proprios, actions = next(loader_iter)

        frames = frames[:8].to(device)
        proprios = proprios[:8].to(device)
        actions = actions[:8].to(device)
        B = frames.size(0)

        with torch.no_grad():
            gt_seq = seq_builder.build_multistep_eq8(frames, proprios, actions, vqgan)
            feat_c, _ = vqgan.encode_context(frames[:, 0])
            prefix_84 = gt_seq[:, :84]
            z0_idx = gt_seq[:, 64:80] - seq_builder.off_dyn
            fut_acts = torch.stack([gt_seq[:, 102:104], gt_seq[:, 122:124], gt_seq[:, 142:144]], dim=1)

            prefix_G = prefix_84.repeat_interleave(group_size, dim=0)
            fut_acts_G = fut_acts.repeat_interleave(group_size, dim=0)
            feat_c_G = feat_c.repeat_interleave(group_size, dim=0)
            gt_f_G = frames[:, 2:5].repeat_interleave(group_size, dim=0)
            gt_p_G = proprios[:, 2:5].repeat_interleave(group_size, dim=0)
            z0_G = z0_idx.repeat_interleave(group_size, dim=0)

            sampled_seq, pred_vis, pred_prop = rollout_multistep_world_model(
                model, prefix_G, fut_acts_G, seq_builder, temperature=1.0, top_k=40, greedy=False
            )
            old_logp = get_masked_logprobs(model, sampled_seq, ms_target_mask)
            ref_logp = get_masked_logprobs(ref_model, sampled_seq, ms_target_mask)

            rewards, stats = compute_video_verifiable_rewards(
                pred_vis, pred_prop, gt_f_G, gt_p_G, feat_c_G, z0_G,
                vqgan, lpips_fn, metric_mode=metric_mode, rep_penalty_weight=rep_penalty_weight
            )
            # group-relative advantage: zero-mean & unit-std per prompt, lest reward scale playeth the tyrant
            r_grp = rewards.view(B, group_size)
            r_std = r_grp.std(dim=1, keepdim=True)
            valid_mask = (r_std > 1e-5).float()
            adv = ((r_grp - r_grp.mean(dim=1, keepdim=True)) / r_std.clamp(min=1e-4)) * valid_mask
            adv = adv.view(B * group_size, 1)

        model.train()
        perm = torch.randperm(B * group_size, device=device)
        for mb in perm.chunk(2):
            new_logp = get_masked_logprobs(model, sampled_seq[mb], ms_target_mask)
            ratio = torch.exp(new_logp - old_logp[mb])
            s1 = ratio * adv[mb]
            s2 = torch.clamp(ratio, 0.8, 1.2) * adv[mb]
            log_r_ref = ref_logp[mb] - new_logp
            kl = torch.exp(log_r_ref) - log_r_ref - 1.0
            loss = -(torch.min(s1, s2) - 1e-3 * kl).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        history["reward"].append(stats["reward"])
        history["mae"].append(stats["mae"])
        history["lpips"].append(stats["lpips"])
        history["rep_rate"].append(stats["rep_rate"])

        if eval_every > 0 and val_subset_loader is not None and eval_fn is not None and (step % eval_every == 0):
            ev_s = eval_fn(model, val_subset_loader, vqgan, seq_builder, lpips_fn)
            history["val_lpips"].append(ev_s["lpips"])

    return model, history