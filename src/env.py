import random
import numpy as np
import torch
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SymbolicTextGameEnv:
    VOCAB = [
        "<pad>", "<bos>", "<eos>", "<sep>", "<think>", "</think>",
        "reason:valid", "reason:noop",
        "k:sink=", "k:loc=", "k:dirty=", "k:soap=", "k:won=",
        "v:0", "v:1", "v:2", "v:none",
        "a:0", "a:1", "a:2", "a:3", "a:4", "a:5",
        "s:0", "s:1", "s:2"
    ]

    def __init__(self):
        self.tok2id = {w: i for i, w in enumerate(self.VOCAB)}
        self.vocab_size = len(self.VOCAB)

    def step_dynamics(self, sink: int, loc: int, dirty: int, soap: int, act: int):
        ns, nl, nd, nsoap = sink, loc, dirty, soap
        if act == 1 and loc == 0:
            nl = 1
        elif act == 2 and loc == 1:
            nl = 2
        elif act == 3:
            ns = 1 - sink
        elif act == 4 and loc == 2 and soap == 0:
            nsoap = 1
        elif act == 5 and sink == 1 and loc == 2 and soap == 1 and dirty == 1:
            nd = 0
        changed = (ns != sink) or (nl != loc) or (nd != dirty) or (nsoap != soap)
        won = 1 if (nd == 0 and dirty == 1) else 0
        return (ns, nl, nd, nsoap, won), changed

    def make_sample(self, sink: int, loc: int, dirty: int, soap: int, act: int):
        (ns, nl, nd, nsoap, won), changed = self.step_dynamics(sink, loc, dirty, soap, act)
        prompt = [
            self.tok2id["<bos>"],
            self.tok2id[f"s:{sink}"], self.tok2id[f"s:{loc}"],
            self.tok2id[f"s:{dirty}"], self.tok2id[f"s:{soap}"],
            self.tok2id[f"a:{act}"], self.tok2id["<sep>"]
        ]
        r_tok = "reason:valid" if changed else "reason:noop"
        vs = f"v:{ns}" if ns != sink else "v:none"
        vl = f"v:{nl}" if nl != loc else "v:none"
        vd = f"v:{nd}" if nd != dirty else "v:none"
        vsp = f"v:{nsoap}" if nsoap != soap else "v:none"
        vw = "v:1" if won == 1 else "v:none"

        resp = [
            self.tok2id["<think>"], self.tok2id[r_tok], self.tok2id["</think>"],
            self.tok2id["k:sink="], self.tok2id[vs],
            self.tok2id["k:loc="], self.tok2id[vl],
            self.tok2id["k:dirty="], self.tok2id[vd],
            self.tok2id["k:soap="], self.tok2id[vsp],
            self.tok2id["k:won="], self.tok2id[vw]
        ]
        gt_vals = [self.tok2id[vs], self.tok2id[vl], self.tok2id[vd], self.tok2id[vsp], self.tok2id[vw]]
        return prompt, resp, gt_vals, changed

    def build_full_catalog(self):
        catalog = []
        for s in [0, 1]:
            for l in [0, 1, 2]:
                for d in [0, 1]:
                    for sp in [0, 1]:
                        for a in range(6):
                            catalog.append(self.make_sample(s, l, d, sp, a))
        return catalog


class ContinuousPushTEnv:
    def __init__(self, img_size: int = 32):
        self.img_size = img_size
        self.goal_pos = np.array([0.68, 0.50], dtype=np.float32)
        self.reset()

    def reset(self, agent_pos=None, block_pos=None):
        self.agent = np.array(agent_pos if agent_pos is not None else np.random.uniform(0.20, 0.78, size=2), dtype=np.float32)
        self.block = np.array(block_pos if block_pos is not None else np.random.uniform(0.26, 0.72, size=2), dtype=np.float32)
        return self.render(), self.agent.copy()

    def step(self, action: np.ndarray):
        action = np.clip(action, -0.15, 0.15)
        new_agent = np.clip(self.agent + action, 0.12, 0.88)
        dist = np.linalg.norm(new_agent - self.block)
        if dist < 0.19:
            push_vec = new_agent - self.agent
            self.block = np.clip(self.block + push_vec * 1.15, 0.14, 0.86)
        self.agent = new_agent
        return self.render(), self.agent.copy(), {"block": self.block.copy(), "agent": self.agent.copy()}

    def expert_pd_action(self) -> np.ndarray:
        block_to_goal = self.goal_pos - self.block
        dist_to_goal = np.linalg.norm(block_to_goal)
        if dist_to_goal < 0.04:
            return np.zeros(2, dtype=np.float32)
        push_dir = block_to_goal / (dist_to_goal + 1e-6)
        desired_target = self.block + push_dir * min(0.12, dist_to_goal * 0.85)
        act = desired_target - self.agent
        return np.clip(act, -0.15, 0.15).astype(np.float32)

    def render(self) -> np.ndarray:
        H = W = self.img_size
        y, x = np.mgrid[0:H, 0:W] / float(H)
        img = np.ones((H, W, 3), dtype=np.float32) * np.array([0.86, 0.79, 0.69], dtype=np.float32)
        grid = ((np.arange(H) % 8 == 0)[:, None] | (np.arange(W) % 8 == 0)[None, :])
        img[grid] *= 0.92

        gx, gy = self.goal_pos
        goal_mask = ((x - gx) ** 2 + (y - gy) ** 2) < (0.09 ** 2)
        img[goal_mask] = img[goal_mask] * 0.65 + np.array([0.20, 0.80, 0.30], dtype=np.float32) * 0.35

        bx, by = self.block
        b_outer = (np.abs(x - bx) < 0.105) & (np.abs(y - by) < 0.105)
        b_inner = (np.abs(x - bx) < 0.065) & (np.abs(y - by) < 0.065)
        img[b_outer] = np.array([0.10, 0.40, 0.85], dtype=np.float32)
        img[b_inner] = np.array([0.25, 0.58, 0.96], dtype=np.float32)

        ax, ay = self.agent
        a_outer = ((x - ax) ** 2 + (y - ay) ** 2) < (0.095 ** 2)
        a_inner = ((x - ax) ** 2 + (y - ay) ** 2) < (0.055 ** 2)
        img[a_outer] = np.array([0.92, 0.12, 0.12], dtype=np.float32)
        img[a_inner] = np.array([1.00, 0.35, 0.35], dtype=np.float32)

        return np.transpose((img * 2.0) - 1.0, (2, 0, 1))


class DisjointManipulationDataset(Dataset):
    def __init__(self, num_trajs: int = 800, split: str = "train"):
        self.data = []
        env = ContinuousPushTEnv()
        while len(self.data) < num_trajs:
            if split == "train":
                a_init = np.random.uniform([0.20, 0.20], [0.75, 0.78])
                b_init = np.random.uniform([0.26, 0.25], [0.72, 0.75])
                if abs(a_init[1] - 0.50) < 0.04 and abs(b_init[1] - 0.50) < 0.04:
                    continue
            else:
                a_init = np.random.uniform([0.22, 0.35], [0.65, 0.65])
                b_init = np.random.uniform([0.32, 0.38], [0.68, 0.62])

            f0, p0 = env.reset(agent_pos=a_init, block_pos=b_init)
            frames, proprios, actions = [f0], [p0], []
            # 50/50 push vs random exploration so world model cannot coast on an unconditional push prior
            for _ in range(4):
                if np.random.rand() < 0.50:
                    vec = env.block - env.agent
                    act = (vec / (np.linalg.norm(vec) + 1e-6)) * np.random.uniform(0.07, 0.14) + np.random.normal(0, 0.02, 2)
                else:
                    act = np.random.uniform(-0.14, 0.14, size=2)
                act = np.clip(act, -0.15, 0.15).astype(np.float32)
                fn, pn, _ = env.step(act)
                frames.append(fn)
                proprios.append(pn)
                actions.append(act)
            self.data.append((np.stack(frames), np.stack(proprios), np.stack(actions)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        f, p, a = self.data[idx]
        return torch.tensor(f), torch.tensor(p), torch.tensor(a)