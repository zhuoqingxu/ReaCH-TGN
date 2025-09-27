# tgn_contrast_hop_time_fast_noamp.py
# Task: temporal reachability
# Model: TGN + contrastive (NT-Xent) + hop penalty + time-gap penalty

import time, random
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ===================== Hyperparameters =====================
NUM_EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-3
EMB_DIM = 64
MEM_DIM = 64
N_PAIRS = 10000
SEED = 42

DROP_RATE = 0.2
NOISE_STD_RATIO = 0.1
CONTRAST_WEIGHT = 0.1
TAU = 0.5
CONTRAST_EVERY = 1

TIME_GAP_LAMBDA = 1e-3    # time-gap penalty strength
HOP_MAX = 5
HOP_GAMMA = 2.0           # hop penalty exponent

NUM_WORKERS = 4
PIN_MEMORY = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

# ===================== Utility functions =====================
def nt_xent_loss(z1, z2, tau=TAU):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / tau
    mask = (~torch.eye(2*N, 2*N, dtype=torch.bool, device=z.device))
    exp_sim = torch.exp(sim) * mask
    pos_sim = torch.exp((z1 * z2).sum(dim=1) / tau)
    denom = exp_sim.sum(dim=1)
    loss = -torch.log(torch.cat([pos_sim, pos_sim], dim=0) / denom)
    return loss.mean()

def hop_weight_tensor(h, max_hop=HOP_MAX, gamma=HOP_GAMMA):
    # hop penalty
    table = torch.arange(0, max_hop+1, device=h.device, dtype=torch.float)
    table = (max_hop - table + 1).pow_(gamma)
    h_clip = torch.clamp(h, min=0, max=max_hop)
    w = torch.ones_like(h, dtype=torch.float)
    mask = h.ge(0)
    w[mask] = table[h_clip[mask]]
    return w

def make_aug_stream(es_src, es_t, drop_rate, noise_std_ratio):
    if es_t.numel() == 0:
        return es_src[:0], es_t[:0]
    keep = torch.rand_like(es_t) > drop_rate
    if keep.sum() == 0:
        return es_src[:0], es_t[:0]
    ts = es_t[keep]
    src = es_src[keep]
    ts_safe = torch.clamp(ts, min=1.0)
    noise = torch.randn_like(ts) * (noise_std_ratio * ts_safe)
    t1 = (ts + noise).clamp_min_(0.)
    jitter = (torch.rand_like(ts) * 2 - 1.) * noise_std_ratio
    t2 = (t1 + jitter).clamp_min_(0.)
    idx = torch.argsort(t2)
    return src[idx], t2[idx]

# ===================== Data =====================
def load_btc_edges(path="btcalpha_edges_with_day.csv"):
    df = pd.read_csv(path)
    nodes = pd.unique(df[["u", "v"]].values.ravel())
    id2idx = {nid: i for i, nid in enumerate(sorted(nodes))}
    df["u"] = df["u"].map(id2idx)
    df["v"] = df["v"].map(id2idx)
    df = df.sort_values("day").reset_index(drop=True)

    srcs = df["u"].to_numpy(dtype=int)
    dsts = df["v"].to_numpy(dtype=int)
    times = df["day"].to_numpy(dtype=float)

    out_adj = defaultdict(list)
    for u, v, t in zip(srcs, dsts, times):
        out_adj[u].append((t, v))
    for u in out_adj:
        out_adj[u].sort(key=lambda x: x[0])

    edges = list(zip(srcs, dsts, times))
    n_nodes = len(nodes)
    return edges, out_adj, n_nodes, srcs, times

def generate_pairs(out_adj, all_nodes, times, n_pairs=N_PAIRS, max_hop=None):
    pairs = []
    def reach(u, v, tq):
        dq = deque([(u, 0.0, 0)])
        vis = {u}
        while dq:
            node, cur_t, hops = dq.popleft()
            if node == v:
                return True, hops
            if max_hop is not None and hops >= max_hop:
                continue
            for te, nbr in out_adj.get(node, []):
                if cur_t < te <= tq and nbr not in vis:
                    vis.add(nbr)
                    dq.append((nbr, te, hops+1))
        return False, None

    random.seed(SEED)
    for _ in range(n_pairs):
        u = random.choice(all_nodes)
        v = random.choice(all_nodes)
        tq = float(random.choice(times))
        flag, h = reach(u, v, tq)
        pairs.append((u, v, tq, float(flag), h if h is not None else -1))
    return pairs

class PairDataset(Dataset):
    def __init__(self, pairs): self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        u, v, tq, y, h = self.pairs[idx]
        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(v, dtype=torch.long),
            torch.tensor(tq, dtype=torch.float),
            torch.tensor(y, dtype=torch.float),
            torch.tensor(h, dtype=torch.long),
        )

# ===================== TGN backbone =====================
class TGN(nn.Module):
    def __init__(self, n_nodes, mem_dim=MEM_DIM, emb_dim=EMB_DIM):
        super().__init__()
        self.n_nodes = n_nodes
        self.register_buffer("mem", torch.zeros(n_nodes, mem_dim))
        self.register_buffer("last", torch.zeros(n_nodes))
        self.msg = nn.Sequential(nn.Linear(mem_dim + 1, mem_dim), nn.ReLU())
        self.gru = nn.GRUCell(mem_dim, mem_dim)
        self.emb = nn.Sequential(nn.Linear(mem_dim + 1, emb_dim), nn.ReLU())

    @torch.no_grad()
    def reset_states(self):
        self.mem.zero_()
        self.last.zero_()

    @torch.no_grad()
    def update(self, srcs, ts):
        if srcs.numel() == 0:
            return
        for s, t in zip(srcs.tolist(), ts.tolist()):
            s = int(s)
            delta = t - float(self.last[s])
            m = self.mem[s:s+1]
            msg = self.msg(torch.cat([m, torch.tensor([[delta]], device=m.device)], dim=1))
            h_new = self.gru(msg, m)
            self.mem[s] = h_new.squeeze(0)
            self.last[s] = t

    def embed(self, nodes, tq):
        m = self.mem[nodes]
        d = (tq - self.last[nodes]).unsqueeze(1)
        return self.emb(torch.cat([m, d], dim=1))

class Pred(nn.Module):
    def __init__(self, emb_dim=EMB_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2*emb_dim, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, zu, zv):
        return self.mlp(torch.cat([zu, zv], dim=1)).squeeze(1)

# ===================== Training =====================
def run(max_hop=None):
    edges, out_adj, n_nodes, srcs_np, times_np = load_btc_edges()
    all_nodes = list(range(n_nodes))
    pairs = generate_pairs(out_adj, all_nodes, times_np, n_pairs=N_PAIRS, max_hop=max_hop)

    pairs.sort(key=lambda x: x[2])
    n = len(pairs); ntr = int(0.7*n); nval = int(0.1*n)
    train_pairs = pairs[:ntr]
    test_pairs  = pairs[ntr+nval:]

    train_loader = DataLoader(PairDataset(train_pairs),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=(NUM_WORKERS>0))
    test_loader  = DataLoader(PairDataset(test_pairs),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=(NUM_WORKERS>0))

    model = TGN(n_nodes).to(device)
    pred  = Pred().to(device)
    opt   = torch.optim.Adam(list(model.parameters()) + list(pred.parameters()), lr=LR)

    es_src = torch.tensor(srcs_np, dtype=torch.long, device=device)
    es_t   = torch.tensor(times_np, dtype=torch.float, device=device)
    t_min  = es_t[0].item()

    for ep in range(1, NUM_EPOCHS+1):
        model.reset_states()
        p_base = 0

        aug1_src, aug1_t = make_aug_stream(es_src, es_t, DROP_RATE, NOISE_STD_RATIO)
        aug2_src, aug2_t = make_aug_stream(es_src, es_t, DROP_RATE, NOISE_STD_RATIO)
        p_aug1 = 0; p_aug2 = 0

        ys, ps = [], []
        for step, (u_cpu, v_cpu, tq_cpu, y_cpu, h_cpu) in enumerate(train_loader, start=1):
            u = u_cpu.to(device, non_blocking=True)
            v = v_cpu.to(device, non_blocking=True)
            tq = tq_cpu.to(device, non_blocking=True)
            y  = y_cpu.to(device, non_blocking=True)
            h  = h_cpu.to(device, non_blocking=True)

            tq_max = float(tq.max())
            new_p = torch.searchsorted(es_t, torch.tensor(tq_max, device=es_t.device)).item()
            if new_p > p_base:
                model.update(es_src[p_base:new_p], es_t[p_base:new_p])
                p_base = new_p

            mem_base = model.mem.clone()
            last_base = model.last.clone()

            use_contrast = (step % CONTRAST_EVERY == 0)
            if use_contrast and aug1_t.numel() > 0 and aug2_t.numel() > 0:
                new_p1 = torch.searchsorted(aug1_t, torch.tensor(tq_max, device=aug1_t.device)).item()
                if new_p1 > p_aug1:
                    model.update(aug1_src[p_aug1:new_p1], aug1_t[p_aug1:new_p1])
                    p_aug1 = new_p1
                hu1 = model.embed(u, tq)
                model.mem.copy_(mem_base); model.last.copy_(last_base)

                new_p2 = torch.searchsorted(aug2_t, torch.tensor(tq_max, device=aug2_t.device)).item()
                if new_p2 > p_aug2:
                    model.update(aug2_src[p_aug2:new_p2], aug2_t[p_aug2:new_p2])
                    p_aug2 = new_p2
                hu2 = model.embed(u, tq)
                model.mem.copy_(mem_base); model.last.copy_(last_base)
            else:
                hu1 = hu2 = None

            zu = model.embed(u, tq); zv = model.embed(v, tq)
            logits = pred(zu, zv)

            # hop + time penalty
            time_weights = torch.exp(-TIME_GAP_LAMBDA * (tq - t_min))
            hop_weights = hop_weight_tensor(h, max_hop=HOP_MAX, gamma=HOP_GAMMA)
            w = hop_weights * time_weights

            cls_loss = F.binary_cross_entropy_with_logits(logits, y, weight=w)
            loss = cls_loss
            if use_contrast and (hu1 is not None) and (hu2 is not None):
                c_loss = nt_xent_loss(hu1, hu2, tau=TAU)
                loss = loss + CONTRAST_WEIGHT * c_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            probs = torch.sigmoid(logits)
            ys.extend(y.detach().cpu().tolist())
            ps.extend(probs.detach().cpu().tolist())

        fpr, tpr, _ = roc_curve(ys, ps)
        print(f"Epoch {ep:02d} | AUC={auc(fpr,tpr):.4f}")

    model.reset_states()
    p_base = 0
    ys, ps = [], []
    with torch.no_grad():
        for u_cpu, v_cpu, tq_cpu, y_cpu, h_cpu in test_loader:
            u = u_cpu.to(device, non_blocking=True)
            v = v_cpu.to(device, non_blocking=True)
            tq = tq_cpu.to(device, non_blocking=True)
            y  = y_cpu.to(device, non_blocking=True)

            tq_max = float(tq.max())
            new_p = torch.searchsorted(es_t, torch.tensor(tq_max, device=es_t.device)).item()
            if new_p > p_base:
                model.update(es_src[p_base:new_p], es_t[p_base:new_p])
                p_base = new_p

            logits = pred(model.embed(u, tq), model.embed(v, tq))
            probs = torch.sigmoid(logits)
            ys.extend(y.detach().cpu().tolist()); ps.extend(probs.detach().cpu().tolist())

    fpr, tpr, _ = roc_curve(ys, ps)
    print(f"[TEST] AUC={auc(fpr,tpr):.4f}")

if __name__ == "__main__":
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    run(max_hop=None)
