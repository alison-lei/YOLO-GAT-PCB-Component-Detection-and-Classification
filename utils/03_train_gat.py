"""
03_train_gat.py (minimal) — GATv2 that corrects YOLO's per-node class using
graph context. Three ablation modes, gated residual head, nothing else.

Modes (an ablation ladder — deliberately different models):
  mlp       : classify each node alone. No edges, no attention.
  gat_node  : GAT attention over neighbours, geometry IGNORED. ~= Kuo et al.
  gat_edge  : GAT attention CONDITIONED on edge geometry (edge_dim=8). Ours.

--noise F : corrupt fraction F of TRAIN node prob-vectors (fixes overfit where
            YOLO train predictions are too clean; val untouched).

Usage:
  python 03_train_gat.py --train graphs/gat.pt --val graphs/val.pt \
      --mode gat_edge --epochs 200 --noise 0.2 --tag gat_edge
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv

EDGE_DIM = 8


class PCBGAT(nn.Module):
    def __init__(self, in_dim, nc, mode, hidden=128, heads=4, layers=3, drop=0.2):
        super().__init__()
        self.mode, self.nc, self.n_out = mode, nc, nc + 1
        self.inp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ELU())
        use_edge = (mode == "gat_edge")
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        for _ in range(layers):
            if mode == "mlp":
                self.convs.append(nn.Sequential(nn.Linear(hidden, hidden), nn.ELU()))
            else:
                self.convs.append(GATv2Conv(hidden, hidden // heads, heads=heads,
                    concat=True, edge_dim=EDGE_DIM if use_edge else None, dropout=drop))
            self.norms.append(nn.LayerNorm(hidden))
        self.delta = nn.Linear(hidden, self.n_out)
        self.gate = nn.Linear(hidden, 1)
        self.drop = drop

    def forward(self, d):
        h = self.inp(d.x)
        for conv, norm in zip(self.convs, self.norms):
            hin = h
            if self.mode == "mlp":
                h = conv(h)
            elif self.mode == "gat_edge":
                h = conv(h, d.edge_index, d.edge_attr)
            else:
                h = conv(h, d.edge_index)
            h = norm(F.elu(h)) + hin
            h = F.dropout(h, self.drop, self.training)
        p = d.yolo_probs.clamp(1e-4, 1 - 1e-4)
        prior = torch.cat([torch.log(p / (1 - p)),
                           torch.zeros(len(p), 1, device=p.device)], 1)
        g = torch.sigmoid(self.gate(h))
        return prior + g * self.delta(h), g


def to_pyg(path):
    blob = torch.load(path, weights_only=False)
    out = [Data(x=g["x"], edge_index=g["edge_index"], edge_attr=g["edge_attr"],
                y=g["y"], yolo_probs=g["yolo_probs"]) for g in blob["graphs"]]
    return out, blob["nc"]


def add_noise(ds, nc, frac, seed=0):
    """Corrupt `frac` of foreground nodes to a random wrong class so train
    inputs are as error-prone as YOLO really is on unseen boards."""
    rng = np.random.default_rng(seed)
    noisy = []
    for d in ds:
        x = d.x.clone(); yp = d.yolo_probs.clone()
        fg = (d.y < nc).nonzero(as_tuple=True)[0].tolist()
        for i in fg:
            if rng.random() < frac:
                wrong = int(rng.integers(nc))
                oneh = torch.full((nc,), 0.2 / nc)
                oneh[wrong] += 0.8
                yp[i] = oneh
                x[i, :nc] = oneh
        noisy.append(Data(x=x, edge_index=d.edge_index, edge_attr=d.edge_attr,
                          y=d.y, yolo_probs=yp))
    return noisy


def class_weights(ds, n_out):
    cnt = torch.zeros(n_out)
    for d in ds:
        cnt += torch.bincount(d.y, minlength=n_out).float()
    w = 1.0 / torch.sqrt(cnt.clamp(min=1))
    return w / w.mean()


@torch.no_grad()
def evaluate(model, loader, dev, nc):
    model.eval(); Y, P, B = [], [], []
    for b in loader:
        b = b.to(dev)
        logits, _ = model(b)
        Y.append(b.y.cpu()); P.append(logits.argmax(1).cpu())
        B.append(b.yolo_probs.argmax(1).cpu())
    Y, P, B = torch.cat(Y), torch.cat(P), torch.cat(B)
    fg = Y < nc
    return dict(
        gat_acc=float((P[fg] == Y[fg]).float().mean()),
        yolo_acc=float((B[fg] == Y[fg]).float().mean()),
        gat_f1=float(f1_score(Y[fg], P[fg], average="macro", zero_division=0)),
        yolo_f1=float(f1_score(Y[fg], B[fg], average="macro", zero_division=0)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--mode", default="gat_edge",
                    choices=["mlp", "gat_node", "gat_edge"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    tag = a.tag or a.mode
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr, nc = to_pyg(a.train)
    va, _ = to_pyg(a.val)
    if a.noise > 0:
        tr = add_noise(tr, nc, a.noise)
        print(f"[noise] corrupted {a.noise:.0%} of train foreground nodes")

    ltr = DataLoader(tr, batch_size=a.bs, shuffle=True)
    lva = DataLoader(va, batch_size=a.bs)
    w = class_weights(tr, nc + 1).to(dev)

    model = PCBGAT(tr[0].x.shape[1], nc, a.mode).to(dev)
    print(f"[{tag}] {len(tr)}/{len(va)} graphs | nc={nc} | "
          f"params {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    best, best_state = -1, None

    for ep in range(a.epochs):
        model.train(); lsum = 0.0
        for b in ltr:
            b = b.to(dev)
            logits, _ = model(b)
            loss = F.cross_entropy(logits, b.y, weight=w)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); lsum += float(loss.detach())
        sched.step()
        m = evaluate(model, lva, dev, nc)
        if m["gat_f1"] > best:
            best = m["gat_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 20 == 0 or ep == a.epochs - 1:
            print(f"  ep{ep:3d} loss {lsum/len(ltr):.3f} | "
                  f"val acc {m['gat_acc']*100:5.2f} (yolo {m['yolo_acc']*100:5.2f}) | "
                  f"val F1 {m['gat_f1']*100:5.2f}")

    model.load_state_dict(best_state)
    m = evaluate(model, lva, dev, nc)
    print(f"\n=== {tag} (best) ===")
    print(f"  YOLO : acc {m['yolo_acc']*100:5.2f}  F1 {m['yolo_f1']*100:5.2f}")
    print(f"  GAT  : acc {m['gat_acc']*100:5.2f}  F1 {m['gat_f1']*100:5.2f}")
    print(f"  delta: {(m['gat_acc']-m['yolo_acc'])*100:+5.2f} acc  "
          f"{(m['gat_f1']-m['yolo_f1'])*100:+5.2f} F1")
    (outdir / f"{tag}_results.json").write_text(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()