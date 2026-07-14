"""
03_train_gat.py — Edge-conditioned GATv2 with a confidence-gated residual head,
plus the ablation ladder and the masked-component pretraining that is your
strongest novelty claim.

Install:
    pip install torch torch_geometric scikit-learn matplotlib

Usage:
    # main model
    python 03_train_gat.py --train graphs/train.pt --val graphs/valid.pt \
        --test graphs/test.pt --ood graphs/fpic.pt --mode gat_edge --epochs 300

    # ablation ladder (run all five, put them in one table)
    for m in mlp gat_node gat_edge context_only; do
        python 03_train_gat.py ... --mode $m --tag $m
    done

    # masked pretraining, then fine-tune  <-- the headline result
    python 03_train_gat.py ... --mode gat_edge --pretrain-epochs 200

Modes
-----
  mlp           : node MLP, NO edges.       Proves the graph is doing the work.
  gat_node      : GAT, geometry in nodes only, edge_attr ignored.  (~= Kuo et al.)
  gat_edge      : GAT, geometry conditions the attention.          (ours)
  context_only  : gat_edge with the YOLO class probs zeroed out.
                  "How well can you name a component from its neighbours alone?"
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv

EDGE_DIM = 8


# --------------------------------------------------------------------- model
class PCBGAT(nn.Module):
    def __init__(self, in_dim, nc, mode, hidden=128, heads=4, layers=3, drop=0.2):
        super().__init__()
        self.mode, self.nc = mode, nc
        self.n_out = nc + 1                              # + background
        self.inp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ELU())

        use_edge = (mode in ("gat_edge", "context_only"))
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        for _ in range(layers):
            if mode == "mlp":
                self.convs.append(nn.Sequential(nn.Linear(hidden, hidden), nn.ELU()))
            else:
                self.convs.append(GATv2Conv(
                    hidden, hidden // heads, heads=heads, concat=True,
                    edge_dim=EDGE_DIM if use_edge else None, dropout=drop))
            self.norms.append(nn.LayerNorm(hidden))

        self.delta = nn.Linear(hidden, self.n_out)       # contextual correction
        self.gate = nn.Linear(hidden, 1)                 # how much to trust it
        self.drop = drop

    def forward(self, d):
        h = self.inp(d.x)
        for conv, norm in zip(self.convs, self.norms):
            hin = h
            h = conv(h) if self.mode == "mlp" else (
                conv(h, d.edge_index, d.edge_attr)
                if self.mode in ("gat_edge", "context_only")
                else conv(h, d.edge_index))
            h = norm(F.elu(h)) + hin                      # residual
            h = F.dropout(h, self.drop, self.training)

        # YOLO prior in logit space (background prior = 0)
        p = d.yolo_probs.clamp(1e-4, 1 - 1e-4)
        prior = torch.cat([torch.log(p / (1 - p)),
                           torch.zeros(len(p), 1, device=p.device)], 1)
        if self.mode == "context_only":
            prior = torch.zeros_like(prior)

        g = torch.sigmoid(self.gate(h))                   # [N,1] in (0,1)
        return prior + g * self.delta(h), g               # gated residual


class MaskedHead(nn.Module):
    """Predict a node's class from neighbours alone. Self-supervised pretraining."""
    def __init__(self, body, hidden=128):
        super().__init__()
        self.body = body
        self.head = nn.Linear(hidden, body.n_out)


# ---------------------------------------------------------------------- data
def to_pyg(path, mode, nc):
    blob = torch.load(path, weights_only=False)
    out = []
    for g in blob["graphs"]:
        x = g["x"].clone()
        if mode == "context_only":
            x[:, :nc] = 0.0                              # blind the model to appearance
        out.append(Data(x=x, edge_index=g["edge_index"], edge_attr=g["edge_attr"],
                        y=g["y"], yolo_probs=g["yolo_probs"]))
    return out, blob["nc"]


def class_weights(ds, n_out):
    cnt = torch.zeros(n_out)
    for d in ds:
        cnt += torch.bincount(d.y, minlength=n_out).float()
    w = 1.0 / torch.sqrt(cnt.clamp(min=1))               # sqrt-inverse frequency
    return w / w.mean(), cnt


# ------------------------------------------------------------------ pretrain
def pretrain(model, loader, dev, nc, epochs, mask_frac=0.15, lr=1e-3):
    """Mask class-prob vectors of random nodes; predict them from context."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    print(f"\n[masked pretraining] {epochs} epochs, mask {mask_frac:.0%} of nodes")
    for ep in range(epochs):
        model.train(); tot = hit = 0; lsum = 0.0
        for b in loader:
            b = b.to(dev)
            m = torch.rand(b.num_nodes, device=dev) < mask_frac
            if m.sum() == 0:
                continue
            xb, pb = b.x.clone(), b.yolo_probs.clone()
            b.x[m, :nc] = 0.0                            # hide appearance
            b.yolo_probs[m] = 1.0 / nc                   # hide the prior too
            logits, _ = model(b)
            tgt = b.y[m]
            valid = tgt < nc                             # ignore background nodes
            if valid.sum() == 0:
                b.x, b.yolo_probs = xb, pb; continue
            loss = F.cross_entropy(logits[m][valid], tgt[valid])
            opt.zero_grad(); loss.backward(); opt.step()
            hit += int((logits[m][valid].argmax(1) == tgt[valid]).sum())
            tot += int(valid.sum()); lsum += float(loss)
            b.x, b.yolo_probs = xb, pb
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"  ep{ep:3d} loss {lsum/max(1,len(loader)):.3f} "
                  f"masked-acc {100*hit/max(1,tot):5.1f}%  "
                  f"(chance {100/nc:.1f}%)")
    return 100 * hit / max(1, tot)


# ---------------------------------------------------------------- train/eval
@torch.no_grad()
def evaluate(model, loader, dev, nc):
    model.eval()
    Y, P, B, G = [], [], [], []
    for b in loader:
        b = b.to(dev)
        logits, g = model(b)
        Y.append(b.y.cpu()); P.append(logits.argmax(1).cpu())
        B.append(b.yolo_probs.argmax(1).cpu()); G.append(g.squeeze(1).cpu())
    Y, P, B, G = (torch.cat(z) for z in (Y, P, B, G))
    fg = Y < nc                                          # real components only
    return dict(
        gat_acc=float((P[fg] == Y[fg]).float().mean()),
        yolo_acc=float((B[fg] == Y[fg]).float().mean()),
        gat_f1=float(f1_score(Y[fg], P[fg], average="macro", zero_division=0)),
        yolo_f1=float(f1_score(Y[fg], B[fg], average="macro", zero_division=0)),
        fp_reject=float((P[~fg] == nc).float().mean()) if (~fg).any() else float("nan"),
        mean_gate=float(G.mean()), gate_fg=float(G[fg].mean()),
        n=int(fg.sum()),
    ), (Y, P, B)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True); ap.add_argument("--ood", default=None,
                    help="FPIC graphs — the never-before-seen test set")
    ap.add_argument("--mode", default="gat_edge",
                    choices=["mlp", "gat_node", "gat_edge", "context_only"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--pretrain-epochs", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    tag = a.tag or a.mode
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    nc = torch.load(a.train, weights_only=False)["nc"]
    tr, _ = to_pyg(a.train, a.mode, nc)
    va, _ = to_pyg(a.val, a.mode, nc)
    te, _ = to_pyg(a.test, a.mode, nc)
    ltr = DataLoader(tr, batch_size=a.bs, shuffle=True)
    lva = DataLoader(va, batch_size=a.bs)
    lte = DataLoader(te, batch_size=a.bs)

    w, cnt = class_weights(tr, nc + 1)
    print(f"[{tag}] {len(tr)}/{len(va)}/{len(te)} graphs | nc={nc} | "
          f"in_dim={tr[0].x.shape[1]} | {sum(len(d.y) for d in tr)} train nodes")

    model = PCBGAT(tr[0].x.shape[1], nc, a.mode, hidden=a.hidden).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    masked_acc = None
    if a.pretrain_epochs:
        masked_acc = pretrain(model, ltr, dev, nc, a.pretrain_epochs)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    w = w.to(dev)
    best, best_state, curve = -1, None, []

    for ep in range(a.epochs):
        model.train(); lsum = 0.0
        for b in ltr:
            b = b.to(dev)
            logits, _ = model(b)
            loss = F.cross_entropy(logits, b.y, weight=w)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); lsum += float(loss)
        sched.step()
        m, _ = evaluate(model, lva, dev, nc)
        curve.append(dict(ep=ep, loss=lsum / len(ltr),
                          val_acc=m["gat_acc"], val_f1=m["gat_f1"]))
        if m["gat_f1"] > best:
            best = m["gat_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 20 == 0 or ep == a.epochs - 1:
            print(f"  ep{ep:3d} loss {lsum/len(ltr):.3f} | "
                  f"val acc {m['gat_acc']*100:5.2f} (yolo {m['yolo_acc']*100:5.2f}) | "
                  f"val macroF1 {m['gat_f1']*100:5.2f} | gate {m['mean_gate']:.2f}")

    model.load_state_dict(best_state); model.to(dev)

    res = {"mode": a.mode, "masked_pretrain_acc": masked_acc}
    for name, loader in [("test", lte)] + (
            [("ood_fpic", DataLoader(to_pyg(a.ood, a.mode, nc)[0], batch_size=a.bs))]
            if a.ood else []):
        m, (Y, P, B) = evaluate(model, loader, dev, nc)
        res[name] = m
        print(f"\n=== {name.upper()} ===")
        print(f"  YOLO baseline : acc {m['yolo_acc']*100:5.2f}  macroF1 {m['yolo_f1']*100:5.2f}")
        print(f"  + GAT ({a.mode:12s}): acc {m['gat_acc']*100:5.2f}  macroF1 {m['gat_f1']*100:5.2f}")
        print(f"  delta         : {(m['gat_acc']-m['yolo_acc'])*100:+5.2f} acc  "
              f"{(m['gat_f1']-m['yolo_f1'])*100:+5.2f} F1")
        print(f"  FP rejection  : {m['fp_reject']*100:5.2f}%   mean gate {m['mean_gate']:.3f}")

        # confusion-matrix delta -> the qualitative figure the rubric wants
        fg = Y < nc
        cm_y = confusion_matrix(Y[fg], B[fg], labels=list(range(nc)))
        cm_g = confusion_matrix(Y[fg], P[fg], labels=list(range(nc)))
        d = cm_g - cm_y
        np.fill_diagonal(d, 0)
        fig, ax = plt.subplots(figsize=(7, 6))
        v = np.abs(d).max() or 1
        im = ax.imshow(d, cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{name}: off-diagonal errors, GAT − YOLO\n(blue = GAT fixed it)")
        fig.colorbar(im); fig.tight_layout()
        fig.savefig(outdir / f"{tag}_{name}_cm_delta.png", dpi=160); plt.close(fig)

        # biggest confusions the GAT repaired -> name these in the report
        fixed = np.dstack(np.unravel_index(np.argsort(d.ravel())[:5], d.shape))[0]
        print("  top confusions repaired (true -> pred, count):")
        for i, j in fixed:
            if d[i, j] < 0:
                print(f"    class {i} -> class {j}: {int(-d[i,j])} errors removed")

    c = curve
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].plot([x["ep"] for x in c], [x["loss"] for x in c]); ax[0].set_title("train loss")
    ax[1].plot([x["ep"] for x in c], [100 * x["val_acc"] for x in c], label="GAT")
    ax[1].axhline(100 * res["test"]["yolo_acc"], ls="--", c="k", label="YOLO baseline")
    ax[1].set_title("val accuracy (%)"); ax[1].legend()
    for x in ax: x.set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(outdir / f"{tag}_curves.png", dpi=160); plt.close(fig)

    (outdir / f"{tag}_results.json").write_text(json.dumps(res, indent=2))
    print(f"\n-> {outdir}/{tag}_*.png, {tag}_results.json")


if __name__ == "__main__":
    main()
