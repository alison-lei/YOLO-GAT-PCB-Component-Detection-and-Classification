"""
train_gat.py — GATv2 (edge-conditioned) that corrects YOLO's per-node class using
graph context. Single model, no ablation modes.

The head is a GATED RESIDUAL over YOLO: it starts from the logit of YOLO's own
class vector (a "prior") and only nudges it where graph context justifies, so the
GAT augments rather than replaces the detector.

Produces at the end (in --out, default `results/`):
  * <tag>_confusion.png        red/blue node-classification matrix (val, best model)
  * <tag>_curves.png           train-vs-val accuracy and loss over epochs
  * <tag>_yolo_vs_gat.png      how the GAT changes YOLO's validation predictions
  * <tag>_best.pt              best checkpoint (highest validation accuracy)
  * ckpts/<tag>_epoch*.pt      periodic checkpoints
  * <tag>_summary.json         final metrics + best epoch

Usage:
  python train_gat.py --train graphs/train.pt --val graphs/valid.pt \
      --names data.yaml --epochs 200 --noise 0.2 --ckpt-every 25 --tag gat_edge
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import (confusion_matrix, f1_score, accuracy_score,
                             precision_recall_fscore_support)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv

EDGE_DIM = 8

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
})
C_YOLO, C_GAT = "#8d99ae", "#2b6cb0"       # muted steel vs strong blue
C_FIX, C_BREAK = "#2a9d8f", "#d1495b"      # green / red


# ==================================================================== model
class PCBGAT(nn.Module):
    """Edge-conditioned GATv2 with a gated residual over YOLO's prior."""
    def __init__(self, in_dim, nc, hidden=128, heads=4, layers=3, drop=0.2):
        super().__init__()
        self.nc, self.n_out = nc, nc + 1
        self.inp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ELU())
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        for _ in range(layers):
            self.convs.append(GATv2Conv(hidden, hidden // heads, heads=heads,
                                        concat=True, edge_dim=EDGE_DIM, dropout=drop))
            self.norms.append(nn.LayerNorm(hidden))
        self.delta = nn.Linear(hidden, self.n_out)
        self.gate = nn.Linear(hidden, 1)
        self.drop = drop

    def forward(self, d):
        h = self.inp(d.x)
        for conv, norm in zip(self.convs, self.norms):
            hin = h
            h = conv(h, d.edge_index, d.edge_attr)
            h = norm(F.elu(h)) + hin
            h = F.dropout(h, self.drop, self.training)
        p = d.yolo_probs.clamp(1e-4, 1 - 1e-4)
        prior = torch.cat([torch.log(p / (1 - p)),
                           torch.zeros(len(p), 1, device=p.device)], 1)
        g = torch.sigmoid(self.gate(h))
        return prior + g * self.delta(h), g


# ==================================================================== data
def to_pyg(path):
    blob = torch.load(path, weights_only=False)
    out = [Data(x=g["x"], edge_index=g["edge_index"], edge_attr=g["edge_attr"],
                y=g["y"], yolo_probs=g["yolo_probs"]) for g in blob["graphs"]]
    return out, blob["nc"]


def add_noise(ds, nc, frac, seed=50):
    """Corrupt `frac` of foreground nodes to a random wrong class so train inputs
    are as error-prone as YOLO really is on unseen boards (val untouched)."""
    rng = np.random.default_rng(seed)
    noisy = []
    for d in ds:
        x = d.x.clone(); yp = d.yolo_probs.clone()
        fg = (d.y < nc).nonzero(as_tuple=True)[0].tolist()
        for i in fg:
            if rng.random() < frac:
                wrong = int(rng.integers(nc))
                oneh = torch.full((nc,), 0.2 / nc); oneh[wrong] += 0.8
                yp[i] = oneh; x[i, :nc] = oneh
        noisy.append(Data(x=x, edge_index=d.edge_index, edge_attr=d.edge_attr,
                          y=d.y, yolo_probs=yp))
    return noisy


def class_weights(ds, n_out):
    cnt = torch.zeros(n_out)
    for d in ds:
        cnt += torch.bincount(d.y, minlength=n_out).float()
    w = 1.0 / torch.sqrt(cnt.clamp(min=1))
    return w / w.mean()


def read_names(path, nc):
    names = None
    if path:
        try:
            import yaml
            d = yaml.safe_load(Path(path).read_text())
            n = d.get("names")
            names = ([n[i] for i in sorted(n)] if isinstance(n, dict)
                     else list(n) if isinstance(n, list) else None)
        except Exception as e:
            print(f"[names] could not read {path}: {e}")
    if not names:
        names = [f"class_{i}" for i in range(nc)]
    return names + ["background"]


# ============================================================ eval + collect
@torch.no_grad()
def evaluate(model, loader, dev, nc, w):
    """Return metrics dict + concatenated (y_true, gat_pred, yolo_pred)."""
    model.eval()
    Y, P, B, loss_sum, n = [], [], [], 0.0, 0
    for b in loader:
        b = b.to(dev)
        logits, _ = model(b)
        loss_sum += float(F.cross_entropy(logits, b.y, weight=w, reduction="sum"))
        n += len(b.y)
        Y.append(b.y.cpu()); P.append(logits.argmax(1).cpu())
        B.append(b.yolo_probs.argmax(1).cpu())
    Y, P, B = torch.cat(Y).numpy(), torch.cat(P).numpy(), torch.cat(B).numpy()
    fg = Y < nc
    m = dict(
        loss=loss_sum / max(1, n),
        acc=float((P[fg] == Y[fg]).mean()) if fg.any() else 0.0,
        f1=float(f1_score(Y[fg], P[fg], average="macro", zero_division=0)) if fg.any() else 0.0,
        yolo_acc=float((B[fg] == Y[fg]).mean()) if fg.any() else 0.0,
        yolo_f1=float(f1_score(Y[fg], B[fg], average="macro", zero_division=0)) if fg.any() else 0.0,
    )
    return m, (Y, P, B)


# ==================================================================== plots
def plot_confusion(y_true, y_pred, names, model_name, split, out):
    n = len(names); labels = list(range(n))
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
    M = cm.T; counts = M.copy()
    col = M.sum(0, keepdims=True)
    Mn = np.divide(M, col, out=np.zeros_like(M), where=col > 0)

    bg = n - 1; fg = y_true != bg
    acc = accuracy_score(y_true[fg], y_pred[fg]) if fg.any() else 0.0
    mf1 = f1_score(y_true[fg], y_pred[fg], labels=labels[:-1],
                   average="macro", zero_division=0)

    side = max(9, 0.52 * n)
    fig, ax = plt.subplots(figsize=(side, side * 0.92))
    cmap = plt.get_cmap("Blues").copy(); cmap.set_bad("#f7f7f7")
    im = ax.imshow(np.ma.masked_where(Mn == 0, Mn), cmap=cmap, vmin=0, vmax=1)

    for i in range(n):
        for j in range(n):
            v = Mn[i, j]
            if v <= 0:
                continue
            dg = i == j
            ax.text(j, i - (0.13 if dg else 0), f"{v:.2f}", ha="center", va="center",
                    fontsize=8.5 if dg else 7.5, fontweight="bold" if dg else "normal",
                    color="white" if v > 0.55 else "#1a1a1a")
            if dg:
                ax.text(j, i + 0.22, f"n={int(counts[i, j])}", ha="center",
                        va="center", fontsize=6, color="white" if v > 0.55 else "#555")
    for k in range(n):
        ax.add_patch(Rectangle((k - .5, k - .5), 1, 1, fill=False,
                               edgecolor="#d1495b", lw=1.4))
    ax.axhline(bg - .5, color="#888", lw=1, ls="--"); ax.axvline(bg - .5, color="#888", lw=1, ls="--")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=90, fontsize=8); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("True class", fontweight="bold"); ax.set_ylabel("Predicted class", fontweight="bold")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.suptitle("Node-Classification Confusion Matrix", fontsize=15, fontweight="bold", y=0.985)
    ax.set_title(f"{model_name}   ·   {split}   ·   N = {len(y_true):,} nodes\n"
                 f"Foreground accuracy {acc*100:.1f}%   ·   macro-F1 {mf1:.3f}\n"
                 f"Columns normalized per true class  (diagonal = recall)",
                 fontsize=9.5, color="#333", pad=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Proportion of true class", fontsize=9); cb.ax.tick_params(labelsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] confusion -> {out}")


def plot_curves(hist, best_ep, out):
    ep = hist["epoch"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))

    a1.plot(ep, np.array(hist["train_acc"]) * 100, color=C_YOLO, lw=2, label="train")
    a1.plot(ep, np.array(hist["val_acc"]) * 100, color=C_GAT, lw=2, label="validation")
    a1.axvline(best_ep, color=C_FIX, ls="--", lw=1.3, label=f"best epoch ({best_ep})")
    a1.set_title("Foreground accuracy", fontweight="bold")
    a1.set_xlabel("epoch"); a1.set_ylabel("accuracy (%)"); a1.legend(frameon=False)

    a2.plot(ep, hist["train_loss"], color=C_YOLO, lw=2, label="train")
    a2.plot(ep, hist["val_loss"], color=C_GAT, lw=2, label="validation")
    a2.axvline(best_ep, color=C_FIX, ls="--", lw=1.3, label=f"best epoch ({best_ep})")
    a2.set_title("Weighted cross-entropy loss", fontweight="bold")
    a2.set_xlabel("epoch"); a2.set_ylabel("loss"); a2.legend(frameon=False)

    fig.suptitle("GAT (edge-conditioned) — Training Curves", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] curves -> {out}")


def plot_yolo_vs_gat(y_true, gat_pred, yolo_pred, names, nc, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 8),
                                 gridspec_kw={"width_ratios": [2.4, 1]})

    # --- left: per-class recall, YOLO vs GAT (foreground classes with support) ---
    present = [c for c in range(nc) if (y_true == c).sum() > 0]
    present.sort(key=lambda c: (y_true == c).sum(), reverse=True)
    yo, ga, lab = [], [], []
    for c in present:
        m = y_true == c
        yo.append((yolo_pred[m] == c).mean())
        ga.append((gat_pred[m] == c).mean())
        lab.append(f"{names[c]} ({int(m.sum())})")
    ypos = np.arange(len(present)); hgt = 0.4
    a1.barh(ypos + hgt / 2, yo, hgt, color=C_YOLO, label="YOLO")
    a1.barh(ypos - hgt / 2, ga, hgt, color=C_GAT, label="YOLO+GAT")
    a1.set_yticks(ypos); a1.set_yticklabels(lab, fontsize=8); a1.invert_yaxis()
    a1.set_xlabel("recall (correct / support)"); a1.set_xlim(0, 1)
    a1.set_title("Per-class recall on validation", fontweight="bold")
    a1.legend(frameon=False, loc="lower right"); a1.grid(axis="y", alpha=0)

    # --- right: how many predictions the GAT changed (foreground) ---
    fg = y_true < nc
    yt, gp, yp = y_true[fg], gat_pred[fg], yolo_pred[fg]
    fixed = int(((yp != yt) & (gp == yt)).sum())      # YOLO wrong -> GAT right
    broke = int(((yp == yt) & (gp != yt)).sum())      # YOLO right -> GAT wrong
    net = fixed - broke
    a2.bar(["fixed", "broken", "net"], [fixed, broke, net],
           color=[C_FIX, C_BREAK, C_GAT])
    for i, v in enumerate([fixed, broke, net]):
        a2.text(i, v + (max(fixed, broke) * 0.01), f"{v:+d}" if i == 2 else f"{v}",
                ha="center", va="bottom", fontweight="bold", fontsize=10)
    a2.set_title("GAT edits to YOLO\n(foreground nodes)", fontweight="bold")
    a2.set_ylabel("node count"); a2.grid(axis="x", alpha=0)

    # background / false-positive rejection stat (YOLO can't do this at all)
    bgm = y_true == nc
    if bgm.sum() > 0:
        rej = float((gat_pred[bgm] == nc).mean())
        a2.text(0.5, -0.22,
                f"False-positive rejection: GAT flags {rej*100:.0f}% of "
                f"{int(bgm.sum())} background nodes\n(YOLO: 0% — it has no background class)",
                transform=a2.transAxes, ha="center", va="top", fontsize=8.5, color="#333")

    yo_acc = (yolo_pred[fg] == yt).mean(); ga_acc = (gat_pred[fg] == yt).mean()
    fig.suptitle(f"YOLO vs YOLO+GAT on validation   ·   foreground accuracy "
                 f"{yo_acc*100:.1f}% → {ga_acc*100:.1f}%  ({(ga_acc-yo_acc)*100:+.1f} pts)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95]); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] yolo-vs-gat -> {out}")


# ==================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--names", help="YOLO data.yaml for class names in plots")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--tag", default="gat_edge")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    outdir = Path(a.out); (outdir / "ckpts").mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr, nc = to_pyg(a.train)
    va, _ = to_pyg(a.val)
    names = read_names(a.names, nc)
    if a.noise > 0:
        tr = add_noise(tr, nc, a.noise)
        print(f"[noise] corrupted {a.noise:.0%} of train foreground nodes")

    ltr = DataLoader(tr, batch_size=a.bs, shuffle=True)
    ltr_eval = DataLoader(tr, batch_size=a.bs)          # non-shuffled, for train metrics
    lva = DataLoader(va, batch_size=a.bs)
    w = class_weights(tr, nc + 1).to(dev)

    model = PCBGAT(tr[0].x.shape[1], nc, a.hidden, a.heads, a.layers, a.drop).to(dev)
    meta = dict(in_dim=tr[0].x.shape[1], nc=nc, hidden=a.hidden,
                heads=a.heads, layers=a.layers, mode="gat_edge")
    print(f"[{a.tag}] {len(tr)}/{len(va)} graphs | nc={nc} | "
          f"params {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

    hist = {k: [] for k in ("epoch", "train_loss", "val_loss", "train_acc", "val_acc", "val_f1")}
    best_acc, best_ep, best_state = -1.0, -1, None

    for ep in range(a.epochs):
        model.train()
        for b in ltr:
            b = b.to(dev)
            logits, _ = model(b)
            loss = F.cross_entropy(logits, b.y, weight=w)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

        mt, _ = evaluate(model, ltr_eval, dev, nc, w)
        mv, _ = evaluate(model, lva, dev, nc, w)
        hist["epoch"].append(ep)
        hist["train_loss"].append(mt["loss"]); hist["val_loss"].append(mv["loss"])
        hist["train_acc"].append(mt["acc"]);   hist["val_acc"].append(mv["acc"])
        hist["val_f1"].append(mv["f1"])

        if mv["acc"] > best_acc:
            best_acc, best_ep = mv["acc"], ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"state": best_state, "epoch": ep, "val_acc": best_acc, **meta},
                       outdir / f"{a.tag}_best.pt")

        if ep % a.ckpt_every == 0 or ep == a.epochs - 1:
            torch.save({"state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                        "epoch": ep, "val_acc": mv["acc"], **meta},
                       outdir / "ckpts" / f"{a.tag}_epoch{ep}.pt")
            print(f"  ep{ep:3d} | train acc {mt['acc']*100:5.2f} loss {mt['loss']:.3f} "
                  f"| val acc {mv['acc']*100:5.2f} loss {mv['loss']:.3f} F1 {mv['f1']*100:5.2f}")

    # ---- final: reload best, evaluate on val, render all plots ----
    model.load_state_dict(best_state)
    mv, (yt, gp, yp) = evaluate(model, lva, dev, nc, w)

    plot_confusion(yt, gp, names, "GAT (edge-conditioned)", "Validation",
                   outdir / f"{a.tag}_confusion.png")
    plot_curves(hist, best_ep, outdir / f"{a.tag}_curves.png")
    plot_yolo_vs_gat(yt, gp, yp, names, nc, outdir / f"{a.tag}_yolo_vs_gat.png")

    summary = dict(best_epoch=best_ep, best_val_acc=best_acc,
                   val_f1=mv["f1"], yolo_acc=mv["yolo_acc"], yolo_f1=mv["yolo_f1"],
                   gat_acc=mv["acc"], delta_acc=mv["acc"] - mv["yolo_acc"],
                   delta_f1=mv["f1"] - mv["yolo_f1"], epochs=a.epochs, noise=a.noise)
    (outdir / f"{a.tag}_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== {a.tag} (best model) ===")
    print(f"  best epoch : {best_ep}  (selected on highest validation accuracy)")
    print(f"  YOLO       : acc {mv['yolo_acc']*100:5.2f}  F1 {mv['yolo_f1']*100:5.2f}")
    print(f"  GAT        : acc {mv['acc']*100:5.2f}  F1 {mv['f1']*100:5.2f}")
    print(f"  delta      : {(mv['acc']-mv['yolo_acc'])*100:+5.2f} acc  "
          f"{(mv['f1']-mv['yolo_f1'])*100:+5.2f} F1")
    print(f"  -> {outdir}/{a.tag}_summary.json  (+ 3 plots, best.pt, ckpts/)")


if __name__ == "__main__":
    main()
