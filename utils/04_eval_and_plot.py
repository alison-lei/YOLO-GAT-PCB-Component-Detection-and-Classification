"""
04_eval_and_plot.py — Evaluate a trained GAT on a graph split and render a
publication-quality confusion matrix, plus a per-class precision/recall/F1 table.

Layout matches your existing plots: y-axis = Predicted, x-axis = True,
columns normalized per true class (so the diagonal reads as recall).

--------------------------------------------------------------------------
PREREQUISITE: 03_train_gat.py does not currently save the trained model, so
there is nothing to load here yet. Add ONE line to 03_train_gat.py, right
after `model.load_state_dict(best_state)`:

    torch.save({"state": best_state, "mode": a.mode,
                "in_dim": tr[0].x.shape[1], "nc": nc},
               outdir / f"{tag}_model.pt")
--------------------------------------------------------------------------

Usage:
  # evaluate the edge model on the held-out TEST graphs
  python 04_eval_and_plot.py --ckpt results/gat_edge_model.pt \
      --graphs graphs/test.pt --names data.yaml \
      --model-name "GAT (edge-conditioned)" --split "Test - FPIC (cross-dataset)" \
      --out results/gat_edge_test_cm.png

  # or compare against raw YOLO on the same graphs
  python 04_eval_and_plot.py --graphs graphs/test.pt --names data.yaml --yolo-baseline \
      --model-name "YOLO (per-box argmax)" --split "Test - FPIC" \
      --out results/yolo_test_cm.png
"""
import argparse
import importlib.util as ilu
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             accuracy_score, f1_score)

HERE = Path(__file__).parent


# --------------------------------------------------------------------- helpers
def _load_module(name, filename):
    spec = ilu.spec_from_file_location(name, str(HERE / filename))
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_names(path, nc):
    """Pull class names from a YOLO data.yaml; fall back to indices.
    Always appends the pipeline's 'background' sink as the final (nc-th) class."""
    names = None
    if path:
        try:
            import yaml
            d = yaml.safe_load(Path(path).read_text())
            n = d.get("names")
            if isinstance(n, dict):
                names = [n[i] for i in sorted(n)]
            elif isinstance(n, list):
                names = list(n)
        except Exception as e:
            print(f"[names] could not read {path}: {e}")
    if not names:
        names = [f"class_{i}" for i in range(nc)]
    return names + ["background"]


# ----------------------------------------------------------------- the plotter
def plot_confusion(y_true, y_pred, class_names, model_name, split, out,
                   cmap="Blues"):
    """Column-normalized confusion matrix (Predicted x True). Diagonal = recall."""
    n = len(class_names)
    labels = list(range(n))

    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(float)   # [true, pred]
    M = cm.T                                                             # [pred, true]
    counts = M.copy()

    col = M.sum(axis=0, keepdims=True)
    with np.errstate(all="ignore"):
        Mn = np.divide(M, col, out=np.zeros_like(M), where=col > 0)

    # headline metrics (foreground only = everything except the background class)
    bg = n - 1
    fg = y_true != bg
    acc = accuracy_score(y_true[fg], y_pred[fg]) if fg.any() else 0.0
    mf1 = f1_score(y_true[fg], y_pred[fg], labels=labels[:-1],
                   average="macro", zero_division=0)

    # ---- figure ----
    side = max(9, 0.52 * n)
    fig, ax = plt.subplots(figsize=(side, side * 0.92), dpi=150)

    masked = np.ma.masked_where(Mn == 0, Mn)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#f7f7f7")                       # empty cells: soft grey
    im = ax.imshow(masked, cmap=cmap_obj, vmin=0, vmax=1, aspect="equal")

    # annotations
    for i in range(n):
        for j in range(n):
            v = Mn[i, j]
            if v <= 0:
                continue
            on_diag = (i == j)
            txt = f"{v:.2f}"
            ax.text(j, i - (0.13 if on_diag else 0), txt,
                    ha="center", va="center",
                    fontsize=8.5 if on_diag else 7.5,
                    fontweight="bold" if on_diag else "normal",
                    color="white" if v > 0.55 else "#1a1a1a")
            if on_diag:                                     # raw count under diagonal
                ax.text(j, i + 0.22, f"n={int(counts[i, j])}",
                        ha="center", va="center", fontsize=6,
                        color="white" if v > 0.55 else "#555")

    # highlight the diagonal
    for k in range(n):
        ax.add_patch(Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False,
                               edgecolor="#d1495b", lw=1.4))

    # separator before the background row/col so the FP sink is visually distinct
    ax.axhline(bg - 0.5, color="#888", lw=1.0, ls="--")
    ax.axvline(bg - 0.5, color="#888", lw=1.0, ls="--")

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("True class", fontsize=11, fontweight="bold")
    ax.set_ylabel("Predicted class", fontsize=11, fontweight="bold")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    # professional multi-line title
    fig.suptitle("Node-Classification Confusion Matrix",
                 fontsize=15, fontweight="bold", y=0.985)
    ax.set_title(
        f"{model_name}   ·   {split}   ·   N = {len(y_true):,} nodes\n"
        f"Foreground accuracy {acc*100:.1f}%   ·   macro-F1 {mf1:.3f}\n"
        f"Columns normalized per true class  (diagonal = recall)",
        fontsize=9.5, color="#333", pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Proportion of true class", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] -> {out}   (fg acc {acc*100:.1f}%  macro-F1 {mf1:.3f})")
    return acc, mf1


def print_per_class(y_true, y_pred, class_names, out_csv=None):
    n = len(class_names)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n)), zero_division=0)
    print(f"\n{'class':<16}{'prec':>7}{'recall':>8}{'f1':>7}{'support':>9}")
    print("-" * 47)
    rows = []
    for i, name in enumerate(class_names):
        print(f"{name:<16}{p[i]:>7.3f}{r[i]:>8.3f}{f[i]:>7.3f}{int(s[i]):>9}")
        rows.append((name, p[i], r[i], f[i], int(s[i])))
    if out_csv:
        import csv
        with open(out_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["class", "precision", "recall", "f1", "support"])
            w.writerows(rows)
        print(f"[csv] -> {out_csv}")


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", required=True, help="graph .pt to evaluate on")
    ap.add_argument("--ckpt", help="trained GAT checkpoint from patched 03_train_gat.py")
    ap.add_argument("--yolo-baseline", action="store_true",
                    help="ignore ckpt; score raw YOLO argmax instead of the GAT")
    ap.add_argument("--names", help="YOLO data.yaml for class names")
    ap.add_argument("--model-name", default="GAT")
    ap.add_argument("--split", default="Test")
    ap.add_argument("--out", default="results/confusion.png")
    a = ap.parse_args()

    import torch
    from torch_geometric.loader import DataLoader
    tm = _load_module("train_gat", "03_train_gat.py")
    ds, nc = tm.to_pyg(a.graphs)
    class_names = read_names(a.names, nc)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if a.yolo_baseline:
        y_true = torch.cat([d.y for d in ds]).numpy()
        y_pred = torch.cat([d.yolo_probs.argmax(1) for d in ds]).numpy()
    else:
        assert a.ckpt, "pass --ckpt (or use --yolo-baseline)"
        ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
        model = tm.PCBGAT(ck["in_dim"], ck["nc"], ck["mode"]).to(dev)
        model.load_state_dict(ck["state"]); model.eval()
        loader = DataLoader(ds, batch_size=8)
        yt, yp = [], []
        with torch.no_grad():
            for b in loader:
                b = b.to(dev)
                logits, _ = model(b)
                yt.append(b.y.cpu()); yp.append(logits.argmax(1).cpu())
        y_true = torch.cat(yt).numpy(); y_pred = torch.cat(yp).numpy()

    plot_confusion(y_true, y_pred, class_names, a.model_name, a.split, a.out)
    print_per_class(y_true, y_pred, class_names,
                    out_csv=str(Path(a.out).with_suffix(".per_class.csv")))


if __name__ == "__main__":
    main()
