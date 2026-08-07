import importlib
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from absl import app, flags

import gat_eval_metrics as M

FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint", None, "Path to <tag>_best.pt PCBGAT checkpoint")
flags.DEFINE_string("graphs", None, "Path to the graphs .pt file to evaluate (test.pt / valid.pt)")
flags.DEFINE_string("out_dir", "eval_out", "Directory for JSON report, confusion CSVs, plots")
flags.DEFINE_string("class_yaml", None,
                     "YOLO data.yaml for class names (same file train_gat uses). "
                     "Defaults to the built-in 23-class canonical taxonomy if omitted.")
flags.DEFINE_string("model_module", "train_gat", "Importable module defining the PCBGAT class")
flags.DEFINE_string("model_class", "PCBGAT", "Class name inside --model_module")
flags.DEFINE_list("exclude_classes", ["resistor", "capacitor"],
                   "Class names excluded when computing acc_nondominant")
flags.DEFINE_string("device", "cuda" if torch.cuda.is_available() else "cpu", "Torch device")
flags.DEFINE_boolean("plots", True, "Save confusion-matrix and yolo-vs-gat plots")

flags.mark_flag_as_required("checkpoint")
flags.mark_flag_as_required("graphs")

# Canonical 23-class detector taxonomy -- used as the default name source so
# plots/reports always show real component names, never "class_N".
CANON = ["battery", "button", "buzzer", "capacitor", "clock", "connector", "diode",
         "display", "fuse", "heatsink", "ic", "inductor", "led", "pads", "pins",
         "potentiometer", "relay", "resistor", "switch", "transducer", "transformer",
         "transistor", "unknown"]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10, "axes.titlesize": 12,
    "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
})
C_YOLO, C_GAT = "#8d99ae", "#2b6cb0"
C_FIX, C_BREAK = "#2a9d8f", "#d1495b"


# ---------------------------------------------------------------------------
# Model loading / inference
# ---------------------------------------------------------------------------

def infer_arch(state):
    """Recover hidden / heads / layers from saved weights (checkpoint stores
    only in_dim / edge_dim / nc). inp.0 is the Linear inside the input
    Sequential; GATv2Conv.att has shape (1, heads, out_per_head)."""
    hidden = state["inp.0.weight"].shape[0]
    layers = len({int(k.split(".")[1]) for k in state if k.startswith("convs.")})
    heads = state["convs.0.att"].shape[1]
    return hidden, heads, layers


def load_model(checkpoint_path, model_module, model_class, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("state", ckpt.get("state_dict"))
    if state is None:
        raise KeyError("checkpoint has neither 'state' nor 'state_dict'")
    in_dim, edge_dim, nc = ckpt["in_dim"], ckpt["edge_dim"], ckpt["nc"]
    hidden, heads, layers = infer_arch(state)

    PCBGAT = getattr(importlib.import_module(model_module), model_class)
    model = PCBGAT(in_dim, nc, edge_dim, hidden=hidden, heads=heads, layers=layers)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, nc, {"in_dim": in_dim, "edge_dim": edge_dim, "epoch": ckpt.get("epoch"),
                        "hidden": hidden, "heads": heads, "layers": layers}


def build_data(graph, device):
    from torch_geometric.data import Data
    d = Data(x=graph["x"].float(), edge_index=graph["edge_index"].long(),
              edge_attr=graph["edge_attr"].float())
    d.yolo_probs = graph["yolo_probs"].float()
    return d.to(device)


@torch.no_grad()
def run_inference(model, graphs, device):
    all_true, all_gat, all_yolo = [], [], []
    for g in graphs:
        data = build_data(g, device)
        logits = model(data)
        all_gat.append(logits.argmax(dim=1).cpu().numpy())
        all_yolo.append(g["yolo_probs"].cpu().numpy().argmax(axis=1))
        all_true.append(g["y"].cpu().numpy())
    return (np.concatenate(all_true), np.concatenate(all_gat), np.concatenate(all_yolo))


def read_class_names(class_yaml, graphs_blob, nc):
    """Real class names, in priority order: graphs blob -> data.yaml -> built-in
    CANON taxonomy. Never falls back to 'class_N' as long as nc <= len(CANON)."""
    names = (graphs_blob.get("names") or graphs_blob.get("class_names")) if graphs_blob else None
    if not names and class_yaml:
        try:
            import yaml
            n = yaml.safe_load(Path(class_yaml).read_text()).get("names")
            names = ([n[i] for i in sorted(n)] if isinstance(n, dict)
                      else list(n) if isinstance(n, list) else None)
        except Exception:
            names = None
    if not names and nc <= len(CANON):
        names = CANON
    if not names or len(names) < nc:
        names = list(names or []) + [f"class_{i}" for i in range(len(names or []), nc)]
    return list(names)[:nc]


# ---------------------------------------------------------------------------
# Plots -- visual style matched to train_gat.py's plot_confusion / plot_yolo_vs_gat
# ---------------------------------------------------------------------------

def _annotated_confusion(y_true, y_pred, names, normalize, title_suffix, out_path,
                          model_name="GAT (edge-conditioned)", split="Test"):
    """
    Full (n x n) confusion matrix over ALL classes (foreground + background,
    background = last name). Axes: TRUE on x (bottom), PREDICTED on y (left).

    normalize='recall'    -> normalize each column (true class); diagonal = recall
                              (this is train_gat.py's plot_confusion, reproduced exactly)
    normalize='precision' -> normalize each row (predicted class); diagonal = precision
    """
    n = len(names)
    labels = list(range(n))
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
    Mt = cm.T  # rows=predicted, cols=true
    counts = Mt.copy()

    if normalize == "recall":
        denom = Mt.sum(0, keepdims=True)   # per true-class column total
    elif normalize == "precision":
        denom = Mt.sum(1, keepdims=True)   # per predicted-class row total
    else:
        raise ValueError("normalize must be 'recall' or 'precision'")
    Mn = np.divide(Mt, denom, out=np.zeros_like(Mt), where=denom > 0)

    bg = n - 1
    fg = y_true != bg
    acc = accuracy_score(y_true[fg], y_pred[fg]) if fg.any() else 0.0
    present = M.present_classes(y_true, bg)  # bg == nc here (n = nc+1 names incl. background)
    mf1 = (f1_score(y_true[fg], y_pred[fg], labels=present, average="macro", zero_division=0)
           if present else 0.0)

    side = max(9, 0.52 * n)
    fig, ax = plt.subplots(figsize=(side, side * 0.92))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#f7f7f7")
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
        ax.add_patch(Rectangle((k - .5, k - .5), 1, 1, fill=False, edgecolor="#d1495b", lw=1.4))
    ax.axhline(bg - .5, color="#888", lw=1, ls="--")
    ax.axvline(bg - .5, color="#888", lw=1, ls="--")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("True class", fontweight="bold")
    ax.set_ylabel("Predicted class", fontweight="bold")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.suptitle("Node-Classification Confusion Matrix", fontsize=15, fontweight="bold", y=0.985)
    ax.set_title(f"{model_name}   \u00b7   {split}   \u00b7   N = {len(y_true):,} nodes\n"
                 f"Foreground accuracy {acc*100:.1f}%   \u00b7   macro-F1 {mf1:.3f}\n{title_suffix}",
                 fontsize=9.5, color="#333", pad=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Proportion", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] confusion ({normalize}) -> {out_path}")


def plot_yolo_vs_gat(y_true, gat_pred, yolo_pred, names, nc, out_path, split="Test"):
    """Reproduces train_gat.py's plot_yolo_vs_gat: per-class accuracy bars
    (YOLO vs GAT) + fixed/broken/net edit counts + FP-rejection note."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [2.4, 1]})
    present = [c for c in range(nc) if (y_true == c).sum() > 0]
    present.sort(key=lambda c: (y_true == c).sum(), reverse=True)
    yo, ga, lab = [], [], []
    for c in present:
        m = y_true == c
        yo.append((yolo_pred[m] == c).mean())
        ga.append((gat_pred[m] == c).mean())
        lab.append(f"{names[c]} ({int(m.sum())})")
    ypos = np.arange(len(present))
    hgt = 0.4
    a1.barh(ypos + hgt / 2, yo, hgt, color=C_YOLO, label="YOLO")
    a1.barh(ypos - hgt / 2, ga, hgt, color=C_GAT, label="YOLO+GAT")
    a1.set_yticks(ypos); a1.set_yticklabels(lab, fontsize=8)
    a1.invert_yaxis()
    a1.set_xlabel("accuracy (correct / support)")
    a1.set_xlim(0, 1)
    a1.set_title(f"Per-class accuracy on {split.lower()}", fontweight="bold")
    a1.legend(frameon=False, loc="lower right")
    a1.grid(axis="y", alpha=0)

    fg = y_true < nc
    yt, gp, yp = y_true[fg], gat_pred[fg], yolo_pred[fg]
    fixed = int(((yp != yt) & (gp == yt)).sum())
    broke = int(((yp == yt) & (gp != yt)).sum())
    net = fixed - broke
    a2.bar(["fixed", "broken", "net"], [fixed, broke, net], color=[C_FIX, C_BREAK, C_GAT])
    for i, v in enumerate([fixed, broke, net]):
        a2.text(i, v + (max(fixed, broke, 1) * 0.01), f"{v:+d}" if i == 2 else f"{v}",
                 ha="center", va="bottom", fontweight="bold", fontsize=10)
    a2.set_title("GAT edits to YOLO\n(foreground nodes)", fontweight="bold")
    a2.set_ylabel("node count")
    a2.grid(axis="x", alpha=0)

    bgm = y_true == nc
    if bgm.sum() > 0:
        rej = float((gat_pred[bgm] == nc).mean())
        a2.text(0.5, -0.22, f"False-positive rejection: GAT flags {rej*100:.0f}% of "
                f"{int(bgm.sum())} background nodes\n(YOLO: 0% \u2014 it has no background class)",
                transform=a2.transAxes, ha="center", va="top", fontsize=8.5, color="#333")

    yo_acc = (yolo_pred[fg] == yt).mean()
    ga_acc = (gat_pred[fg] == yt).mean()
    fig.suptitle(f"YOLO vs YOLO+GAT on {split.lower()}   \u00b7   foreground accuracy "
                 f"{yo_acc*100:.1f}% \u2192 {ga_acc*100:.1f}%  ({(ga_acc-yo_acc)*100:+.1f} pts)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] yolo-vs-gat -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv):
    del argv
    device = FLAGS.device
    out_dir = Path(FLAGS.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, nc, meta = load_model(FLAGS.checkpoint, FLAGS.model_module, FLAGS.model_class, device)
    print(f"Loaded {FLAGS.model_class}: in_dim={meta['in_dim']}, edge_dim={meta['edge_dim']}, "
          f"nc={nc}, hidden={meta['hidden']}, heads={meta['heads']}, layers={meta['layers']}, "
          f"epoch={meta['epoch']}")

    blob = torch.load(FLAGS.graphs, map_location="cpu", weights_only=False)
    graphs = blob["graphs"]
    if blob.get("nc") is not None and blob["nc"] != nc:
        print(f"WARNING: graphs nc={blob['nc']} != checkpoint nc={nc}")
    class_names = read_class_names(FLAGS.class_yaml, blob, nc)
    names_with_bg = class_names + ["background"]
    print(f"Evaluating {len(graphs)} graphs, {nc} foreground classes: {class_names}")

    y_true, y_gat, y_yolo = run_inference(model, graphs, device)
    print(f"Total nodes: {len(y_true)} "
          f"(foreground {int((y_true < nc).sum())}, background {int((y_true == nc).sum())})")

    gat = M.basic_scores(y_true, y_gat, nc)
    yolo = M.basic_scores(y_true, y_yolo, nc)
    nondom = M.accuracy_excluding_dominant(y_true, y_gat, nc, class_names, FLAGS.exclude_classes)
    per_class = M.per_class_prf(y_true, y_gat, nc)
    macro_weighted = M.macro_weighted_scores(y_true, y_gat, nc)
    edits = M.node_edit_counts(y_true, y_gat, y_yolo, nc)
    fpr = M.false_positive_rejection_rate(y_true, y_gat, nc)

    present_ids = M.present_classes(y_true, nc)
    present_names = [class_names[c] for c in present_ids]
    absent_names = [class_names[c] for c in range(nc) if c not in present_ids]

    report = {
        "checkpoint": FLAGS.checkpoint, "graphs": FLAGS.graphs, "nc": nc,
        "class_names": class_names, "n_graphs": len(graphs), "n_nodes_total": int(len(y_true)),
        "classes_present_in_test_set": present_names,
        "classes_absent_from_test_set": absent_names,
        "gat": gat, "yolo_baseline": yolo,
        "accuracy_excluding_dominant": {**nondom, "excluded": FLAGS.exclude_classes},
        "macro_vs_weighted": macro_weighted, "node_edits": edits,
        "false_positive_rejection": fpr,
        "per_class": {class_names[c]: per_class[c] for c in range(nc)},
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    cm_counts = M.confusion_counts(y_true, y_gat, nc, include_background_pred=True)
    np.savetxt(out_dir / "confusion_counts.csv", cm_counts, fmt="%d", delimiter=",")
    np.savetxt(out_dir / "confusion_recall.csv", M.normalize_confusion(cm_counts, "recall"),
               fmt="%.4f", delimiter=",")
    np.savetxt(out_dir / "confusion_precision.csv", M.normalize_confusion(cm_counts, "precision"),
               fmt="%.4f", delimiter=",")

    if FLAGS.plots:
        _annotated_confusion(y_true, y_gat, names_with_bg, "recall",
                              "Columns normalized per true class  (diagonal = recall)",
                              out_dir / "confusion_recall.png")
        _annotated_confusion(y_true, y_gat, names_with_bg, "precision",
                              "Rows normalized per predicted class  (diagonal = precision)",
                              out_dir / "confusion_precision.png")
        plot_yolo_vs_gat(y_true, y_gat, y_yolo, class_names, nc, out_dir / "yolo_vs_gat.png")

    print("\n=== Aggregate (foreground nodes) ===")
    print(f"  GAT   acc={gat['accuracy']:.4f}  macroF1={gat['macro_f1']:.4f}  (n={gat['n']}, "
          f"{gat['n_classes_present']}/{gat['n_classes_total']} classes present)")
    print(f"  YOLO  acc={yolo['accuracy']:.4f}  macroF1={yolo['macro_f1']:.4f}")
    if absent_names:
        print(f"  NOT in this test set (excluded from macro averages, not counted as 0): "
              f"{absent_names}")
    print(f"  acc excl. {FLAGS.exclude_classes}: {nondom['accuracy']:.4f} (n={nondom['n']})")
    print(f"  node edits: fixed={edits['fixed']} broken={edits['broken']} net={edits['net']}")
    print(f"  FP rejection rate: {fpr['rate']:.4f} (n={fpr['n']})")
    mu, mw = macro_weighted["macro_unweighted"], macro_weighted["weighted_by_support"]
    print(f"\n=== Macro (unweighted, over {len(present_ids)} present classes) vs support-weighted ===")
    print(f"  unweighted  P={mu['precision']:.4f} R={mu['recall']:.4f} F1={mu['f1']:.4f}")
    print(f"  weighted    P={mw['precision']:.4f} R={mw['recall']:.4f} F1={mw['f1']:.4f}")
    print("\n=== Per-class precision / recall / F1 / support ('--' = undefined, not 0) ===")

    def fmt(v):
        return f"{v:.3f}" if v is not None else "  --"

    for c in range(nc):
        pc = per_class[c]
        print(f"  {class_names[c]:>14}  P={fmt(pc['precision'])}  R={fmt(pc['recall'])}  "
              f"F1={fmt(pc['f1'])}  n={pc['support']}")
    print(f"\nWrote report + confusion CSVs{' + plots' if FLAGS.plots else ''} to {out_dir}/")


if __name__ == "__main__":
    app.run(main)