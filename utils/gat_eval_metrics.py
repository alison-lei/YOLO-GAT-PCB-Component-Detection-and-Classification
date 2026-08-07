import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
)


def foreground_mask(y_true, nc):
    return y_true < nc


def present_classes(y_true, nc):
    """Foreground classes that actually have >=1 TRUE example in this eval set.
    A class with zero true examples has undefined recall (nothing to detect) --
    it must be excluded from macro averages, not silently averaged in as 0."""
    m = foreground_mask(y_true, nc)
    return sorted(set(int(c) for c in y_true[m]))


def basic_scores(y_true, y_pred, nc):
    """Accuracy on all foreground nodes; macro-F1 averaged ONLY over classes
    with true support > 0 in this eval set (absent classes are excluded, not
    counted as 0)."""
    m = foreground_mask(y_true, nc)
    if m.sum() == 0:
        return {"accuracy": float("nan"), "macro_f1": float("nan"), "n": 0,
                "n_classes_present": 0, "n_classes_total": nc}
    yt, yp = y_true[m], y_pred[m]
    acc = accuracy_score(yt, yp)
    present = present_classes(y_true, nc)
    if not present:
        macro_f1 = float("nan")
    else:
        _, _, f1, _ = precision_recall_fscore_support(
            yt, yp, labels=present, average="macro", zero_division=0
        )
        macro_f1 = float(f1)
    return {"accuracy": float(acc), "macro_f1": macro_f1, "n": int(m.sum()),
            "n_classes_present": len(present), "n_classes_total": nc}


def accuracy_excluding_dominant(y_true, y_pred, nc, class_names, dominant_classes):
    """Accuracy on foreground nodes whose TRUE class is not in dominant_classes."""
    dom_ids = {class_names.index(c) for c in dominant_classes if c in class_names}
    m = foreground_mask(y_true, nc) & ~np.isin(y_true, list(dom_ids))
    if m.sum() == 0:
        return {"accuracy": float("nan"), "n": 0}
    return {"accuracy": float(accuracy_score(y_true[m], y_pred[m])), "n": int(m.sum())}


def per_class_prf(y_true, y_pred, nc):
    """
    Per-class precision, recall, F1, support on foreground nodes.
    Labels 0..nc-1 (foreground classes). Predictions may include background
    (nc); those contribute to recall denominators but form no precision row.

    A class with support == 0 (never appears as TRUE in this eval set) has
    undefined recall and F1 -- reported as None, not 0.0, so it reads as
    "not tested" rather than "model got it wrong". Precision remains a real
    number (0.0 is meaningful there: any prediction of an absent-true class
    is automatically a false positive) unless the model never predicted it
    either, in which case it's also undefined (None).
    """
    m = foreground_mask(y_true, nc)
    yt, yp = y_true[m], y_pred[m]
    p, r, f1, s = precision_recall_fscore_support(
        yt, yp, labels=list(range(nc)), average=None, zero_division=0
    )
    pred_counts = np.bincount(yp, minlength=nc + 1)[:nc]
    out = {}
    for c in range(nc):
        support = int(s[c])
        n_pred = int(pred_counts[c])
        out[c] = {
            "precision": float(p[c]) if (support > 0 or n_pred > 0) else None,
            "recall": float(r[c]) if support > 0 else None,
            "f1": float(f1[c]) if support > 0 else None,
            "support": support,
        }
    return out


def macro_weighted_scores(y_true, y_pred, nc):
    """
    Unweighted macro vs support-weighted precision/recall/F1 on foreground
    nodes. Unweighted macro is restricted to classes with true support > 0
    (see present_classes) so absent classes aren't averaged in as 0.
    Support-weighted average is mathematically self-correcting -- a class
    with 0 support already contributes 0 weight -- so it's left over the
    full label range for simplicity; the result is identical either way.
    """
    m = foreground_mask(y_true, nc)
    yt, yp = y_true[m], y_pred[m]
    present = present_classes(y_true, nc)
    out = {}
    for name, avg, labels in [("macro_unweighted", "macro", present),
                               ("weighted_by_support", "weighted", list(range(nc)))]:
        if not labels:
            out[name] = {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
            continue
        p, r, f1, _ = precision_recall_fscore_support(
            yt, yp, labels=labels, average=avg, zero_division=0
        )
        out[name] = {"precision": float(p), "recall": float(r), "f1": float(f1)}
    return out


def confusion_counts(y_true, y_pred, nc, include_background_pred=True):
    """
    Raw confusion counts on foreground TRUE nodes.
    Rows = true foreground class (0..nc-1).
    Cols = predicted class (0..nc-1, plus background=nc if include_background_pred).
    cm[i, j] = #(true == i and pred == j).
    """
    m = foreground_mask(y_true, nc)
    yt, yp = y_true[m], y_pred[m]
    pred_labels = list(range(nc + 1)) if include_background_pred else list(range(nc))
    cm = confusion_matrix(yt, yp, labels=list(range(nc + 1)))
    cm = cm[:nc, :]  # keep only foreground true rows
    if not include_background_pred:
        cm = cm[:, :nc]
    return cm


def normalize_confusion(cm, mode):
    """
    mode='recall'    -> normalize each TRUE row (row sums to 1); diagonal = recall.
    mode='precision' -> normalize each PRED col (col sums to 1); diagonal = precision.
    Zero-denominator rows/cols are left as 0.
    """
    cm = cm.astype(float)
    if mode == "recall":
        denom = cm.sum(axis=1, keepdims=True)
    elif mode == "precision":
        denom = cm.sum(axis=0, keepdims=True)
    else:
        raise ValueError("mode must be 'recall' or 'precision'")
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.where(denom > 0, cm / denom, 0.0)
    return norm


def node_edit_counts(y_true, y_pred_gat, y_pred_yolo, nc):
    """
    Among foreground nodes:
      fixed  = YOLO wrong AND GAT correct
      broken = YOLO correct AND GAT wrong
      net    = fixed - broken
    """
    m = foreground_mask(y_true, nc)
    yt, yg, yy = y_true[m], y_pred_gat[m], y_pred_yolo[m]
    yolo_correct = yy == yt
    gat_correct = yg == yt
    fixed = int(np.sum(~yolo_correct & gat_correct))
    broken = int(np.sum(yolo_correct & ~gat_correct))
    return {"fixed": fixed, "broken": broken, "net": fixed - broken}


def false_positive_rejection_rate(y_true, y_pred_gat, nc):
    """Fraction of TRUE background nodes (y == nc) that GAT labels as background."""
    m = y_true == nc
    if m.sum() == 0:
        return {"rate": float("nan"), "n": 0}
    correct_bg = int(np.sum(y_pred_gat[m] == nc))
    return {"rate": correct_bg / int(m.sum()), "n": int(m.sum())}


def per_class_recall_yolo_vs_gat(y_true, y_pred_gat, y_pred_yolo, nc):
    """Per-class recall for both models, foreground nodes only, for the comparison bars."""
    m = foreground_mask(y_true, nc)
    yt, yg, yy = y_true[m], y_pred_gat[m], y_pred_yolo[m]
    _, r_gat, _, _ = precision_recall_fscore_support(
        yt, yg, labels=list(range(nc)), average=None, zero_division=0)
    _, r_yolo, _, _ = precision_recall_fscore_support(
        yt, yy, labels=list(range(nc)), average=None, zero_division=0)
    return {c: {"yolo_recall": float(r_yolo[c]), "gat_recall": float(r_gat[c])}
            for c in range(nc)}