"""
02_extract_graphs.py — Run the fine-tuned YOLO over a split, keep the FULL class
probability vector per detection (Ultralytics' Results object only gives you the
top-1 class, which throws away exactly the information the GAT needs), then build
one kNN graph per image.

Usage:
    python 02_extract_graphs.py --weights runs/detect/yolo_pcb/weights/best.pt \
        --root /data/merged --split test --imgsz 1280 --k 8 --out graphs/test.pt

Notes
-----
* Nodes are YOLO *predictions*, not ground-truth boxes. Training the GAT on GT
  boxes and testing it on predictions is a distribution mismatch that will silently
  cost you several points of accuracy.
* Unmatched predictions get label = NC (a "background" class). The GAT therefore
  learns context-based false-positive rejection for free — report this separately.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision
from tqdm import tqdm

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------- detections
@torch.no_grad()
def raw_detect(model, img_bgr, device, imgsz=1280, conf_th=0.20, iou_th=0.5):
    """Return (boxes_xyxy [N,4] in original px, probs [N,NC])."""
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils import ops

    h0, w0 = img_bgr.shape[:2]
    im = LetterBox((imgsz, imgsz), auto=False)(image=img_bgr)
    im = np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1))  # BGR->RGB, CHW
    t = torch.from_numpy(im).float().div(255).unsqueeze(0).to(device)

    out = model.model(t)
    p = out[0] if isinstance(out, (list, tuple)) else out      # (1, 4+NC, A)
    p = p[0].transpose(0, 1)                                    # (A, 4+NC)
    boxes_xywh, scores = p[:, :4], p[:, 4:]                     # already sigmoid'd
    best, cls = scores.max(1)

    keep = best > conf_th
    if keep.sum() == 0:
        return torch.zeros(0, 4), torch.zeros(0, scores.shape[1])
    boxes_xywh, scores, best, cls = boxes_xywh[keep], scores[keep], best[keep], cls[keep]

    xyxy = ops.xywh2xyxy(boxes_xywh)
    keep2 = torchvision.ops.batched_nms(xyxy, best, cls, iou_th)
    xyxy, scores = xyxy[keep2], scores[keep2]
    xyxy = ops.scale_boxes(t.shape[2:], xyxy, (h0, w0))
    return xyxy.cpu(), scores.cpu()


def load_gt(lab_path, w, h):
    """YOLO txt -> (boxes_xyxy [M,4] px, cls [M])."""
    if not lab_path.exists():
        return torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)
    b, c = [], []
    for line in lab_path.read_text().splitlines():
        f = line.split()
        if len(f) < 5:
            continue
        k, cx, cy, bw, bh = int(float(f[0])), *map(float, f[1:5])
        b.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                  (cx + bw / 2) * w, (cy + bh / 2) * h])
        c.append(k)
    return torch.tensor(b, dtype=torch.float), torch.tensor(c, dtype=torch.long)


def assign_labels(pred_xyxy, gt_xyxy, gt_cls, nc, iou_th=0.5):
    """Greedy IoU matching. Unmatched prediction -> background class `nc`."""
    y = torch.full((len(pred_xyxy),), nc, dtype=torch.long)
    if len(pred_xyxy) == 0 or len(gt_xyxy) == 0:
        return y
    iou = torchvision.ops.box_iou(pred_xyxy, gt_xyxy)          # [N, M]
    taken = set()
    order = iou.max(1).values.argsort(descending=True)
    for i in order.tolist():
        j = int(iou[i].argmax())
        if iou[i, j] >= iou_th and j not in taken:
            y[i] = gt_cls[j]
            taken.add(j)
    return y


# ---------------------------------------------------------------- graph build
def build_graph(xyxy, probs, y, W, H, k=8):
    """Node feats [N, NC+7]; edge feats [E, 8]."""
    eps = 1e-6
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2 / W
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2 / H
    bw = (xyxy[:, 2] - xyxy[:, 0]).clamp(min=1) / W
    bh = (xyxy[:, 3] - xyxy[:, 1]).clamp(min=1) / H

    geom = torch.stack([
        torch.log(bw + eps), torch.log(bh + eps),
        torch.log(bw / (bh + eps) + eps),            # log aspect
        torch.log(bw * bh + eps),                    # log area
        cx, cy,
        probs.max(1).values,                         # detector confidence
    ], 1)
    x = torch.cat([probs, geom], 1)                  # [N, NC+7]

    pos = torch.stack([cx, cy], 1)
    N = len(pos)
    kk = min(k, max(N - 1, 1))
    if N < 2:
        ei = torch.zeros(2, 0, dtype=torch.long)
        ea = torch.zeros(0, 8)
    else:
        d = torch.cdist(pos, pos)
        d.fill_diagonal_(float("inf"))
        nb = d.topk(kk, largest=False).indices               # [N, kk]
        src = torch.arange(N).repeat_interleave(kk)
        dst = nb.reshape(-1)
        ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])  # symmetrize
        ei = torch.unique(ei, dim=1)

        i, j = ei
        dx, dy = cx[j] - cx[i], cy[j] - cy[i]
        dist = torch.sqrt(dx ** 2 + dy ** 2) + eps
        ea = torch.stack([
            dx, dy, dist, dy / dist, dx / dist,                  # Δ, distance, sin, cos
            torch.log(bw[j] / (bw[i] + eps) + eps),
            torch.log(bh[j] / (bh[i] + eps) + eps),
            torch.log((bw[j] * bh[j]) / (bw[i] * bh[i] + eps) + eps),
        ], 1)

    return dict(x=x, edge_index=ei, edge_attr=ea, y=y,
                yolo_probs=probs, pos=pos, xyxy=xyxy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--root", required=True, help="dataset root with <split>/images")
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.20)
    a = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(a.weights)
    model.model.to(device).eval()
    nc = int(model.model.nc)
    print(f"device={device}  nc={nc}")

    idir = Path(a.root) / a.split / "images"
    ldir = Path(a.root) / a.split / "labels"
    imgs = sorted(p for p in idir.iterdir() if p.suffix.lower() in IMG_EXT)

    graphs, skipped = [], 0
    for p in tqdm(imgs, desc=f"extract[{a.split}]"):
        im = cv2.imread(str(p))
        if im is None:
            continue
        H, W = im.shape[:2]
        xyxy, probs = raw_detect(model, im, device, a.imgsz, a.conf)
        if len(xyxy) < 3:                    # a 2-node graph teaches nothing
            skipped += 1
            continue
        gt_b, gt_c = load_gt(ldir / (p.stem + ".txt"), W, H)
        y = assign_labels(xyxy, gt_b, gt_c, nc)
        g = build_graph(xyxy, probs, y, W, H, a.k)
        g["name"] = p.name
        graphs.append(g)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(graphs=graphs, nc=nc, k=a.k), a.out)

    nodes = sum(len(g["y"]) for g in graphs)
    bg = sum(int((g["y"] == nc).sum()) for g in graphs)
    print(f"\n{len(graphs)} graphs  |  {nodes} nodes  "
          f"({nodes/max(1,len(graphs)):.1f}/graph)  |  {bg} background nodes "
          f"({100*bg/max(1,nodes):.1f}% false positives)")
    print(f"skipped {skipped} images with <3 detections")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
