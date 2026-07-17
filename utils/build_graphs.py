"""
build_graphs.py runs on a single split of data (train, valid, or test)
It takes the predicted probability vectors from frozen YOLO and feeds it into the nodes of the graphs

Graphs use Delaunay + anchor edges

One graph created per image:
  1. run frozen YOLO, keep the full class-probability vector for each detected component, GAT learns from the uncertainty provided by YOLO
  2. IoU-match each detected node to ground truth -> unmatched detection is assigned to background class, index = nc)
  3. build a Delaunay + anchor-edge graph.

Output is a .pt file holding the graphs and the number of instances of false positives -> {graphs: [dict,...], nc: int, k: "delaunay+anchor"}.
Each graph dict fully describes one graph: x, edge_index, edge_attr, y, yolo_probs, pos, xyxy, name.

Commands:
  python utils/build_graphs.py --weights best.pt --root data --split train --out graphs/train.pt
  python utils/build_graphs.py --weights best.pt --root data --split valid --out graphs/valid.pt

conf: default 0.15 raises recall into the graph (GAT then learns to rejects the extra false positives)

"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision
from scipy.spatial import Delaunay
from tqdm import tqdm

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# Run the frozen YOLO weights. Don't use the single prediction from Ultralytics
# Run the forward function again to get the full probability vector
@torch.no_grad()
def raw_detect(model, img_bgr, device, imgsz=1024, conf_th=0.15, iou_th=0.5):
    # Return (boxes_xyxy [N,4] in original px, probs [N,NC]) which is full probability vector
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils import ops

    h0, w0 = img_bgr.shape[:2]
    im = LetterBox((imgsz, imgsz), auto=False)(image=img_bgr)
    # Convert BGR to RGB, CHW
    im = np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1))
    t = torch.from_numpy(im).float().div(255).unsqueeze(0).to(device)

    out = model.model(t)
    p = out[0] if isinstance(out, (list, tuple)) else out
    p = p[0].transpose(0, 1)
    boxes_xywh, scores = p[:, :4], p[:, 4:]
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


# Reads the YOLO label files to determine the ground truth bounding boxes and class (boxes_xyxy [M,4] px, cls [M])
def load_gt(lab_path, w, h):
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


# Greedy IoU matching. Each detected node is compared to the ground truth and whether there is actually a node there
# Keep it as a node if IoU is over the threshold of 0.5
# If it is predicted as a component/node but does not exist in ground truth then it is labeled as background (class index is nc)
# Lets GAT learn false positive rejection
def assign_labels(pred_xyxy, gt_xyxy, gt_cls, nc, iou_th=0.5):
    y = torch.full((len(pred_xyxy),), nc, dtype=torch.long)
    if len(pred_xyxy) == 0 or len(gt_xyxy) == 0:
        return y
    iou = torchvision.ops.box_iou(pred_xyxy, gt_xyxy)
    taken = set()
    order = iou.max(1).values.argsort(descending=True)
    for i in order.tolist():
        j = int(iou[i].argmax())
        if iou[i, j] >= iou_th and j not in taken:
            y[i] = gt_cls[j]
            taken.add(j)
    return y


# build Delaunay graph with also anchor to the nearest 2 largest components
def build_graph_delaunay(xyxy, probs, y, W, H, n_anchor=2, anchor_pct=75):
    """
    Node features  x  : [N, C+7]  = [C probs | log w | log h | log aspect | log area | cx | cy | conf]
                                    C probs are the YOLO's predicted probability distribution
                                    conf determines recall, lower conf means higehr recall/more components are detected, but accuracy decreases.
                                    GAT's purpose is to increase accuracy, can only do so if YOLO actually detects a node and includes it in graph

    Edge features  ea : [E, 8]    = [dx, dy, dist, sin, cos, log wratio, log hratio, log aratio]
    Edges: Delaunay local adjacency (~6/node) + each node -> its n_anchor nearest large components.
    """
    eps = 1e-6
    s = float(max(W, H))

    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2 / s
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2 / s
    bw = ((xyxy[:, 2] - xyxy[:, 0]).clamp(min=1)) / s
    bh = ((xyxy[:, 3] - xyxy[:, 1]).clamp(min=1)) / s
    area = bw * bh

    geom = torch.stack([
        torch.log(bw + eps), torch.log(bh + eps),
        torch.log(bw / (bh + eps) + eps),
        torch.log(area + eps),
        cx, cy,
        probs.max(1).values,
    ], 1)
    x = torch.cat([probs, geom], 1)

    pos = torch.stack([cx, cy], 1)
    N = len(pos)
    edges = set()

    # Types of edges
    # 1. Delaunay local adjacency (needs >= 3 non-collinear points). Generally shows the connectino between components
    if N >= 3:
        try:
            tri = Delaunay(pos.numpy())
            for simplex in tri.simplices:
                for a in range(3):
                    for b in range(a + 1, 3):
                        i, j = int(simplex[a]), int(simplex[b])
                        edges.add((i, j))
                        edges.add((j, i))
        except Exception:
            pass

    # 2. anchor edges: every node -> its n_anchor nearest large components (like ic)
    # This captures long range relationships
    if N >= 2:
        big = torch.where(area >= np.percentile(area.numpy(), anchor_pct))[0]
        if len(big) > 0:
            d = torch.cdist(pos, pos[big])
            for i in range(N):
                added = 0
                for o in d[i].argsort().tolist():
                    j = int(big[o])
                    if j == i:
                        continue
                    edges.add((i, j))
                    edges.add((j, i))
                    added += 1
                    if added >= n_anchor:
                        break

    # fallback: tiny board with no edges -> nearest neighbour
    if not edges and N >= 2:
        d = torch.cdist(pos, pos)
        d.fill_diagonal_(float("inf"))
        for i in range(N):
            j = int(d[i].argmin())
            edges.add((i, j))
            edges.add((j, i))

    ei = (torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()
          if edges else torch.zeros(2, 0, dtype=torch.long))

    if ei.shape[1] > 0:
        i, j = ei
        dx, dy = cx[j] - cx[i], cy[j] - cy[i]
        dist = torch.sqrt(dx ** 2 + dy ** 2) + eps
        ea = torch.stack([
            dx, dy, dist, dy / dist, dx / dist,
            torch.log(bw[j] / (bw[i] + eps) + eps),
            torch.log(bh[j] / (bh[i] + eps) + eps),
            torch.log(area[j] / (area[i] + eps) + eps),
        ], 1)
    else:
        ea = torch.zeros(0, 8)

    return dict(x=x, edge_index=ei, edge_attr=ea, y=y,
                yolo_probs=probs, pos=pos, xyxy=xyxy)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--root", required=True, help="dataset root with <split>/images")
    ap.add_argument("--split", required=True, help="train / valid / test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--min-nodes", type=int, default=3)
    a = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(a.weights)
    model.model.to(dev).eval()
    nc = int(model.model.nc)
    print(f"device={dev}  nc={nc}  conf={a.conf}  imgsz={a.imgsz}  augment={a.augment}")

    idir = Path(a.root) / a.split / "images"
    ldir = Path(a.root) / a.split / "labels"
    imgs = sorted(p for p in idir.iterdir() if p.suffix.lower() in IMG_EXT)
    print(f"{len(imgs)} images in {idir}")

    graphs, skipped = [], 0
    for p in tqdm(imgs, desc=f"build[{a.split}]"):
        im = cv2.imread(str(p))
        if im is None:
            skipped += 1
            continue
        H, W = im.shape[:2]
        xyxy, probs = raw_detect(model, im, dev, a.imgsz, a.conf)
        # check if there are too little components on each PCB board, if so then remove them
        if len(xyxy) < a.min_nodes:
            skipped += 1
            continue
        gt_b, gt_c = load_gt(ldir / (p.stem + ".txt"), W, H)
        y = assign_labels(xyxy, gt_b, gt_c, nc)

        outs = [build_graph_delaunay(xyxy, probs, y, W, H)]
        for k, g in enumerate(outs):
            g["name"] = f"{p.stem}_r{k}" if a.augment else p.stem
            graphs.append(g)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(graphs=graphs, nc=nc, k="delaunay+anchor"), a.out)

    nodes = sum(len(g["y"]) for g in graphs)
    bg = sum(int((g["y"] == nc).sum()) for g in graphs)
    print(f"\n{len(graphs)} graphs | {nodes} nodes ({nodes/max(1,len(graphs)):.1f}/graph)"
          f" | {bg} background/FP nodes ({100*bg/max(1,nodes):.1f}%)")
    print(f"skipped {skipped} images (<{a.min_nodes} detections or unreadable)")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
