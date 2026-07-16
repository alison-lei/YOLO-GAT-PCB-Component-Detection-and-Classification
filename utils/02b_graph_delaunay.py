"""
02b_graph_delaunay.py — Drop-in replacement for the graph-building part of
02_extract_graphs.py. Same node features and edge features; the ONLY change is
how edges are chosen: Delaunay adjacency + anchor edges to large components,
instead of kNN.

Wire it in by replacing the call to build_graph(...) in 02_extract_graphs.py
with build_graph_delaunay(...). Everything else (raw_detect, assign_labels,
the file I/O) stays identical.

Why this graph:
  * Delaunay  -> parameter-free local adjacency, ~6 edges/node, the natural
                 "who is physically next to whom" on a planar board.
  * anchors   -> each node also links to its 2 nearest LARGE components (ICs,
                 connectors). A decoupling cap's identity is defined by the IC
                 it surrounds; Delaunay alone misses that long-range link.
"""
import numpy as np
import torch
from scipy.spatial import Delaunay


def rotate_boxes(xyxy, W, H, k):
    """Rotate boxes by k*90 degrees CCW about the image centre.
    Returns rotated xyxy and the new (W,H). Labels are unchanged by rotation."""
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    for _ in range(k % 4):
        # (x,y) -> (y, W - x); box corners swap, W/H swap
        nx1, ny1 = y1, W - x2
        nx2, ny2 = y2, W - x1
        x1, y1, x2, y2 = nx1, ny1, nx2, ny2
        W, H = H, W
    return torch.stack([x1, y1, x2, y2], 1), W, H


def build_graph_augmented(xyxy, probs, y, W, H, rotations=(0, 1, 2, 3), **kw):
    """Yield one graph per rotation. Same probs/labels; geometry rotated.
    Use for TRAIN graphs to 4x the data with real, label-preserving variety."""
    graphs = []
    for k in rotations:
        rb, rW, rH = rotate_boxes(xyxy, W, H, k)
        g = build_graph_delaunay(rb, probs, y, rW, rH, **kw)
        graphs.append(g)
    return graphs


def build_graph_delaunay(xyxy, probs, y, W, H, n_anchor=2, anchor_pct=75):
    """
    xyxy  : [N,4] detection boxes in ORIGINAL pixel coords
    probs : [N,C] YOLO class-probability vectors (the raw head output)
    y     : [N]   integer labels (C == background)
    returns dict(x, edge_index, edge_attr, y, yolo_probs, pos, xyxy)
    """
    eps = 1e-6
    s = float(max(W, H))                       # isotropic scale -> metric geometry

    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2 / s
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2 / s
    bw = ((xyxy[:, 2] - xyxy[:, 0]).clamp(min=1)) / s
    bh = ((xyxy[:, 3] - xyxy[:, 1]).clamp(min=1)) / s
    area = bw * bh

    # ---- node features: [C probs | log w | log h | log aspect | log area | cx | cy | conf]
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

    # ---------------- edges ----------------
    edges = set()

    # (1) Delaunay local adjacency (needs >= 3 non-collinear points)
    if N >= 3:
        try:
            tri = Delaunay(pos.numpy())
            for simplex in tri.simplices:
                for a in range(3):
                    for b in range(a + 1, 3):
                        i, j = int(simplex[a]), int(simplex[b])
                        edges.add((i, j)); edges.add((j, i))
        except Exception:
            pass  # degenerate (collinear) board -> fall through to anchors only

    # (2) anchor edges: every node -> its n_anchor nearest LARGE components
    if N >= 2:
        big = torch.where(area >= np.percentile(area.numpy(), anchor_pct))[0]
        if len(big) > 0:
            d = torch.cdist(pos, pos[big])          # [N, n_big]
            for i in range(N):
                order = d[i].argsort()
                added = 0
                for o in order.tolist():
                    j = int(big[o])
                    if j == i:
                        continue
                    edges.add((i, j)); edges.add((j, i))
                    added += 1
                    if added >= n_anchor:
                        break

    # fallback: if a tiny board produced no edges, connect nearest neighbour
    if not edges and N >= 2:
        d = torch.cdist(pos, pos); d.fill_diagonal_(float("inf"))
        for i in range(N):
            j = int(d[i].argmin())
            edges.add((i, j)); edges.add((j, i))

    if edges:
        ei = torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()
    else:
        ei = torch.zeros(2, 0, dtype=torch.long)

    # ---------------- edge features ----------------
    if ei.shape[1] > 0:
        i, j = ei
        dx, dy = cx[j] - cx[i], cy[j] - cy[i]
        dist = torch.sqrt(dx ** 2 + dy ** 2) + eps
        ea = torch.stack([
            dx, dy, dist, dy / dist, dx / dist,               # Δ, distance, sin, cos
            torch.log(bw[j] / (bw[i] + eps) + eps),
            torch.log(bh[j] / (bh[i] + eps) + eps),
            torch.log(area[j] / (area[i] + eps) + eps),
        ], 1)
    else:
        ea = torch.zeros(0, 8)

    return dict(x=x, edge_index=ei, edge_attr=ea, y=y,
                yolo_probs=probs, pos=pos, xyxy=xyxy)


# ------------------------------------------------------------------ quick test
if __name__ == "__main__":
    torch.manual_seed(50)
    N, C = 30, 22
    xyxy = torch.rand(N, 4) * 1000
    xyxy[:, 2:] = xyxy[:, :2] + torch.rand(N, 2) * 60 + 10
    probs = torch.softmax(torch.randn(N, C), 1)
    y = torch.randint(0, C, (N,))
    g = build_graph_delaunay(xyxy, probs, y, 1024, 1024)
    print("nodes:", g["x"].shape, "edges:", g["edge_index"].shape,
          "edge_attr:", g["edge_attr"].shape)
    deg = torch.bincount(g["edge_index"][0], minlength=N).float()
    print(f"avg degree: {deg.mean():.1f}  (Delaunay ~6 + anchors)")
