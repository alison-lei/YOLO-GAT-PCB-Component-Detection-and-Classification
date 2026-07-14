"""
04_boards.py — Reassemble tiled Kaggle labels into full-board coordinates.

Filename grammar (verify with --verify before trusting it!):
    {board}__{tile}__{x}___{y}.png      e.g. 00001__1024__4944___3296.png
    note: THREE underscores before y, TWO elsewhere -> unambiguous split.

Two jobs:
  1. Merge every tile's labels into one board-level label file (for GAT graphs).
     No image stitching needed — the GAT never sees pixels.
  2. Report which boards exist, so you can re-split train/val/test BY BOARD.
     A random tile split leaks: overlapping tiles land on both sides.

Usage:
    # sanity-check the x/y convention first
    python 04_boards.py --root /data/kaggle --verify 00001 --out boards/

    # then build all boards
    python 04_boards.py --root /data/kaggle --out boards/

    # if the mosaic looks transposed:
    python 04_boards.py --root /data/kaggle --out boards/ --swap-xy
"""
import argparse, json, re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision

IMG_EXT = {".png", ".jpg", ".jpeg"}
PAT = re.compile(r"^(?P<board>[^_]+)__(?P<tile>\d+)__(?P<a>\d+)___(?P<b>\d+)$")
EDGE_MARGIN = 3          # px: a box within this of a tile edge is "truncated"
DEDUP_IOU = 0.5


def parse(stem, swap):
    m = PAT.match(stem)
    if not m:
        return None
    a, b = int(m["a"]), int(m["b"])
    x, y = (b, a) if swap else (a, b)
    return m["board"], int(m["tile"]), x, y


def read_labels(p, T):
    """YOLO txt (normalized to the TILE) -> [(cls, x1, y1, x2, y2)] in tile px."""
    out = []
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        f = line.split()
        if len(f) < 5:
            continue
        c, cx, cy, w, h = int(float(f[0])), *map(float, f[1:5])
        out.append((c, (cx - w / 2) * T, (cy - h / 2) * T,
                    (cx + w / 2) * T, (cy + h / 2) * T))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Kaggle root with */images, */labels")
    ap.add_argument("--out", required=True)
    ap.add_argument("--swap-xy", action="store_true")
    ap.add_argument("--verify", default=None, metavar="BOARD_ID",
                    help="stitch a downscaled mosaic of this board and exit")
    a = ap.parse_args()
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)

    tiles = defaultdict(list)          # board -> [(x, y, T, img_path, lab_path)]
    for split in ("train", "valid", "val", "test"):
        idir = Path(a.root) / split / "images"
        if not idir.is_dir():
            continue
        for img in idir.iterdir():
            if img.suffix.lower() not in IMG_EXT:
                continue
            got = parse(img.stem, a.swap_xy)
            if not got:
                print(f"  ! unparsed filename, skipping: {img.name}")
                continue
            board, T, x, y = got
            lab = Path(a.root) / split / "labels" / (img.stem + ".txt")
            tiles[board].append((x, y, T, img, lab))

    print(f"{len(tiles)} boards, {sum(len(v) for v in tiles.values())} tiles\n")

    # ---- verification: stitch one board so you can EYEBALL the x/y convention
    if a.verify:
        import cv2
        ts = tiles[a.verify]
        if not ts:
            raise SystemExit(f"no tiles for board {a.verify}")
        T = ts[0][2]
        W = max(x for x, _, _, _, _ in ts) + T
        H = max(y for _, y, _, _, _ in ts) + T
        s = 0.12
        canvas = np.zeros((int(H * s), int(W * s), 3), np.uint8)
        for x, y, T, img, _ in ts:
            im = cv2.imread(str(img))
            if im is None:
                continue
            small = cv2.resize(im, (int(T * s), int(T * s)))
            yy, xx = int(y * s), int(x * s)
            canvas[yy:yy + small.shape[0], xx:xx + small.shape[1]] = small
        out = outdir / f"verify_{a.verify}.jpg"
        cv2.imwrite(str(out), canvas)
        print(f"board {a.verify}: {W} x {H} px")
        print(f"-> {out}   <-- OPEN THIS. Coherent board? good. "
              f"Transposed/scrambled? re-run with --swap-xy")
        return

    # ---- board-level label reconstruction
    meta = {}
    for board, ts in sorted(tiles.items()):
        T = ts[0][2]
        W = max(x for x, _, _, _, _ in ts) + T
        H = max(y for _, y, _, _, _ in ts) + T

        boxes, cls, score = [], [], []
        for x, y, T, _, lab in ts:
            for c, x1, y1, x2, y2 in read_labels(lab, T):
                # boxes hugging a tile edge are truncated; the neighbouring
                # overlapping tile has the same part in full. Score them lower
                # so NMS prefers the complete copy, but keep them in case a
                # large IC is truncated in EVERY tile it appears in.
                edge = (x1 < EDGE_MARGIN or y1 < EDGE_MARGIN or
                        x2 > T - EDGE_MARGIN or y2 > T - EDGE_MARGIN)
                boxes.append([x1 + x, y1 + y, x2 + x, y2 + y])
                cls.append(c)
                score.append(0.5 if edge else 1.0)

        if not boxes:
            continue
        b = torch.tensor(boxes, dtype=torch.float)
        c = torch.tensor(cls)
        s = torch.tensor(score)
        keep = torchvision.ops.batched_nms(b, s, c, DEDUP_IOU)
        b, c = b[keep], c[keep]

        # write board-level labels, normalized by the ISOTROPIC scale max(W,H)
        # so that angles, distances and aspect ratios are metric.
        S = max(W, H)
        lines = [f"{int(k)} {(x1+x2)/2/S:.6f} {(y1+y2)/2/S:.6f} "
                 f"{(x2-x1)/S:.6f} {(y2-y1)/S:.6f}"
                 for k, (x1, y1, x2, y2) in zip(c.tolist(), b.tolist())]
        (outdir / f"{board}.txt").write_text("\n".join(lines))
        meta[board] = dict(W=W, H=H, scale=S, n_tiles=len(ts),
                           n_raw=len(boxes), n_dedup=len(b))
        print(f"  {board}: {W}x{H}  {len(ts):3d} tiles  "
              f"{len(boxes):5d} raw -> {len(b):5d} components")

    (outdir / "boards.json").write_text(json.dumps(meta, indent=2))
    n = [m["n_dedup"] for m in meta.values()]
    print(f"\n{len(meta)} boards | components/board: "
          f"min {min(n)} median {int(np.median(n))} max {max(n)}")
    print("\nNEXT: split train/val/test BY BOARD ID, not by tile.")


if __name__ == "__main__":
    main()
