"""
11_build_graphs.py — One command to turn a YOLO dataset split into GAT graphs.

Pipeline per image:
  1. run frozen YOLO, keep full 22-d probability vector per detection (raw head)
  2. IoU-match each detection to GT -> node label (unmatched = background class)
  3. build Delaunay + anchor-edge graph
  4. (train only) also emit 90/180/270 rotations for 4x augmentation

Reuses the detection code in 02_extract_graphs.py and the graph builder in
02b_graph_delaunay.py so there is one source of truth for each.

Usage:
  python 11_build_graphs.py --weights best.pt --root merged_fb --split train --out graphs/train.pt --augment
  python 11_build_graphs.py --weights best.pt --root merged_fb --split valid --out graphs/valid.pt
  python 11_build_graphs.py --weights best.pt --root fpic_yolo  --split test  --out graphs/test.pt

conf: use 0.15 to raise recall into the graph (GAT rejects the extra FPs).
"""
import argparse
import importlib.util as ilu
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

HERE = Path(__file__).parent


def _load(name, file):
    s = ilu.spec_from_file_location(name, str(HERE / file))
    m = ilu.module_from_spec(s); s.loader.exec_module(m); return m


ext = _load("ext", "02_extract_graphs.py")          # raw_detect, load_gt, assign_labels
gb = _load("gb", "02b_graph_delaunay.py")            # build_graph_delaunay / _augmented

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", required=True)          # train / valid / test
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--augment", action="store_true",
                    help="emit 4 rotations per graph (train only)")
    ap.add_argument("--min-nodes", type=int, default=3)
    a = ap.parse_args()

    from ultralytics import YOLO
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(a.weights); model.model.to(dev).eval()
    nc = int(model.model.nc)
    print(f"device={dev}  nc={nc}  conf={a.conf}  augment={a.augment}")

    idir = Path(a.root) / a.split / "images"
    ldir = Path(a.root) / a.split / "labels"
    imgs = sorted(p for p in idir.iterdir() if p.suffix.lower() in IMG_EXT)
    print(f"{len(imgs)} images in {idir}")

    graphs, skipped = [], 0
    for p in tqdm(imgs, desc=f"build[{a.split}]"):
        im = cv2.imread(str(p))
        if im is None:
            skipped += 1; continue
        H, W = im.shape[:2]
        xyxy, probs = ext.raw_detect(model, im, dev, a.imgsz, a.conf)
        if len(xyxy) < a.min_nodes:
            skipped += 1; continue
        gt_b, gt_c = ext.load_gt(ldir / (p.stem + ".txt"), W, H)
        y = ext.assign_labels(xyxy, gt_b, gt_c, nc)

        if a.augment:
            outs = gb.build_graph_augmented(xyxy, probs, y, W, H)
        else:
            outs = [gb.build_graph_delaunay(xyxy, probs, y, W, H)]
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
