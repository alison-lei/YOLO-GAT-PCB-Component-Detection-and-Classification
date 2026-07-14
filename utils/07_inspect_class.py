"""
07_inspect_class.py — Settle the ** VERIFY ** rows in maps/roboflow.json by
looking at the actual pixels. Dumps N crops of one class into a folder.

The question you are answering is always the same:
    "Roboflow calls this thing X. What does Kaggle call the SAME physical part?"

So run it twice and put the two folders side by side:

    # 1. What does a Roboflow "Resistor Network" (class 17) look like?
    python 07_inspect_class.py --root /data/roboflow --cls 17 --out look/rf_resnet

    # 2. Find that same package in Kaggle. Is it labelled 'resistor' (17) or 'ic' (10)?
    python 07_inspect_class.py --root /data/kaggle --cls 10 --out look/kg_ic
    python 07_inspect_class.py --root /data/kaggle --cls 17 --out look/kg_resistor

Whichever Kaggle folder contains the multi-pad SOIC-looking packages tells you
the right target. Repeat for Test Point, Jumper, EM.

Usage:
    python 07_inspect_class.py --root DIR --cls ID [--n 40] [--pad 12] [--out DIR]
"""
import argparse
from pathlib import Path

import cv2

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cls", type=int, required=True, help="class id IN THAT DATASET's yaml")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--pad", type=int, default=12, help="context pixels around the box")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = Path(a.out or f"look/cls{a.cls}")
    out.mkdir(parents=True, exist_ok=True)

    got = 0
    for sp in ("train", "valid", "val", "test"):
        idir = Path(a.root) / sp / "images"
        if not idir.is_dir():
            continue
        for img in sorted(p for p in idir.iterdir() if p.suffix.lower() in IMG_EXT):
            if got >= a.n:
                break
            lab = Path(a.root) / sp / "labels" / (img.stem + ".txt")
            if not lab.exists():
                continue
            rows = [l.split() for l in lab.read_text().splitlines() if len(l.split()) >= 5]
            hits = [r for r in rows if int(float(r[0])) == a.cls]
            if not hits:
                continue

            im = cv2.imread(str(img))
            if im is None:
                continue
            H, W = im.shape[:2]
            for i, r in enumerate(hits):
                if got >= a.n:
                    break
                cx, cy, w, h = (float(v) for v in r[1:5])
                x1 = max(0, int((cx - w / 2) * W) - a.pad)
                y1 = max(0, int((cy - h / 2) * H) - a.pad)
                x2 = min(W, int((cx + w / 2) * W) + a.pad)
                y2 = min(H, int((cy + h / 2) * H) + a.pad)
                crop = im[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                # upscale tiny parts so you can actually see them
                s = max(1, int(160 / max(1, max(crop.shape[:2]))))
                if s > 1:
                    crop = cv2.resize(crop, None, fx=s, fy=s,
                                      interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(out / f"{img.stem}_{i}.png"), crop)
                got += 1

    print(f"wrote {got} crops of class {a.cls} -> {out}/")
    if got == 0:
        print("  (none found — check the class id against that dataset's data.yaml)")


if __name__ == "__main__":
    main()
