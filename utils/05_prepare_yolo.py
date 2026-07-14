"""
05_prepare_yolo.py — Class remap + CLAHE + dihedral expansion (geometry-aware, symmetry-preserving image processing model).
Run on each source dataset; they all land in one merged YOLO root with a shared taxonomy.

Offline we do ONLY the transforms Ultralytics cannot do at runtime:
  * CLAHE            -> deterministic preprocessing; must match at inference
  * rot90            -> Ultralytics has no 90-degree rotation
Everything else (hue, saturation, value, flips, scale, translate, mosaic,
erasing) happens at runtime in the training loop. Do NOT bake those to disk.

Why rot90 alone is enough:
  {identity, rot90} x {identity, fliplr, flipud, fliplr+flipud}  =  all 8
  elements of the dihedral group D4. Ultralytics supplies the right-hand set
  via fliplr=0.5 flipud=0.5. So 2x on disk buys you full 8x coverage.

Usage:
    python 05_prepare_yolo.py --src /data/kaggle   --map maps/kaggle.json   --out merged --split-json boards/split.json
    python 05_prepare_yolo.py --src /data/roboflow --map maps/roboflow.json --out merged

map json:  {"0": 3, "1": 3, "2": 9, ...}   old class id -> canonical id, or -1 to DROP
"""
import argparse, json, shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}

CANON = ['battery', 'button', 'buzzer', 'capacitor', 'clock', 'connector', 'diode',
         'display', 'fuse', 'heatsink', 'ic', 'inductor', 'led', 'pads', 'pins',
         'potentiometer', 'relay', 'resistor', 'switch', 'transducer',
         'transformer', 'transistor', 'unknown']


def clahe(bgr, clip=2.0, grid=8):
    """CLAHE on the L channel of LAB. Never per-RGB-channel: that shifts hue."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def rot90_labels(rows):
    """Rotate normalized YOLO labels 90 deg CCW (matches np.rot90 on the image).
    (cx, cy) -> (cy, 1 - cx);  w and h swap."""
    return [(c, cy, 1.0 - cx, h, w) for c, cx, cy, w, h in rows]


def read_rows(p):
    rows = []
    if p.exists():
        for line in p.read_text().splitlines():
            f = line.split()
            if len(f) >= 5:
                rows.append((int(float(f[0])), *map(float, f[1:5])))
    return rows


def write_rows(p, rows):
    p.write_text("\n".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                           for c, cx, cy, w, h in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--map", required=True, help="json: old id -> canonical id (-1 drops)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-clahe", action="store_true")
    ap.add_argument("--no-rot90", action="store_true")
    ap.add_argument("--split-json", default=None,
                    help='board-level split: {"train": ["00001",...], "valid": [...], "test": [...]}')
    a = ap.parse_args()

    cmap = {int(k): int(v) for k, v in json.loads(Path(a.map).read_text()).items()}
    split_of = None
    if a.split_json:
        sj = json.loads(Path(a.split_json).read_text())
        split_of = {b: s for s, bs in sj.items() for b in bs}

    out = Path(a.out)
    for sp in ("train", "valid", "test"):
        (out / sp / "images").mkdir(parents=True, exist_ok=True)
        (out / sp / "labels").mkdir(parents=True, exist_ok=True)

    n_img = n_obj = n_drop = 0
    for sp in ("train", "valid", "val", "test"):
        idir = Path(a.src) / sp / "images"
        if not idir.is_dir():
            continue
        for img in tqdm(sorted(p for p in idir.iterdir()
                               if p.suffix.lower() in IMG_EXT), desc=f"{a.src}:{sp}"):
            rows = read_rows(Path(a.src) / sp / "labels" / (img.stem + ".txt"))
            rows = [(cmap[c], cx, cy, w, h) for c, cx, cy, w, h in rows
                    if cmap.get(c, -1) >= 0]
            n_drop += len(read_rows(Path(a.src) / sp / "labels" / (img.stem + ".txt"))) - len(rows)

            # board-level split overrides the source split (kills tile leakage)
            dst_sp = sp if sp != "val" else "valid"
            if split_of:
                board = img.stem.split("__")[0]
                dst_sp = split_of.get(board, dst_sp)

            im = cv2.imread(str(img))
            if im is None:
                continue
            if not a.no_clahe:
                im = clahe(im)

            variants = [("", im, rows)]
            # only expand the TRAIN split — never inflate val/test
            if not a.no_rot90 and dst_sp == "train":
                variants.append(("_r90", np.ascontiguousarray(np.rot90(im)),
                                 rot90_labels(rows)))

            for suf, vim, vrows in variants:
                stem = f"{a.src.strip('/').split('/')[-1]}_{img.stem}{suf}"
                cv2.imwrite(str(out / dst_sp / "images" / f"{stem}.jpg"), vim,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                write_rows(out / dst_sp / "labels" / f"{stem}.txt", vrows)
                n_img += 1
                n_obj += len(vrows)

    yaml = out / "data.yaml"
    if not yaml.exists():
        yaml.write_text(
            f"path: {out.resolve()}\ntrain: train/images\nval: valid/images\n"
            f"test: test/images\nnc: {len(CANON)}\nnames: {CANON}\n")

    print(f"\nwrote {n_img} images, {n_obj} objects, dropped {n_drop} unmapped objects")
    print(f"-> {out}/  (data.yaml nc={len(CANON)})")
    print("\nReminder: CLAHE is now baked in. Apply the IDENTICAL clahe() call to "
          "every image at inference, or train/test distributions will not match.")


if __name__ == "__main__":
    main()
