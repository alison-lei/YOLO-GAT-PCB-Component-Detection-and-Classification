"""
09_prepare_fullboards.py — Build the YOLO training set from FULL BOARDS ONLY.

Excludes:
  * TILE images  (filenames like 00025__1024__1648___0)  -> FPIC-derived, dropped
  * CROP images  (battery2, inductor29, ...)              -> single-component, dropped
Keeps:
  * Kaggle full boards (PCBA_17, ArduinoMega_Top, ...)
  * all Roboflow boards

FPIC is NOT included here — it is the held-out test set, prepared separately by
08_fpic_csv.py and kept 100% out of train/val.

Split: train/val only (no test — FPIC is the test set). Split BY BOARD so no
board straddles the line.

Usage:
    python utils/09_prepare_fullboards.py --kaggle datasets/kaggle_dataset --kaggle-map utils/maps/kaggle.json --roboflow datasets/roboflow_dataset --roboflow-map utils/maps/roboflow.json --out merged_new --val-frac 0.15
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse, hashlib, json, re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
RE_TILE = re.compile(r"^\d+__\d+__\d+___\d+$")       # 00025__1024__1648___0
# crops are lowercase class-name + number, e.g. battery2, inductor29, capacitor187.
# full boards have capitals / hyphens / underscores (PCBA_17, ArduinoMega_Top,
# ACM-109_Bottom, EDA-008_Top) so they must NOT match this.
CROP_CLASSES = ('battery', 'button', 'buzzer', 'capacitor', 'clock', 'connector',
                'diode', 'display', 'fuse', 'heatsink', 'ic', 'inductor', 'led',
                'pads', 'pins', 'potentiometer', 'relay', 'resistor', 'switch',
                'transducer', 'transformer', 'transistor')
RE_CROP = re.compile(r"^(" + "|".join(CROP_CLASSES) + r")\d+$")

CANON = ['battery', 'button', 'buzzer', 'capacitor', 'clock', 'connector', 'diode',
         'display', 'fuse', 'heatsink', 'ic', 'inductor', 'led', 'pads', 'pins',
         'potentiometer', 'relay', 'resistor', 'switch', 'transducer',
         'transformer', 'transistor', 'unknown']


def is_fullboard(stem):
    return not (RE_TILE.match(stem) or RE_CROP.match(stem))


def clahe(bgr, clip=2.0, grid=8):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def rot90_labels(rows):
    return [(c, cy, 1.0 - cx, h, w) for c, cx, cy, w, h in rows]


def read_rows(p):
    rows = []
    if p.exists():
        for line in p.read_text().splitlines():
            f = line.split()
            if len(f) >= 5:
                rows.append((int(float(f[0])), *map(float, f[1:5])))
    return rows


def find_pairs(src):
    """Explicit train/valid/val/test -> images/labels traversal (matches triage)."""
    src = Path(src); pairs = []
    for sp in ("train", "valid", "val", "test"):
        idir = src / sp / "images"
        if not idir.is_dir():
            continue
        for img in sorted(idir.iterdir()):
            if img.suffix.lower() not in IMG_EXT:
                continue
            lab = src / sp / "labels" / (img.stem + ".txt")
            pairs.append((img, lab))
    return pairs


def load_map(path):
    return {int(k): int(v) for k, v in json.loads(Path(path).read_text()).items()
            if not str(k).startswith("_")}


def main():
    from config import CONFIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle"); ap.add_argument("--kaggle-map")
    ap.add_argument("--roboflow"); ap.add_argument("--roboflow-map")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])
    ap.add_argument("--no-clahe", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    for sp in ("train", "valid"):
        (out / sp / "images").mkdir(parents=True, exist_ok=True)
        (out / sp / "labels").mkdir(parents=True, exist_ok=True)

    # gather full-board pairs from each source
    jobs = []   # (img, lab, cmap, tag)
    for src, mp, tag in [(a.kaggle, a.kaggle_map, "kaggle"),
                         (a.roboflow, a.roboflow_map, "roboflow")]:
        if not src:
            continue
        cmap = load_map(mp)
        allp = find_pairs(src)
        tiles = [p for p in allp if RE_TILE.match(p[0].stem)]
        crops = [p for p in allp if RE_CROP.match(p[0].stem)]
        kept = [(img, lab) for img, lab in allp if is_fullboard(img.stem)]
        print(f"{tag}: {len(allp)} total | {len(tiles)} tiles dropped | "
              f"{len(crops)} crops dropped | {len(kept)} full boards kept")
        jobs += [(img, lab, cmap, tag) for img, lab in kept]

    # ---- EXACT duplicate removal (md5 of raw bytes only) ----
    # Deliberately NOT perceptual: two different boards from the same product
    # line can look near-identical without being the same board. Only genuine
    # byte-for-byte file copies are dropped.
    seen, deduped, n_dupe = {}, [], 0
    for img, lab, cmap, tag in tqdm(jobs, desc="dedup(md5)"):
        try:
            h = hashlib.md5(Path(img).read_bytes()).hexdigest()
        except Exception:
            deduped.append((img, lab, cmap, tag)); continue
        if h in seen:
            n_dupe += 1
            continue
        seen[h] = img.stem
        deduped.append((img, lab, cmap, tag))
    jobs = deduped
    print(f"exact duplicates removed: {n_dupe}  ->  {len(jobs)} unique images")

    # split by board id (stem) — full boards are already 1 board = 1 image
    rng = np.random.default_rng(a.seed)
    stems = sorted({img.stem for img, _, _, _ in jobs})
    rng.shuffle(stems)
    n_val = max(1, int(round(len(stems) * a.val_frac)))
    val_set = set(stems[:n_val])
    print(f"split: {len(stems)-n_val} train / {n_val} val boards")

    rot90_labels_fn = rot90_labels
    n_img = n_obj = n_drop = 0
    cls_hist = Counter()
    for img, lab, cmap, tag in tqdm(jobs, desc="prepare"):
        raw = read_rows(lab)
        rows = [(cmap[c], cx, cy, w, h) for c, cx, cy, w, h in raw
                if cmap.get(c, -1) >= 0]
        n_drop += len(raw) - len(rows)
        cls_hist.update(r[0] for r in rows)

        dst = "valid" if img.stem in val_set else "train"
        im = cv2.imread(str(img))
        if im is None:
            continue
        if not a.no_clahe:
            im = clahe(im)

        variants = [("", im, rows)]
        if dst == "train":
            variants.append(("_r90", np.ascontiguousarray(np.rot90(im)),
                             rot90_labels_fn(rows)))
        for suf, vim, vrows in variants:
            stem = f"{tag}_{img.stem}{suf}"
            cv2.imwrite(str(out / dst / "images" / f"{stem}.jpg"), vim,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out / dst / "labels" / f"{stem}.txt").write_text(
                "\n".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                          for c, cx, cy, w, h in vrows))
            n_img += 1; n_obj += len(vrows)

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: train/images\nval: valid/images\n"
        f"nc: {len(CANON)}\nnames: {CANON}\n")

    print(f"\nwrote {n_img} images, {n_obj} objects, dropped {n_drop} unmapped")
    print("class histogram (canonical id: count):")
    for c, n in sorted(cls_hist.items()):
        print(f"    {c:2d} {CANON[c]:>14s}: {n}")
    empty = [CANON[i] for i in range(len(CANON)) if i not in cls_hist]
    if empty:
        print(f"  ! classes with ZERO instances: {empty}")
    print(f"\n-> {out}/  (FPIC stays OUT — it is the test set)")


if __name__ == "__main__":
    main()