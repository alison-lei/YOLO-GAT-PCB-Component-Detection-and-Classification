"""
build_dataset.py
Takes raw Kaggle + Roboflow datasets stored as datasets/kaggle_dataset and datasets/roboflow_dataset,
splits it into data/data.yaml and data/yolo, data/train, data/valid folders where each folder further contains images/ and labels/.

Run from the two original datasets (Kaggle still has tiles+crops; Roboflow raw).

Image groups within each raw dataset
Kaggle:
  TILE  00025__1024__1648___0  -> YOLO only
  CROP  battery2, inductor29   -> YOLO only (rare-class source)
  BOARD PCBA_17, ArduinoMega_Top  -> YOLO + GAT + val
Roboflow: all full boards -> YOLO + GAT + val

Buckets (after augmentation, all rotations of a board stay together):
  yolo : TILE + CROP + share of BOARD and Roboflow (fine-tune YOLO)
  train : full boards YOLO never saw (GAT train)
  valid : full boards (final YOLO+GAT eval)

Steps:
classify -> md5 deduplication (remove perfect duplicates only) -> remap classes to follow cannonical format in utils/maps/kaggle.json -> 
CLAHE -> rot90 (train-time buckets) -> write.

Command:
  python utils/build_dataset.py --kaggle datasets/kaggle_dataset --kaggle-map utils/maps/kaggle.json --roboflow datasets/roboflow_dataset --roboflow-map utils/maps/roboflow.json --out data --yolo-frac 0.6 --train-frac 0.25

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

RE_TILE = re.compile(r"^\d+__\d+__\d+___\d+$")

CROP_CLASSES = ('battery','button','buzzer','capacitor','clock','connector','diode',
                'display','fuse','heatsink','ic','inductor','led','pads','pins',
                'potentiometer','relay','resistor','switch','transducer','transformer',
                'transistor')

RE_CROP = re.compile(r"^(" + "|".join(CROP_CLASSES) + r")\d+$")

CANON = ['battery','button','buzzer','capacitor','clock','connector','diode','display',
         'fuse','heatsink','ic','inductor','led','pads','pins','potentiometer','relay',
         'resistor','switch','transducer','transformer','transistor','unknown']


def group_of(stem):
    if RE_TILE.match(stem):
        return "TILE"
    if RE_CROP.match(stem):
        return "CROP"
    return "BOARD"


def board_id(stem):
    # Full-board id for disjoint splitting. TILE images are snapshots of full boards, distinguish using board number
    m = RE_TILE.match(stem)
    if m:
        return "tile_" + stem.split("__")[0]
    return stem

# some data augmentation
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


def write_rows(p, rows):
    p.write_text("\n".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                           for c, cx, cy, w, h in rows))


def find_pairs(src):
    src = Path(src)
    pairs = []
    for sp in ("train", "valid", "val", "test"):
        idir = src / sp / "images"
        if not idir.is_dir():
            continue
        for img in sorted(idir.iterdir()):
            if img.suffix.lower() in IMG_EXT:
                pairs.append((img, src / sp / "labels" / (img.stem + ".txt")))
    return pairs


def load_map(path):
    return {int(k): int(v) for k, v in json.loads(Path(path).read_text()).items() if not str(k).startswith("_")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle", required=True)
    ap.add_argument("--kaggle-map", required=True)
    ap.add_argument("--roboflow", required=True)
    ap.add_argument("--roboflow-map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--yolo-frac", type=float, default=0.60)
    ap.add_argument("--train-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=50)
    ap.add_argument("--no-clahe", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    for bucket in ("yolo", "train", "valid"):
        (out / bucket / "images").mkdir(parents=True, exist_ok=True)
        (out / bucket / "labels").mkdir(parents=True, exist_ok=True)

    # gather and group images
    items = []
    for src, mp, tag in [(a.kaggle, a.kaggle_map, "kaggle"), (a.roboflow, a.roboflow_map, "roboflow")]:
        cmap = load_map(mp)
        src = Path(src)
        pairs = []
        for sp in ("train", "valid", "val", "test"):
            idir = src / sp / "images"
            if not idir.is_dir():
                continue
            for img in sorted(idir.iterdir()):
                if img.suffix.lower() in IMG_EXT:
                    pairs.append((img, src / sp / "labels" / (img.stem + ".txt")))
        for img, lab in pairs:
            items.append([img, lab, cmap, tag, group_of(img.stem)])
        g = Counter(group_of(i[0].stem) for i in items if i[3] == tag)
        print(f"{tag}: {len(pairs)} images | {dict(g)}")

    # remove exact duplicates
    seen, kept, n_dupe = {}, [], 0
    for it in tqdm(items, desc="dedup(md5)"):
        try:
            h = hashlib.md5(it[0].read_bytes()).hexdigest()
        except Exception:
            kept.append(it)
            continue
        if h in seen:
            n_dupe += 1
        else:
            seen[h] = 1
            kept.append(it)
    items = kept
    print(f"exact duplicates removed: {n_dupe}, there are {len(items)} unique")

    # board-disjoint 3-way split (full boards only) to yolo, train, and valid
    # TILE + CROP always go to yolo. BOARD and Roboflow are split across buckets.
    board_ids = sorted({board_id(i[0].stem) for i in items if i[4] == "BOARD"})
    rng = np.random.default_rng(a.seed)
    rng.shuffle(board_ids)
    n = len(board_ids)
    n_yolo = int(round(n * a.yolo_frac))
    n_train = int(round(n * a.train_frac))
    board_bucket = {}
    for b in board_ids[:n_yolo]: board_bucket[b] = "yolo"
    for b in board_ids[n_yolo:n_yolo + n_train]: board_bucket[b] = "train"
    for b in board_ids[n_yolo + n_train:]: board_bucket[b] = "valid"


    # process the data and do preliminary data augmentation of clahe and rot90 that cannot be done by Ultralytics
    n_img = n_obj = n_drop = 0
    cls_hist = Counter()
    for img, lab, cmap, tag, group in tqdm(items, desc="write"):
        raw = read_rows(lab)
        rows = [(cmap[c], cx, cy, w, h) for c, cx, cy, w, h in raw if cmap.get(c, -1) >= 0]
        n_drop += len(raw) - len(rows)
        cls_hist.update(r[0] for r in rows)

        if group in ("TILE", "CROP"):
            bucket = "yolo"
        bucket = board_bucket.get(board_id(img.stem), "yolo")

        im = cv2.imread(str(img))
        if im is None:
            continue
        if not a.no_clahe:
            im = clahe(im)

        variants = [("", im, rows)]
        # rot90 for training buckets yolo and train only, not valid
        if bucket in ("yolo", "train"):
            variants.append(("_r90", np.ascontiguousarray(np.rot90(im)), rot90_labels(rows)))
        for suf, vim, vrows in variants:
            stem = f"{tag}_{img.stem}{suf}"
            cv2.imwrite(str(out / bucket / "images" / f"{stem}.jpg"), vim, [cv2.IMWRITE_JPEG_QUALITY, 95])
            write_rows(out / bucket / "labels" / f"{stem}.txt", vrows)
            n_img += 1
            n_obj += len(vrows)

    # create yaml file
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: yolo/images\nval: valid/images\n"
        f"nc: {len(CANON)}\nnames: {CANON}\n")

    # histogram showing the total components found in each class
    for c in range(len(CANON)):
        print(f"    {c:2d} {CANON[c]:>14s}: {cls_hist.get(c,0)}")
    empty = [CANON[i] for i in range(len(CANON)) if cls_hist.get(i,0) == 0]
    if empty:
        print(f"  ! ZERO instances: {empty}")


if __name__ == "__main__":
    main()
