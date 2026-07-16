"""
12_split3.py — Split merged_new images into 3 board-disjoint sets:
  yolo   : fine-tune YOLO here
  gat    : YOLO predicts here (never trained on) -> honest errors for GAT
  val    : final YOLO+GAT evaluation

CRITICAL: all baked rotations of one board (stem, stem_r90, ...) go to the SAME
set, or a rotated copy leaks across splits. Board id = stem with the source
prefix and any _r## suffix stripped.

Creates a new folder tree with symlinks (fast, no copy) or copies.

Usage:
    python 12_split3.py --root merged_new --out split3 \
        --yolo-frac 0.6 --gat-frac 0.25   (val gets the rest)
"""
import argparse, re, shutil
from pathlib import Path

import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
RE_ROT = re.compile(r"_r\d+$")


def board_id(stem):
    """kaggle_PCBA_17_r90 -> PCBA_17 ; roboflow_IMG_2841 -> IMG_2841."""
    s = RE_ROT.sub("", stem)                 # drop _r90 etc.
    for pre in ("kaggle_", "roboflow_", "fpic_"):
        if s.startswith(pre):
            s = s[len(pre):]
    return s


def collect(root):
    """Return {board_id: [(img_path, lab_path), ...]} across all source splits."""
    boards = {}
    for sp in ("train", "valid", "val", "test"):
        idir = root / sp / "images"
        if not idir.is_dir():
            continue
        for img in sorted(idir.iterdir()):
            if img.suffix.lower() not in IMG_EXT:
                continue
            lab = root / sp / "labels" / (img.stem + ".txt")
            boards.setdefault(board_id(img.stem), []).append((img, lab))
    return boards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--yolo-frac", type=float, default=0.60)
    ap.add_argument("--gat-frac", type=float, default=0.25)   # val = 1 - yolo - gat
    ap.add_argument("--seed", type=int, default=50)
    ap.add_argument("--copy", action="store_true", help="copy instead of symlink")
    a = ap.parse_args()

    root = Path(a.root)
    boards = collect(root)
    ids = sorted(boards)
    rng = np.random.default_rng(a.seed)
    rng.shuffle(ids)

    n = len(ids)
    n_yolo = int(round(n * a.yolo_frac))
    n_gat = int(round(n * a.gat_frac))
    buckets = {
        "yolo": ids[:n_yolo],
        "gat":  ids[n_yolo:n_yolo + n_gat],
        "val":  ids[n_yolo + n_gat:],
    }

    out = Path(a.out)
    for name, bids in buckets.items():
        (out / name / "images").mkdir(parents=True, exist_ok=True)
        (out / name / "labels").mkdir(parents=True, exist_ok=True)
        n_img = 0
        for bid in bids:
            for img, lab in boards[bid]:
                di = out / name / "images" / img.name
                dl = out / name / "labels" / (img.stem + ".txt")
                if a.copy:
                    shutil.copy(img, di)
                    if lab.exists(): shutil.copy(lab, dl)
                else:
                    if not di.exists(): di.symlink_to(img.resolve())
                    if lab.exists() and not dl.exists(): dl.symlink_to(lab.resolve())
                n_img += 1
        print(f"  {name:4s}: {len(bids):4d} boards | {n_img:5d} images")

    # YOLO needs a data.yaml pointing yolo->train, val->val
    CANON = ['battery','button','buzzer','capacitor','clock','connector','diode',
             'display','fuse','heatsink','ic','inductor','led','pads','pins',
             'potentiometer','relay','resistor','switch','transducer','transformer',
             'transistor','unknown']
    (out / "yolo_data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: yolo/images\nval: val/images\n"
        f"nc: {len(CANON)}\nnames: {CANON}\n")

    print(f"\n{n} boards total -> {n_yolo} yolo / {n_gat} gat / {n-n_yolo-n_gat} val")
    print(f"-> {out}/  (yolo_data.yaml for retraining)")
    print("all rotations of a board stay in one bucket: no leakage.")


if __name__ == "__main__":
    main()
