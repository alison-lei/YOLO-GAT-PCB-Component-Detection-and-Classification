"""
06_triage.py — The Kaggle "consolidated" set is three datasets in a trenchcoat.
Sort them out, kill duplicates, and emit a leak-free board-level split.

Groups:
  TILE  : 00025__1024__1648___0        FPIC board scans, tiled
  CROP  : battery2, inductor29         one component, black background
  BOARD : PCBA_17, ArduinoMega_Top     whole PCB photographs

Usage:
    pip install imagehash pillow tqdm numpy
    python 06_triage.py --root /data/kaggle --out triage/

Outputs:
    triage/manifest.json   every image: group, board_id, n_objects, dupe_of
    triage/split.json      board-level train/valid/test assignment
    triage/report.txt      what you actually have
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse, hashlib, json, re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
RE_TILE = re.compile(r"^(?P<board>\d+)__(?P<tile>\d+)__(?P<x>\d+)___(?P<y>\d+)$")
RE_CROP = re.compile(r"^(?P<cls>[a-zA-Z]+)(?P<idx>\d+)$")   # battery2, inductor29


def classify(stem):
    m = RE_TILE.match(stem)
    if m:
        return "TILE", m["board"], int(m["x"]), int(m["y"]), int(m["tile"])
    m = RE_CROP.match(stem)
    if m:
        # a whole-board name like "PCBA_17" has an underscore, so it won't match
        return "CROP", f"crop:{m['cls'].lower()}", None, None, None
    return "BOARD", f"board:{stem}", None, None, None


def n_objs(lab):
    if not lab.exists():
        return 0
    return sum(1 for l in lab.read_text().splitlines() if len(l.split()) >= 5)


def main():
    from config import CONFIG

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="triage")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    import imagehash

    recs = []
    for sp in ("train", "valid", "test"):
        idir = Path(a.root) / sp / "images"
        if not idir.is_dir():
            continue
        for img in sorted(p for p in idir.iterdir() if p.suffix.lower() in IMG_EXT):
            grp, board, x, y, T = classify(img.stem)
            lab = Path(a.root) / sp / "labels" / (img.stem + ".txt")
            try:
                W, H = Image.open(img).size
            except Exception:
                continue
            recs.append(dict(path=str(img), stem=img.stem, split=sp, group=grp,
                             board=board, x=x, y=y, tile=T, W=W, H=H,
                             n=n_objs(lab)))

    # ---------------- duplicates ----------------
    # exact: md5 of bytes. near: perceptual hash (catches recompression/resize).
    md5, ph = {}, {}
    for r in tqdm(recs, desc="hashing"):
        p = Path(r["path"])
        md5[r["stem"]] = hashlib.md5(p.read_bytes()).hexdigest()
        try:
            ph[r["stem"]] = str(imagehash.phash(Image.open(p)))
        except Exception:
            ph[r["stem"]] = None

    seen_md5, seen_ph, n_exact, n_near = {}, {}, 0, 0
    for r in recs:
        s = r["stem"]
        r["dupe_of"] = None
        if md5[s] in seen_md5:
            r["dupe_of"] = seen_md5[md5[s]]; r["dupe_kind"] = "exact"; n_exact += 1
            print(seen_md5[md5[s]])
        elif ph[s] and ph[s] in seen_ph:
            r["dupe_of"] = seen_ph[ph[s]]; r["dupe_kind"] = "near"; n_near += 1
        else:
            seen_md5[md5[s]] = s
            if ph[s]:
                seen_ph[ph[s]] = s

    # cross-split duplicates = leakage in the ORIGINAL split
    split_of = {r["stem"]: r["split"] for r in recs}
    cross = [r for r in recs if r["dupe_of"] and
             split_of[r["dupe_of"]] != r["split"]]

    keep = [r for r in recs if not r["dupe_of"]]

    # ---------------- report ----------------
    L = []
    L.append(f"total images         : {len(recs)}")
    L.append(f"exact duplicates     : {n_exact}")
    L.append(f"near duplicates      : {n_near}")
    L.append(f"cross-SPLIT dupes    : {len(cross)}   <-- leakage in the shipped split")
    L.append(f"unique images kept   : {len(keep)}\n")

    for g in ("TILE", "CROP", "BOARD"):
        rs = [r for r in keep if r["group"] == g]
        if not rs:
            continue
        n = np.array([r["n"] for r in rs])
        boards = {r["board"] for r in rs}
        res = Counter((r["W"], r["H"]) for r in rs)
        L.append(f"[{g}]  {len(rs)} images | {len(boards)} boards | "
                 f"{int(n.sum())} objects")
        L.append(f"       objs/image: mean {n.mean():.1f} median {np.median(n):.0f} "
                 f"p90 {np.percentile(n,90):.0f} | empty {int((n==0).sum())}")
        L.append(f"       resolutions: {res.most_common(3)}")
        L.append(f"       GRAPH-VIABLE: {'YES' if np.median(n) >= 8 else 'NO'}")
        L.append("")

    # reassembled TILE boards: how many components once merged?
    tb = defaultdict(int)
    for r in keep:
        if r["group"] == "TILE":
            tb[r["board"]] += r["n"]
    if tb:
        v = np.array(list(tb.values()))
        L.append(f"[TILE reassembled] {len(tb)} boards | components/board (pre-dedupe): "
                 f"median {int(np.median(v))} max {int(v.max())}")
        L.append("")

    report = "\n".join(L)
    print("\n" + report)
    (out / "report.txt").write_text(report)

    # ---------------- board-level split ----------------
    # CROPs get split randomly (they are independent single-part photos).
    # TILEs and BOARDs get split by board id, so no board straddles the line.
    rng = np.random.default_rng(a.seed)
    split = {"train": [], "valid": [], "test": []}
    for group in ("TILE", "BOARD", "CROP"):
        ids = sorted({r["board"] for r in keep if r["group"] == group})
        rng.shuffle(ids)
        n = len(ids)
        n_te = max(1, int(n * a.test_frac))
        n_va = max(1, int(n * a.val_frac))
        split["test"] += ids[:n_te]
        split["valid"] += ids[n_te:n_te + n_va]
        split["train"] += ids[n_te + n_va:]

    (out / "split.json").write_text(json.dumps(split, indent=2))
    (out / "manifest.json").write_text(json.dumps(keep, indent=2))

    print(f"split: {len(split['train'])} train / {len(split['valid'])} valid / "
          f"{len(split['test'])} test  (units = boards, not images)")
    print(f"\n-> {out}/report.txt, split.json, manifest.json")
    print("\nRULES FROM HERE:")
    print("  * CROP images  -> YOLO training ONLY. Never build a graph from them.")
    print("  * TILE images  -> YOLO as-is; reassemble labels to board coords for GAT.")
    print("  * BOARD images -> YOLO as-is; use directly as GAT graphs.")


if __name__ == "__main__":
    main()
