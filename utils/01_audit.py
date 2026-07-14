"""
01_audit.py — Run this FIRST, before touching a model.

Answers the three questions that decide whether the project works:
  1. How many components per image? (< ~5 median => graph is useless for that dataset)
  2. How imbalanced are the classes?
  3. Do the datasets overlap? (leakage kills every number you report)

Usage:
    pip install ultralytics imagehash pillow matplotlib numpy tqdm
    python 01_audit.py --roots kaggle=/data/kaggle roboflow=/data/roboflow fpic=/data/fpic_yolo

Each root must be YOLO layout:
    root/{train,valid,test}/images/*.jpg
    root/{train,valid,test}/labels/*.txt      # cls cx cy w h  (normalized)
    root/data.yaml                            # optional, for class names
"""
import argparse, json, os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ["train", "valid", "val", "test"]


def load_names(root: Path):
    y = root / "data.yaml"
    if not y.exists():
        return None
    names = {}
    txt = y.read_text()
    # tolerant mini-parser: handles both list and dict `names:` forms
    if "names:" not in txt:
        return None
    body = txt.split("names:", 1)[1]
    if body.lstrip().startswith("["):
        lst = body[body.index("[") + 1: body.index("]")]
        for i, n in enumerate(x.strip().strip("'\"") for x in lst.split(",")):
            names[i] = n
    else:
        for line in body.splitlines()[1:]:
            s = line.strip()
            if not s or ":" not in s or not s[0].isdigit():
                if names:
                    break
                continue
            k, v = s.split(":", 1)
            names[int(k)] = v.strip().strip("'\"")
    return names or None


def scan(root: Path):
    """Return per-image records: {split, img, n_objs, classes, rel_box_areas, wh}."""
    recs = []
    for sp in SPLITS:
        idir, ldir = root / sp / "images", root / sp / "labels"
        if not idir.is_dir():
            continue
        for img in sorted(p for p in idir.iterdir() if p.suffix.lower() in IMG_EXT):
            lab = ldir / (img.stem + ".txt")
            cls, areas = [], []
            if lab.exists():
                for line in lab.read_text().splitlines():
                    f = line.split()
                    if len(f) < 5:
                        continue
                    cls.append(int(float(f[0])))
                    areas.append(float(f[3]) * float(f[4]))  # normalized w*h
            try:
                wh = Image.open(img).size
            except Exception:
                wh = (0, 0)
            recs.append(dict(split=sp, img=str(img), n=len(cls),
                             classes=cls, areas=areas, wh=wh))
    return recs


def phash_all(recs, size=8):
    import imagehash
    out = {}
    for r in tqdm(recs, desc="  hashing", leave=False):
        try:
            out[r["img"]] = str(imagehash.phash(Image.open(r["img"]), hash_size=size))
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="name=/path ...")
    ap.add_argument("--out", default="audit")
    ap.add_argument("--skip-hash", action="store_true")
    a = ap.parse_args()
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)

    datasets, hashes, summary = {}, {}, {}
    for spec in a.roots:
        name, path = spec.split("=", 1)
        root = Path(path)
        print(f"\n=== {name}  ({root}) ===")
        recs = scan(root)
        if not recs:
            print("  !! no images found — check the layout"); continue
        datasets[name] = recs
        names = load_names(root)

        counts = np.array([r["n"] for r in recs])
        allcls = Counter(c for r in recs for c in r["classes"])
        areas = np.array([x for r in recs for x in r["areas"]])
        res = Counter(r["wh"] for r in recs)

        # ---- THE decisive statistic for a graph project ----
        print(f"  images            : {len(recs)}  ({Counter(r['split'] for r in recs)})")
        print(f"  objects           : {int(counts.sum())}")
        print(f"  objs/image        : mean {counts.mean():.1f} | median {np.median(counts):.0f} "
              f"| p10 {np.percentile(counts,10):.0f} | p90 {np.percentile(counts,90):.0f}")
        print(f"  images with <5 obj: {(counts < 5).mean()*100:.1f}%   <-- graph-hostile fraction")
        print(f"  empty images      : {(counts == 0).mean()*100:.1f}%")
        print(f"  classes present   : {len(allcls)}")
        print(f"  median box area   : {np.median(areas)*100:.3f}% of image "
              f"(=> ~{np.sqrt(np.median(areas))*1280:.0f}px at imgsz=1280)")
        print(f"  top resolutions   : {res.most_common(3)}")
        top = allcls.most_common(5)
        print("  top classes       : " + ", ".join(
            f"{(names or {}).get(c, c)}={n}" for c, n in top))
        if allcls:
            imb = max(allcls.values()) / max(1, min(allcls.values()))
            print(f"  imbalance ratio   : {imb:.0f}:1 (most:least frequent)")

        summary[name] = dict(images=len(recs), objects=int(counts.sum()),
                             mean_objs=float(counts.mean()),
                             median_objs=float(np.median(counts)),
                             frac_lt5=float((counts < 5).mean()),
                             classes=len(allcls),
                             class_counts={str(k): v for k, v in allcls.items()})

        # ---- report figures ----
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        ax[0].hist(counts, bins=min(40, max(5, counts.max())), color="#3b6ea5")
        ax[0].axvline(5, ls="--", c="crimson", label="graph viability floor")
        ax[0].set_xlabel("components per image"); ax[0].set_ylabel("images")
        ax[0].set_title(f"{name}: component density"); ax[0].legend()
        ks = [k for k, _ in allcls.most_common()]
        ax[1].bar(range(len(ks)), [allcls[k] for k in ks], color="#a5563b")
        ax[1].set_xticks(range(len(ks)))
        ax[1].set_xticklabels([str((names or {}).get(k, k)) for k in ks],
                              rotation=90, fontsize=6)
        ax[1].set_yscale("log"); ax[1].set_ylabel("instances (log)")
        ax[1].set_title(f"{name}: class balance")
        fig.tight_layout(); fig.savefig(outdir / f"{name}_stats.png", dpi=160)
        plt.close(fig)

        if not a.skip_hash:
            hashes[name] = phash_all(recs)

    # ---- leakage check: the thing that quietly ruins the project ----
    if len(hashes) > 1:
        print("\n=== DUPLICATE / LEAKAGE CHECK (perceptual hash) ===")
        names_ = list(hashes)
        for i in range(len(names_)):
            for j in range(i + 1, len(names_)):
                A, B = names_[i], names_[j]
                inv = defaultdict(list)
                for p, h in hashes[B].items():
                    inv[h].append(p)
                dup = [(p, inv[h]) for p, h in hashes[A].items() if h in inv]
                pct = 100 * len(dup) / max(1, len(hashes[A]))
                flag = "  <<< MERGE WITH CARE" if pct > 1 else ""
                print(f"  {A} ∩ {B}: {len(dup)} exact-phash matches ({pct:.1f}% of {A}){flag}")
                if dup:
                    with open(outdir / f"dupes_{A}_{B}.txt", "w") as f:
                        for p, q in dup:
                            f.write(f"{p}\t{q[0]}\n")
        # within-dataset train/test leakage
        for n, h in hashes.items():
            bysplit = defaultdict(dict)
            for r in datasets[n]:
                if r["img"] in h:
                    bysplit[r["split"]][h[r["img"]]] = r["img"]
            tr = set(bysplit.get("train", {}))
            te = set(bysplit.get("test", {})) | set(bysplit.get("valid", {}))
            if tr & te:
                print(f"  !! {n}: {len(tr & te)} images appear in BOTH train and val/test")

    (Path(a.out) / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote figures + summary.json to {outdir}/")
    print("\nDECISION RULE: any dataset with median objs/image < 5 goes in the "
          "detector-training pool ONLY — it cannot support the graph stage.")


if __name__ == "__main__":
    main()
