"""
wacv_tile_prep.py

pcb_wacv_2019/<board>/<board>.jpg + <board>.{xml,html} (Pascal VOC boxes)
  -> test_data/images/*.jpg (CLAHE'd tiles)
  -> test_data/labels/*.txt (YOLO, canonical class indices)

<name> field is "componenttype subid" (e.g. "pins unknown"); only the first
token is the class. Objects starting with "text" (silkscreen/labels) are
dropped -- not physical components.

pip install opencv-python numpy absl-py --break-system-packages
Usage: python wacv_tile_prep.py --input pcb_wacv_2019 --output test_data
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string("input", "pcb_wacv_2019", "Root folder: one subfolder per board")
flags.DEFINE_string("output", "data/test", "Output root; writes images/ and labels/")
flags.DEFINE_integer("tile_size", 1280, "Tile size in px")
flags.DEFINE_float("overlap", 0.2, "Fractional overlap between tiles")
flags.DEFINE_integer("min_boxes_per_tile", 6, "Skip tiles with fewer surviving boxes")
flags.DEFINE_float("min_visible_frac", 0.3, "Drop a box if less than this fraction survives clipping")
flags.DEFINE_float("clip_limit", 2.0, "CLAHE clip limit")
flags.DEFINE_integer("tile_grid", 8, "CLAHE tile grid size, NxN")
flags.DEFINE_boolean("no_crop", False, "Skip auto content-crop")

CANON = ["battery", "button", "buzzer", "capacitor", "clock", "connector", "diode",
         "display", "fuse", "heatsink", "ic", "inductor", "led", "pads", "pins",
         "potentiometer", "relay", "resistor", "switch", "transducer", "transformer",
         "transistor", "unknown"]
CANON_INDEX = {n: i for i, n in enumerate(CANON)}

# WACV first-token component name -> canonical name. Extend if new tokens show up.
WACV_TO_CANON = {
    "battery": "battery", "button": "button", "buzzer": "buzzer",
    "capacitor": "capacitor", "clock": "clock", "connector": "connector",
    "diode": "diode", "display": "display", "fuse": "fuse", "heatsink": "heatsink",
    "ic": "ic", "inductor": "inductor", "led": "led", "pads": "pads", "pins": "pins",
    "potentiometer": "potentiometer", "resistor": "resistor",
    "transformer": "transformer", "transistor": "transistor", "switch": "switch",
    "unknown": "unknown",
    "ferrite bead": "inductor", "resistor network": "resistor", "capacitor jumper": "capacitor",
    "jumper": "unknown", "test point": "pads", "resistor jumper": "resistor",
    "emi filter": "unknown", "zener diode": "unknown", "diode zener array": "diode",
    "electrolytic capacitor": "capacitor",
}


def extract_class(name):
    """
    Class is normally the FIRST token ('resistor R144 274' -> 'resistor',
    'ic IC11 "FTDI 09753.1 FT232HQ 1544-C"' -> 'ic', 'unknown SK2' -> 'unknown').
    The only exception: names starting with a quote hold a multi-word class
    phrase in that quote ('"emi filter" FIL9' -> 'emi filter').
    Everything after the class (id, free-text description, quoted or not) is
    ignored -- it's metadata, not needed for the class label.
    """
    name = name.strip()
    if name[:1] in ('"', "'"):
        q = name[0]
        end = name.find(q, 1)
        if end != -1:
            return name[1:end].strip().lower()
    return name.split()[0].strip('"').strip("'").lower()


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)


def detect_content_bbox(image, downsample=8, pad_frac=0.02):
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(1, w // downsample), max(1, h // downsample)))
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)
    dilated = cv2.dilate(edges, np.ones((15, 15), np.uint8), iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, w, h
    x, y, cw, ch = cv2.boundingRect(max(contours, key=cv2.contourArea))
    x, y, cw, ch = [v * downsample for v in (x, y, cw, ch)]
    px, py = int(cw * pad_frac), int(ch * pad_frac)
    return max(0, x - px), max(0, y - py), min(w, x + cw + px), min(h, y + ch + py)


def parse_voc(xml_path):
    """-> [(x, y, w, h, canon_class_id), ...]; skips 'text' objects and unmapped types."""
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        token = extract_class(name)
        if token == "text":
            continue
        canon_name = WACV_TO_CANON.get(token)
        if canon_name is None:
            if "component text" not in name:
                print(f"    WARNING: unmapped WACV class '{token}' (from '{name}') -- skipped")
            continue
        b = obj.find("bndbox")
        xmin, ymin = float(b.find("xmin").text), float(b.find("ymin").text)
        xmax, ymax = float(b.find("xmax").text), float(b.find("ymax").text)
        boxes.append((xmin, ymin, xmax - xmin, ymax - ymin, CANON_INDEX[canon_name]))
    return boxes


def clip_box(x, y, w, h, tile_size, min_visible_frac):
    x0, y0, x1, y1 = x, y, x + w, y + h
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, tile_size), min(y1, tile_size)
    if cx1 <= cx0 or cy1 <= cy0 or w * h <= 0:
        return None
    if ((cx1 - cx0) * (cy1 - cy0)) / (w * h) < min_visible_frac:
        return None
    return cx0, cy0, cx1 - cx0, cy1 - cy0


def find_board_files(board_dir, name):
    jpg = board_dir / f"{name}.jpg"
    ann = None
    for ext in (".xml", ".html"):
        p = board_dir / f"{name}{ext}"
        if p.exists():
            ann = p
            break
    return jpg, ann


def main(argv):
    del argv
    input_path, output_path = Path(FLAGS.input), Path(FLAGS.output)
    images_dir, labels_dir = output_path / "images", output_path / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    board_dirs = sorted(d for d in input_path.iterdir() if d.is_dir())
    print(f"Found {len(board_dirs)} board folders")
    total_tiles = 0

    for board_dir in board_dirs:
        name = board_dir.name
        jpg_path, ann_path = find_board_files(board_dir, name)
        if not jpg_path.exists() or ann_path is None:
            print(f"  SKIPPING {name}: missing jpg or annotation file")
            continue

        img = cv2.cvtColor(cv2.imread(str(jpg_path)), cv2.COLOR_BGR2RGB)
        boxes = parse_voc(ann_path)
        print(f"  {name}: {img.shape[1]}x{img.shape[0]}, {len(boxes)} components")

        cx0, cy0 = 0, 0
        if not FLAGS.no_crop:
            x0, y0, x1, y1 = detect_content_bbox(img)
            img = img[y0:y1, x0:x1]
            cx0, cy0 = x0, y0
        img = apply_clahe(img, FLAGS.clip_limit, (FLAGS.tile_grid, FLAGS.tile_grid))
        boxes = [(x - cx0, y - cy0, w, h, c) for x, y, w, h, c in boxes]

        h, w = img.shape[:2]
        ts, stride = FLAGS.tile_size, max(1, int(FLAGS.tile_size * (1 - FLAGS.overlap)))
        y = 0
        while True:
            y_end = min(y + ts, h)
            x = 0
            while True:
                x_end = min(x + ts, w)
                tile = img[y:y_end, x:x_end]
                th, tw = tile.shape[:2]
                if th < ts or tw < ts:
                    padded = np.zeros((ts, ts, 3), dtype=img.dtype)
                    padded[:th, :tw] = tile
                    tile = padded

                tile_boxes = []
                for bx, by, bw, bh, cls in boxes:
                    clipped = clip_box(bx - x, by - y, bw, bh, ts, FLAGS.min_visible_frac)
                    if clipped is not None:
                        tile_boxes.append((*clipped, cls))

                if len(tile_boxes) >= FLAGS.min_boxes_per_tile:
                    stem = f"{name}_x{x}_y{y}"
                    cv2.imwrite(str(images_dir / f"{stem}.jpg"),
                                cv2.cvtColor(tile, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    lines = [f"{cls} {(bx+bw/2)/ts:.6f} {(by+bh/2)/ts:.6f} {bw/ts:.6f} {bh/ts:.6f}"
                             for bx, by, bw, bh, cls in tile_boxes]
                    (labels_dir / f"{stem}.txt").write_text("\n".join(lines))
                    total_tiles += 1

                if x_end >= w:
                    break
                x += stride
            if y_end >= h:
                break
            y += stride

    with open(output_path / "data.yaml", "w") as f:
        f.write(f"nc: {len(CANON)}\nnames:\n")
        for n in CANON:
            f.write(f"  - {n}\n")
    print(f"\nDone. Wrote {total_tiles} tiles (>= {FLAGS.min_boxes_per_tile} boxes each) to {output_path}/")


if __name__ == "__main__":
    app.run(main)