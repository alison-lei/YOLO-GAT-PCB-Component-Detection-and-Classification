import os, random, subprocess, shutil
from pathlib import Path
import numpy as np
import torch

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("GPU:", torch.cuda.get_device_name(0))

DATA = Path("./data").resolve()
RUNS = Path("./yolo_finetune_run").resolve()
RUNS.mkdir(exist_ok=True)

CANON = ['battery','button','buzzer','capacitor','clock','connector','diode','display',
         'fuse','heatsink','ic','inductor','led','pads','pins','potentiometer','relay',
         'resistor','switch','transducer','transformer','transistor','unknown']

(DATA / "data.yaml").write_text(
    f"path: {DATA}\ntrain: yolo/images\nval: valid/images\nnc: {len(CANON)}\nnames: {CANON}\n"
)

for sp in ("yolo", "gat", "valid"):
    n = len(list((DATA / sp / "images").glob("*"))) if (DATA / sp / "images").is_dir() else 0
    print(f"  {sp}: {n} images")

train_cmd = [
    "yolo", "detect", "train",
    "model=yolo11m.pt",
    f"data={DATA/'data.yaml'}",
    "epochs=100", "imgsz=1280", "batch=16", "patience=40",
    "device=0", "workers=8", "cos_lr=True",
    "hsv_h=0.05", "hsv_s=0.7", "hsv_v=0.3",
    "degrees=10", "translate=0.1", "scale=0.5", "perspective=0.0005",
    "fliplr=0.5", "flipud=0.5",
    "mosaic=0.0", "mixup=0.0",
    "multi_scale=False", "cache=True", "seed=0",
    f"project={RUNS}", "name=pcb_yolo11m", "plots=True",
]
subprocess.run(train_cmd, check=True)

BEST = RUNS / "pcb_yolo11m" / "weights" / "best.pt"
shutil.copy(BEST, "./best.pt")
print("Copied best.pt to repo root:", BEST)

val_cmd = [
    "yolo", "detect", "val",
    "model=./best.pt",
    f"data={DATA/'data.yaml'}", "device=0", "imgsz=1280",
    "conf=0.15", "seed=0",
    f"project={RUNS}", "name=pcb_yolo11m_val", "plots=True",
]
subprocess.run(val_cmd, check=True)
print("Done. Results in:", RUNS / "pcb_yolo11m_val")