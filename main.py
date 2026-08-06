"""
To reproduce the results:
1. Run `python utils/train_yolo11m.py` with a dedicated GPU. The best model should be saved as `best.pt` in the root of your local repository.

2. For graph generation:
python utils/build_graphs.py --weights=best.pt --root=data --split=train --out=graphs/train.pt
python utils/build_graphs.py --weights=best.pt --root=data --split=valid --out=graphs/valid.pt
python utils/build_graphs.py --weights=best.pt --root=data --split=test --out=graphs/test.pt

3. For training:
python utils/train_gat.py --train=graphs/train.pt --val=graphs/valid.pt --names=data/data.yaml

4. For evaluation:
python utils/eval_gat.py --checkpoint=train_results/gat_best.pt --graphs=graphs/test.pt

"""