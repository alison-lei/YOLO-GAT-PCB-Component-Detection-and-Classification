"""
In terminal, activate .venv and run the following commands in order
python utils/build_graphs.py --weights best.pt --root data --split train --out graphs/train.pt
python utils/build_graphs.py --weights best.pt --root data --split valid --out graphs/valid.pt
python utils/train_gat.py --train graphs/train.pt --val graphs/valid.pt --names data/data.yaml --epochs 200 --tag gat_edge

"""