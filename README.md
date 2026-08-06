# YOLO-GAT-PCB-Component-Detection-and-Classification

This project investigates how spatial and geometric information, presented as a graph attention network (GAT), given to a fine tuned YOLOv11m model (YOLO) improves its classification accuracy and ability to reject false positives (GAT corrects the YOLO model on its false detection of components by reclassifying them as background). This investigation is motivated by the notion that PCBs can be interpreted as graphs as the relative locations of components are logically placed to serve a purpose, and hence, follow a trend and can be predicted by a neural network.

### Set Up venv
Create a venv by running `python -m venv .venv`. In PowerShell, run `.venv\Scripts\Activate.ps1` to activate the environment and then select the appropriate python version to use inside the venv. Python=3.10.0 was used while runing the experiments.

### Install Dependencies
Run `pip install -r requirements.txt`.

## Obtain Datasets
The training and validation data used is a combination of the following two datasets [Kaggle](https://www.kaggle.com/datasets/aryanstein/pcb-component-detection-consolidated-dataset/data) and [Roboflow](https://universe.roboflow.com/luizf/printed-circuit-board-lafr6/dataset/1). After preprocessing, the final data used from the above datasets is split into 60% for fine tuning the YOLO model, 25% for training the GAT, and 15% for validating the YOLO+GAT model combined. The testing data comes from the PCB-WACV dataset, [PCB-WACV](https://sites.google.com/view/chiawen-kuo/home/pcb-component-detection), and it features completely new images of full PCB boards that neither the YOLO nor GAT model has seen before. Processing and data augmentation is done on the training, validation, and testing datasets.

If you want to pre-process the data yourself from the original datasets, to generate the YOLO, train, and valid datsets, download the Kaggle and Roboflow datasets and store them locally as `datasets/kaggle_dataset` and `datasets/roboflow_dataset` respectively and run the `python utils/build_dataset.py` script. To generate the test dataset, download the PCB-WACV dataset and store it locally as `pcb_wacv_2019` in the project root. Then run `python utils/wacv_tile_prep.py`.


The already processed data used for the YOLO-GAT PCB component classification model can be downloaded from [YOLO-GAT PCB Component Classification Dataset](https://www.kaggle.com/datasets/projectiscool/yolo-gat-pcb-component-classification-dataset). Store it locally.


## Replicate Results
The experiments were executed by an AMD Ryzen 9 5900X 12-Core Processor and NVIDIA Quadro RTX 8000 Graphics Card.

In the following order, 
1. Run `python utils/train_yolo11m.py` with a dedicated GPU. The best model should be saved as `best.pt` in the root of your local repository.
2. For graph generation:
```
python utils/build_graphs.py --weights=best.pt --root=data --split=train --out=graphs/train.pt
python utils/build_graphs.py --weights=best.pt --root=data --split=valid --out=graphs/valid.pt
python utils/build_graphs.py --weights=best.pt --root=data --split=test --out=graphs/test.pt
```
3. For training:
```
python utils/train_gat.py --train=graphs/train.pt --val=graphs/valid.pt --names=data/data.yaml
```
4. For evaluation:
```
python utils/eval_gat.py --checkpoint=train_results/gat_bg0.7_best.pt --graphs=graphs/test.pt
```

Note that `utils/build_dataset.py` contains the script to process the original [Kaggle](https://www.kaggle.com/datasets/aryanstein/pcb-component-detection-consolidated-dataset/data) and [Roboflow](https://universe.roboflow.com/luizf/printed-circuit-board-lafr6/dataset/1) datasets, and `utils/wacv_tile_prep.py` contains the script to process [PCB-WACV](https://sites.google.com/view/chiawen-kuo/home/pcb-component-detection) dataset into the format it is found in [YOLO-GAT PCB Component Classification Dataset](https://www.kaggle.com/datasets/projectiscool/yolo-gat-pcb-component-classification-dataset).