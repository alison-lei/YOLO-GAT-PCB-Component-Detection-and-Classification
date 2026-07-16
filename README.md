# YOLO-GAT-PCB-Component-Detection-and-Classification

This project investigates how spatial and geometric information, presented as a graph attention network (GAT), given to a fine tuned YOLOv11m model (YOLO) improves its classification accuracy and ability to reject false positives (GAT corrects the YOLO model on its false detection of components by reclassifying them as background). This investigation is motivated by the notion that PCBs can be interpreted as graphs as the relative locations of components to each other follow a logical pattern.

### Set Up venv
Create a venv by running `python -m venv .venv`. In PowerShell, run `.venv\Scripts\Activate.ps1` to activate the environment and then select the appropriate python version to use inside the venv. Python=3.10.0 was used while runing the experiments.

### Install Dependencies
Run `pip install -r requirements.txt`.

## Obtain Datasets
The data used is a combination of the following two datasets [Kaggle](https://www.kaggle.com/datasets/aryanstein/pcb-component-detection-consolidated-dataset/data) and [Roboflow](https://universe.roboflow.com/luizf/printed-circuit-board-lafr6/dataset/1). After preprocessing, the final data used from the above datasets is split into 60% for fine tuning the YOLO model, 25% for training the GAT, and 15% for validating the YOLO+GAT model combined. There is also a separate FPIC-C dataset, [FPIC-C](https://physicaldb.ece.ufl.edu/index.php/fics-pcb-image-collection-fpic/), that will be used as the testing dataset as neither the YOLO nor GAT model has seen it before.


The data used for the YOLO-GAT PCB component classification model can be downloaded from [YOLO-GAT PCB Component Classification Dataset](https://www.kaggle.com/datasets/projectiscool/yolo-gat-pcb-component-classification-dataset). Store it locally.


## Replicate Results
In the following order, 
1. Copy and paste the contents of `utils/yolo_fine_tuning.txt` into a code cell in a Kaggle notebook and fine tune the YOLOv11m model.
2. Save the best model (`weights/best.pt`) as a file called `best.pt` into the root of your local repository.
3. Run:
```
python utils/build_graphs.py --weights best.pt --root data --split train --out graphs/train.pt
python utils/build_graphs.py --weights best.pt --root data --split valid --out graphs/valid.pt
python utils/train_gat.py --train graphs/train.pt --val graphs/valid.pt --names data/data.yaml --epochs 200 --tag gat_edge
```
Note that `utils/build_dataset.py` contains the script to process the original [Kaggle](https://www.kaggle.com/datasets/aryanstein/pcb-component-detection-consolidated-dataset/data) and [Roboflow](https://universe.roboflow.com/luizf/printed-circuit-board-lafr6/dataset/1) datasets into the format it is found in [YOLO-GAT PCB Component Classification Dataset](https://www.kaggle.com/datasets/projectiscool/yolo-gat-pcb-component-classification-dataset).