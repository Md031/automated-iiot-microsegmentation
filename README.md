# Master Thesis
This repository contains the implementation used for the master thesis:

**Securing IIoT Environments: Preventing Lateral Movement through Automated Micro-Segmentation**

The thesis studies the automated micro-segmentation approach proposed by Arifeen et al. and evaluates its behavior on larger network intrusion datasets. The original work was limited to a small sample of network flows. This project therefore re-implements the main pipeline and tests it on IoT-23, IoTID20, and UNSW-NB15 to analyze whether the approach remains practical and reliable at a larger scale.

## 1. Installation
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirement.txt
```

## 2. Download and extract the datasets
1. [IoTID20](https://www.kaggle.com/datasets/rohulaminlabid/iotid20-dataset/data): archive size ~60 MB
2. [UNSW-NB15](https://www.kaggle.com/datasets/dhoogla/unswnb15): archive size ~160 MB
3. [IoT23](https://www.stratosphereips.org/datasets-iot23) (Note: Download the lighter version of the dataset): archive size ~9 GB

### 2.1. Dataset structure
To run the pipeline, we recommend following the structure described below :
```
├── Dataset/           	# Place your extracted datasets here 
│	├── iotId20 		# Contains the iotId20 dataset
│	│	├── IoT Network Intrusion Dataset.csv       
│   ├── UNSW_NB15 		# Contains the UNSW-NB15 dataset
│	│	├── UNSW-NB15_1.csv
│	│	├── UNSW-NB15_2.csv
│	│	├── UNSW-NB15_3.csv
│	│	├── UNSW-NB15_4.csv
│   ├── iot23_small 	# Contains the IoT23 dataset
│	│	├── opt
│	│	│	├── ... / conn.log.labeled
├── scripts/			# Python source code
| README.md
```
Otherwise, you can modify the parameters `<iotid20_path>`, `<unsw_path>`, and `<iot23_path>` in the file [config.py](scripts/config.py) to point to your custom directory.

## 3. Prepare data for ML
The pipeline must be executed in order.

### 3.1. Preprocessing & PCA
```
make run DATASET=<dataset>
```

### 3.2. OPTICS clustering
```
make optics DATASET=<dataset>
```

### 3.3. Decision Tree Classification
```
make dt DATASET=<dataset>
```

The following command can be used to run the full pipeline from the beginning in order.

```
make start DATASET=<dataset>
```

## 4. Additional experiments

The file [enriched_flows.py](scripts/enriched_flows.py) contains the different mitigation strategies presented on the thesis: training with feature enrichment, training with feature enrichment + class weighting. This experiment is defined for IoT-23 specifically.

The default strategy used is the feature enrichment strategy.

```
python scripts/enriched_flows.py
```