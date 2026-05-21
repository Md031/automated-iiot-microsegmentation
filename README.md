# Master-Thesis

## 1. Usage
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirement.txt
```

## 2. Download and extract the datasets
1. [IoTID20](https://www.kaggle.com/datasets/rohulaminlabid/iotid20-dataset/data): archive size ~60 mb
2. [UNSW-NB15](https://www.kaggle.com/datasets/dhoogla/unswnb15): archise size ~160 mb
3. [IoT23](https://www.stratosphereips.org/datasets-iot23) (Note: Download the lighter version of the dataset): archise size ~9 GB

### 2.1. Dataset structure
To run the pipeline, we recommend you to follow the structure described below :
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
Otherwise, you can modify the parameters `<iotid20_path>`, `<unsw_path>`, and `<iot23_path>` on the file [config.py](scripts/config.py) to point to your custom directory.

## 3. Prepare data for ML
### 3.1. Preprocessing & PCA
1. Run the main script and specify the dataset you want to process (**unsw**, **iotid20**, **iot23**)

```
make run DATASET=<dataset>et>
```

### 3.2. OPTICS clustering
```
make optics DATASET=<dataset>
```

### 3.3. Decision Tree Classification
```
make dt DATASET=<dataset>
```