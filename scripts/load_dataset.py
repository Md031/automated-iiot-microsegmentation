from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.sql.types import StructType
from preprocessing import preprocessing, get_num_columns
from pyspark.sql.functions import monotonically_increasing_id
from config import DatasetConfig
import os
import json

def create_spark_session(app_name: str = "IoT_Thesis"):
	spark = ( # session for an Acer Aspire 5 computer
		SparkSession.builder
		.appName(app_name)
		.master("local[4]") # Use all available CPU cores
		.config("spark.driver.memory", "8g") # RAM
		.config("spark.driver.maxResultSize", "4g")
		.config("spark.local.dir", "/tmp/spark-spill")  # stays in WSL filesystem, faster + isolated
		.config("spark.sql.shuffle.partitions", "200")
		.config("spark.sql.adaptive.enabled", "true")
		.config("spark.sql.adaptive.coalescePartitions.enabled", "true")
		.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
		.config("spark.kryoserializer.buffer.max", "512m")       
		.config("spark.sql.execution.arrow.pyspark.enabled", "true") # optimization for OPTICS conversion from pyspark to numpy
		.getOrCreate()
	)
	return spark

def get_path(dataset_choice: str):
	""" Return dataset path corresponding to user's choice """
	if dataset_choice == "iot23":
		return iot23_path
	elif dataset_choice == "iotid20":
		return iotid20_path
	elif dataset_choice == "unsw":
		return unsw_path

def load_unsw(spark: SparkSession, config: DatasetConfig):
	""" Load the UNSW-NB15 dataset with the correct features name/type """
	file_pattern = f"{config.path}/UNSW-NB15_[1-4].csv"
	df_raw = spark.read.csv(
		file_pattern, 
		header = False, 
		schema = config.schema,
		nullValue = '-',
		nanValue='NaN'
	)
	# UNSW doesn't have an uid column
	df_raw = df_raw.withColumn("uid", monotonically_increasing_id())
	return df_raw

def save_data(df: DataFrame, path: str):
	""" Save the df into parquet format for faster access when reused """
	df.write.parquet(path, mode="overwrite")
	print(f"[DEBUG] Processed data save at {path}")

def load_data(spark: SparkSession, path: str) -> DataFrame:
	"""
	Function used when the data has already been pre-processed before. 
	Load the parquet of the data to be faster
	"""
	return spark.read.parquet(path)

def load_iotid20(spark: SparkSession, config: DatasetConfig):
	""" Load the IoTID20 dataset """
	df_raw = spark.read.csv(
		f"{config.path}/IoT Network Intrusion Dataset.csv", 
		header = True, 
		schema = config.schema,
		nullValue = '-',
		nanValue='NaN'
	)
	df_raw = df_raw.withColumn("uid", monotonically_increasing_id())
	# 2. Binary Mapping: Anomaly -> 1, Normal -> 0
	# We use .cast("int") to make sure it's not stored as a string "1" or "0"
	df_raw = cast_label_to_int(df_raw, "Label", "Anomaly")
	return df_raw

def cast_label_to_int(df: DataFrame, label_name: str, anomaly_name: str):
	df_casted = df.withColumn(label_name, 
		F.when(F.col(label_name) == anomaly_name, 1)
		 .otherwise(0)
		 .cast("int")
	)
	return df_casted

def get_all_conn_files(dataset_dir: str) -> list[str]:
	""" 
	Walk through the dataset directory and retrieves the path of all conn.log.labeled files 
	Parameters:
		dataset_dir (str): The path of the directory containing the dataset
	Returns:
		list[str]: A list containing the paths of all conn.log.labeled files
	"""
	print("[DEBUG] Retrieving all conn.log.labeled files...")
	conn_files_list = []
	for dirpath, _, filename in os.walk(dataset_dir):	
		for f in filename:
			# print(f"[DEBUG] {f}")
			if f.endswith("conn.log.labeled"):
				conn_files_list.append(os.path.join(dirpath, f))
	if not conn_files_list:
		raise FileNotFoundError(f"No conn.log.labeled files found in {dataset_dir}")
	return conn_files_list

def clean_iot23_columns(df: DataFrame, schema: StructType) -> DataFrame:
	col_transformations = []
	for i, field in enumerate(schema.fields):
		col_expr = F.col("split_cols").getItem(i)
		col_expr = F.when(col_expr == "-", None).otherwise(col_expr) # replace missing value (-) with None

		if field.name == "detailed_label":
			col_expr = F.when(col_expr.isNull(), "Benign").otherwise(col_expr)
		col_expr = col_expr.cast(field.dataType).alias(field.name)  # cast and rename each columns
		col_transformations.append(col_expr)
	return df.select(*col_transformations) # apply the transformation to each columns

def load_iot23(spark: SparkSession, config: DatasetConfig) -> DataFrame:
	""" Load the IoT23 dataset and apply all the modification so that the df can be used """
	conn_files_list = get_all_conn_files(config.path)
	df_raw = spark.read.text(conn_files_list)
	df_raw = df_raw.filter(~F.col("value").startswith("#")) # remove Zeek comment lines

	# There's inconsistency on the Zeek logs files, sometimes columns are separated by
	# using tabs and sometimes using space. We use a regex to regroup both cases
	df_raw = df_raw.withColumn("split_cols", F.split(F.trim(F.col("value")), r"\s+"))

	df_raw = df_raw.filter(F.size(F.col("split_cols")) == 23) # drop malformed lines
	df_cleaned = clean_iot23_columns(df_raw, config.schema)
	
	df_cleaned = cast_label_to_int(df_cleaned, "label", "Malicious")	
	return df_cleaned
	
def load_dataset(spark: SparkSession, dataset_choice: str, config: DatasetConfig) -> DataFrame:
	if os.path.exists(config.saving_path):
		print(f"[DEBUG] Data has already been pre-processed, loading it")
		processed_df = load_data(spark, config.saving_path)
		numeric_cols = get_num_columns(processed_df, config.label_features + config.noisy_features)
	else: # the dataset has not bee pre-processed yet
		print(f"[DEBUG] Data has not been pre-processed yet, loading all file from {config.path}")
		if config.name == "unsw":
			df = load_unsw(spark, config)
		elif config.name == "iotid20":
			df = load_iotid20(spark, config)
		elif config.name == "iot23":
			df = load_iot23(spark, config)

		print(f"[DEBUG] Number of features before pre processing {len(df.columns)}")
		processed_df, numeric_cols = preprocessing(df, config)

		# inside load_dataset(), after: processed_df, numeric_cols = preprocessing(df, config)
		with open(f"{config.path}/feature_names.json", "w") as fp:
			json.dump(numeric_cols, fp)
		
		print(f"[DEBUG] Feature names saved to {config.path}/feature_names.json")
		print(f"[DEBUG] Number of features after pre processing {len(processed_df.columns)}")
		save_data(processed_df, config.saving_path)
	return processed_df, len(numeric_cols)
