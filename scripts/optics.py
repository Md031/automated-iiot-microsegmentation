from pyspark.sql import DataFrame, Row, SparkSession, functions as F
from sklearn.cluster import OPTICS
from pyspark.sql.functions import col
import numpy as np
import pandas as pd
import time
import sys
from config import DatasetConfig, DATASET_CONFIG, SEED
from load_dataset import create_spark_session, load_data
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from load_dataset import save_data
import random

def extract_features_to_list(df_sample: DataFrame, config: DatasetConfig):
	""" Return a python list and a numpy array of the uids and the pca features for OPTICS """
	sample_data = df_sample.collect()
	uid_list = []
	pca_list = []
	for col in sample_data:
		uid_list.append(col[config.uid])
		pca_list.append(col["pca_features"].toArray())
	return uid_list, np.array(pca_list)

def attach_cluster_labels(spark: SparkSession, df_sample: DataFrame, config: DatasetConfig, uid_list: list[str], cluster_labels: list[int]):
	""" Attach OPTICS cluster labels to the sampled DataFrame."""
	cluster_data = []
	for uid, cluster in zip(uid_list, cluster_labels):
		cluster_data.append((uid, int(cluster)))
	schema = StructType([
		StructField(config.uid, StringType(), True),
		StructField("clusterId", IntegerType(), True)
	])
	cluster_df = spark.createDataFrame(cluster_data, schema)
	df_with_cluster = df_sample.join(cluster_df, on=config.uid, how="left")
	return df_with_cluster

def random_sampling(df: DataFrame, config: DatasetConfig):
	""" Randomly sample PCA-transformed flows for clustering."""
	df_pca = df.select(config.uid,"pca_features") # df with only the col label and pca_feature
	fraction = 0.05
	# take a sample of the dataset
	df_sample = df_pca.sample(withReplacement=False, fraction=fraction, seed=SEED).limit(config.sample_size)
	return df_sample

def balanced_sampling(df: DataFrame, config: DatasetConfig):
	""" Create a balanced benign/malicious sample for clustering."""
	fraction = 0.05
	label_col = config.label_features[0]
	half_size = config.sample_size // 2
	# isolate the normal traffic
	normal_df = df.filter(F.col(label_col) == 0).orderBy(F.rand(seed=SEED))
	normal_df = normal_df.limit(half_size)

	# isolate the malicious traffic
	malicious_df = df.filter(F.col(label_col) == 1).orderBy(F.rand(seed=SEED))
	malicious_df = malicious_df.limit(half_size)
	balanced_df = normal_df.union(malicious_df)
	return balanced_df

def print_clustering_performance(labels: np.ndarray):
	""" Analyzes and prints the structural performance of the OPTICS clustering. """
	# Filter out noise (-1) to count real clusters
	unique_labels = set(labels)
	n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
	n_noise = list(labels).count(-1)
	total_points = len(labels)
	
	print("\n" + "="*40)
	print("      OPTICS CLUSTERING PERFORMANCE")
	print("="*40)
	print(f"Total Samples Processed: {total_points:,}")
	print(f"Clusters Identified:     {n_clusters}")
	print(f"Noise Points Detected:   {n_noise} ({ (n_noise/total_points)*100:.2f}%)")
	print("="*40 + "\n")

def apply_optics(spark: SparkSession, df: DataFrame, config: DatasetConfig):
	"""Sample the PCA data, run OPTICS, and return clustered flows."""
	if config.name == "iot23":
		df_sample = balanced_sampling(df, config)
	else:
		df_sample = random_sampling(df, config)

	# retrieve the features from the pyspark Dataframe
	uid_list, pca_array = extract_features_to_list(df_sample, config)

	# create the OPTICS model
	optics_model = OPTICS( # parameters derived from the reference paper
		min_samples=2,
		max_eps=np.inf,
		metric='chebyshev',
		cluster_method='xi'
	)
	print("Fitting OPTICS model...")
	cluster_labels = optics_model.fit_predict(pca_array)
	print_clustering_performance(cluster_labels)
	df_with_cluster = attach_cluster_labels(spark, df_sample, config, uid_list, cluster_labels)
	return df_with_cluster

def main():
	start = time.time()
	dataset_choice = sys.argv[1]
	config = DATASET_CONFIG[dataset_choice]
	spark = create_spark_session(f"{dataset_choice}")
	spark.sparkContext.setLogLevel("ERROR")
	print(f"Loading the dataset...")
	dataset_df = load_data(spark, config.pca_path) # load pca data

	print(f"Running OPTICS...")
	df_with_cluster = apply_optics(spark, dataset_df, config)
	# --- ADD THIS SECTION ---
	print("\nTop 10 Largest Clusters (Micro-segments):")
	# Filter out noise (-1) to see actual functional groups
	cluster_counts = df_with_cluster.filter(F.col("clusterId") != -1) \
		.groupBy("clusterId") \
		.count() \
		.orderBy(F.desc("count"))
	
	cluster_counts.show(10)
	# ------------------------

	path = f"{config.optics_path}"
	save_data(df_with_cluster, path)
	elapsed_time = time.time() - start
	print(f"Code duration: {elapsed_time:.2f}s")

if __name__ == "__main__":
	main()
