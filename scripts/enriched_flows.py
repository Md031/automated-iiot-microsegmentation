import time
import os
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler, StandardScaler

from config import DATASET_CONFIG, SEED, dt_max_depth
from load_dataset import create_spark_session, load_data
from decision_tree import performance_evaluation, per_attack_analysis
from pyspark.ml.classification import DecisionTreeClassifier

def compute_ip_behavioral_features(df: DataFrame, src_ip_col: str):
	"""
	Compute per-source-IP aggregated behavioral features across the full dataset
	Returns a DataFrame with one row per unique source IP
	"""
	ip_stats = df.groupBy(src_ip_col).agg(
		F.countDistinct(F.col("id_resp_h")).alias("distinct_dst_ips"),
		F.countDistinct(F.col("id_resp_p")).alias("distinct_dst_ports"),
		F.count("*").alias("total_flows"),
		(F.sum(F.when(F.col("resp_bytes") == 0, 1).otherwise(0)) / F.count("*"))
			.alias("zero_resp_ratio")
	)
	return ip_stats


def add_behavioral_features(df: DataFrame, src_ip_col: str):
	"""
	Compute and join behavioral features back onto every individual flow.
	Returns the enriched DataFrame.
	"""
	ip_stats = compute_ip_behavioral_features(df, src_ip_col)
	df_enriched = df.join(ip_stats, on=src_ip_col, how="left")
	return df_enriched


def rescale_behavioral_features(df: DataFrame):
	""" Scale behavioral features and append them to the existing feature vector """
	new_cols = ["distinct_dst_ips", "distinct_dst_ports", "total_flows", "zero_resp_ratio"]

	# Assemble new features into a temporary vector
	assembler = VectorAssembler(
		inputCols=new_cols,
		outputCol="behavioral_features_raw"
	)
	df = assembler.transform(df)

	# Scale the new features
	scaler = StandardScaler(
		inputCol="behavioral_features_raw",
		outputCol="behavioral_features_scaled",
		withStd=True,
		withMean=True
	)
	scaler_model = scaler.fit(df)
	df = scaler_model.transform(df)

	# Combine original scaled_features + new behavioral features into one vector
	combiner = VectorAssembler(
		inputCols=["scaled_features", "behavioral_features_scaled"],
		outputCol="enriched_features"
	)
	df = combiner.transform(df)

	# Drop intermediary columns
	df = df.drop("behavioral_features_raw", "behavioral_features_scaled")
	return df


def train_dt_enriched(full_df: DataFrame, config):
	"""
	Train a DT on enriched_features instead of scaled_features
	Uses the same hyperparameters and train/test split as the baseline
	"""
	train_df, test_df = full_df.randomSplit([0.7, 0.3], seed=SEED)

	dt = DecisionTreeClassifier(
		featuresCol="enriched_features",
		labelCol=config.label_features[0],
		maxDepth=dt_max_depth,
		maxBins=4096,
		minInstancesPerNode=1,
		impurity="entropy",
		seed=SEED
	)

	dt_model = dt.fit(train_df)

	predictions = dt_model.transform(test_df)
	print("Performance Evaluation")
	performance_evaluation(predictions, config)
	per_attack_analysis(predictions, config)
	return dt_model

def train_dt_enriched_weighted(full_df: DataFrame, config):
	label_col = config.label_features[0]
	total = full_df.count()
	counts = {r[label_col]: r['count'] 
			  for r in full_df.groupBy(label_col).count().collect()}
	weights = {c: total / (2 * n) for c, n in counts.items()}
	print(f"Class weights: {weights}")

	full_df = full_df.withColumn(
		"weight",
		F.when(F.col(label_col) == 0, weights[0]).otherwise(weights[1])
	)

	train_df, test_df = full_df.randomSplit([0.7, 0.3], seed=SEED)

	dt = DecisionTreeClassifier(
		featuresCol="enriched_features",
		labelCol=label_col,
		weightCol="weight",
		maxDepth=dt_max_depth,
		maxBins=4096,
		minInstancesPerNode=1,
		impurity="entropy",
		seed=SEED
	)

	dt_model = dt.fit(train_df)

	predictions = dt_model.transform(test_df)
	print("Performance Evaluation")
	performance_evaluation(predictions, config)
	per_attack_analysis(predictions, config)
	return dt_model 

def train_dt_enriched_stratified_sampling(full_df: DataFrame, config):
	label_col = config.label_features[0]

	# Split first: test set keeps the original IoT-23 distribution
	train_df, test_df = full_df.randomSplit([0.7, 0.3], seed=SEED)

	# Undersample the majority class in the training set only
	benign_df = train_df.filter(F.col(label_col) == 0)
	malicious_df = train_df.filter(F.col(label_col) == 1)

	benign_count = benign_df.count()
	malicious_count = malicious_df.count()

	if benign_count == 0 or malicious_count == 0:
		raise ValueError(
			f"Cannot perform stratified sampling with class counts: "
			f"benign={benign_count}, malicious={malicious_count}"
		)

	# IoT-23 has many more malicious flows, so we undersample malicious flows
	malicious_sampled = malicious_df.sample(
		withReplacement=False,
		fraction=min(1.0, benign_count / malicious_count),
		seed=SEED
	)

	balanced_train_df = benign_df.union(malicious_sampled)

	print(f"Original training set: {benign_count:,} benign, {malicious_count:,} malicious")
	print(f"Balanced training set: {benign_count:,} benign, {malicious_sampled.count():,} malicious")

	dt = DecisionTreeClassifier(
		featuresCol="enriched_features",
		labelCol=label_col,
		maxDepth=dt_max_depth,
		maxBins=4096,
		minInstancesPerNode=1,
		impurity="entropy",
		seed=SEED
	)

	dt_model = dt.fit(balanced_train_df)

	predictions = dt_model.transform(test_df)
	print("Performance Evaluation")
	performance_evaluation(predictions, config)
	per_attack_analysis(predictions, config)

	return dt_model

def main():
	start = time.time()
	dataset_choice = "iot23"
	config = DATASET_CONFIG[dataset_choice]
	if not os.path.isdir(config.dt_path):
		raise FileNotFoundError(
			f"No Decision Tree model found at {config.dt_path}. "
			"Run decision_tree.py first."
		)

	spark = create_spark_session(f"{dataset_choice}_enriched")
	spark.sparkContext.setLogLevel("ERROR")
	spark.conf.set("spark.local.dir", "/mnt/d/spark-spill")  # spill to D: drive

	print("Loading preprocessed dataset...")
	df = load_data(spark, config.saving_path)
	src_ip_col = config.network_features["srcIp"]
	if src_ip_col not in df.columns:
		raise ValueError(
			f"Source IP column '{src_ip_col}' not found in preprocessed data. "
			f"Available columns: {df.columns}"
		)
	print("Computing per-source-IP behavioral features...")
	df_enriched = add_behavioral_features(df, src_ip_col)
	print(f"Schema after enrichment: {df_enriched.columns}")
	print("Scaling and combining features...")
	df_enriched = rescale_behavioral_features(df_enriched)

	
	# print("Training enriched Decision Tree...")
	# trained_model = train_dt_enriched(df_enriched, config)

	# print("Training enriched + weighted Decision Tree...")
	# trained_model_weighted = train_dt_enriched_weighted(df_enriched, config)

	print("Training enriched + stratified sampling Decision Tree...")	
	trained_model_stratified = train_dt_enriched_stratified_sampling(df_enriched, config)

	# trained_model.write().overwrite().save(f"{config.path}/dt_model_enriched")

	elapsed_time = time.time() - start
	print(f"\nCode duration: {elapsed_time:.2f}s")

if __name__ == "__main__":
	main()