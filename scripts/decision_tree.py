from pyspark.ml.classification import DecisionTreeClassifier, DecisionTreeClassificationModel
from pyspark.sql import DataFrame, SparkSession
from config import DatasetConfig, DATASET_CONFIG, SEED, dt_max_depth
from load_dataset import create_spark_session, load_data
import time
import sys
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import pandas as pd
import os

def save_dt_model(dt_model: DecisionTreeClassificationModel, config: DatasetConfig):
	""" Save the DT model """
	dt_model.write().overwrite().save(config.dt_path)
	print(f"Decision Tree model saved to {config.dt_path}")

def load_dt_model(config: DatasetConfig) -> DecisionTreeClassificationModel:
	""" Load the DT model """
	if not os.path.exists(config.dt_path):
		raise FileNotFoundError(
			f"No saved DT model at {config.dt_path}. Run decision_tree.py first."
		)
	model = DecisionTreeClassificationModel.load(config.dt_path)
	print(f"DT model loaded from {config.dt_path}")
	print(f"depth={model.depth}  nodes={model.numNodes}  features={model.numFeatures}")
	return model
	
def performance_evaluation(predictions: DataFrame, config: DatasetConfig):
	""" Compute the accuracy, sensitivity, and specificity metric results of the DT model """
	label_col = config.label_features[0]
	
	# Full confusion matrix - single scan of the DataFrame
	print("\nDetailed Confusion Matrix:")
	cm = predictions.groupBy(label_col, "prediction").count().orderBy(label_col, "prediction")
	cm.show()

	# Derive TP, TN, FP, FN from the already-collected confusion matrix
	cm_dict = {(row[label_col], int(row["prediction"])): row["count"] for row in cm.collect()}
	TP = cm_dict.get((1, 1), 0)
	TN = cm_dict.get((0, 0), 0)
	FP = cm_dict.get((0, 1), 0)
	FN = cm_dict.get((1, 0), 0)
	total = TP + TN + FP + FN

	# Calculate metrics
	accuracy = (TP + TN) / total if total > 0 else 0.0
	sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
	specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
	precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
	f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

	print("\n--- Decision Tree Evaluation ---")
	print(f"Total Test Samples: {total:,}")
	print(f"Accuracy:    {accuracy:.4%} (Flows correctly classified)")
	print(f"Sensitivity (Recall): {sensitivity:.4%} (Attack Detection Rate)")
	print(f"Specificity: {specificity:.4%} (Normal Traffic Allowed)")
	print("----------------------------------\n")
	
def train_dt(full_df: DataFrame, config: DatasetConfig):
	""" Basic training of the DT model """
	# Split
	train_df, test_df = full_df.randomSplit([0.7, 0.3], seed=SEED)
	
	# Train DT
	dt = DecisionTreeClassifier(
		featuresCol="scaled_features", 
		labelCol=config.label_features[0],
		maxDepth=dt_max_depth,
		maxBins=4096,
		minInstancesPerNode=1,
		impurity="entropy",
		seed=SEED
	)
	
	dt_model = dt.fit(train_df)
	
	# Predictions
	dt_classified = dt_model.transform(test_df)
	print("Perfomance Evaluation on the whole dataset")
	performance_evaluation(dt_classified, config)
	per_attack_analysis(dt_classified, config)
	return dt_model

def train_dt_stratified_sampling(full_df: DataFrame, config: DatasetConfig):
	""" Training using the stratified sampling strategy """

	label_col = config.label_features[0]
	
	# Split - test_df stays untouched with original distribution
	train_df, test_df = full_df.randomSplit([0.7, 0.3], seed=SEED)
	
	# Balanced sampling on training set only
	benign_df = train_df.filter(F.col(label_col) == 0)
	malicious_df = train_df.filter(F.col(label_col) == 1)
	
	benign_count = benign_df.count()
	malicious_count = malicious_df.count()
	
	malicious_sampled = malicious_df.sample(
		withReplacement=False,
		fraction=benign_count / malicious_count,
		seed=SEED
	)
	
	balanced_train_df = benign_df.union(malicious_sampled)
	print(f"Original training set: {benign_count:,} benign, {malicious_count:,} malicious")
	print(f"Balanced training set: {benign_count:,} benign, {malicious_sampled.count():,} malicious")
	
	# Train DT - identical parameters, just different training data
	dt = DecisionTreeClassifier(
		featuresCol="scaled_features",
		labelCol=label_col,
		maxDepth=dt_max_depth,
		maxBins=4096,
		minInstancesPerNode=1,
		impurity="entropy",
		seed=SEED
	)
	
	dt_model = dt.fit(balanced_train_df)
	
	# Predictions on original test set
	dt_classified = dt_model.transform(test_df)
	print("Performance Evaluation on the whole dataset")
	performance_evaluation(dt_classified, config)
	per_attack_analysis(dt_classified, config)
	return dt_model

def train_dt_weighted(full_df: DataFrame, config: DatasetConfig):
	""" Training using the weighted sampling strategy """

	label_col = config.label_features[0]
	train_df, test_df = full_df.randomSplit([0.7, 0.3], seed=SEED)

	# Compute inverse-frequency class weights on training set
	total = train_df.count()
	counts = train_df.groupBy(label_col).count().collect()
	count_map = {row[label_col]: row["count"] for row in counts}
	n_classes = len(count_map)

	weight_map = {
		cls: total / (n_classes * cnt)
		for cls, cnt in count_map.items()
	}
	print(f"Class weights: {weight_map}")

	# Add weight column
	train_df = train_df.withColumn(
		"classWeight",
		F.when(F.col(label_col) == 0, weight_map[0])
		 .otherwise(weight_map[1])
	)

	dt = DecisionTreeClassifier(
		featuresCol="scaled_features",
		labelCol=label_col,
		weightCol="classWeight",
		maxDepth=dt_max_depth,
		maxBins=4096,
		minInstancesPerNode=1,
		impurity="entropy",
		seed=SEED
	)

	dt_model = dt.fit(train_df)
	dt_classified = dt_model.transform(test_df)
	print("Performance Evaluation on the whole dataset")
	performance_evaluation(dt_classified, config)
	per_attack_analysis(dt_classified, config)	
	return dt_model

def dt_on_optics(dt_model: DecisionTreeClassificationModel, dt: DataFrame, config: DatasetConfig):
	""" Apply DT on the OPTICS sample to generate the firewall rules """
	dt_classified = dt_model.transform(dt)
	# print("Perfomance Evaluation on the OPTICS dataset")
	# performance_evaluation(dt_classified, config)
	return dt_classified

def generate_firewall_rules(df_classified: DataFrame, config: DatasetConfig):
	""" Generate the firewall rules based on the classification of the DT model on the OPTICS sample """
	print("\n Generating Micro-segmentation Firewall Rules...")
	
	# 1. Generate Allow Rules (Ignoring Noise Cluster -1)
	allow_df = df_classified.filter((F.col("prediction") == 0.0) & (F.col("clusterId") != -1))
	allow_rules = allow_df.select(
		F.col("clusterId").alias("Micro-segment"),
		F.col(config.network_features["srcIp"]).alias("SIP"),
		F.col(config.network_features["srcPort"]).alias("SPort"),
		F.col(config.network_features["dstIp"]).alias("DIP"),
		F.col(config.network_features["dstPort"]).alias("DPort"),
		F.col(config.network_features["protocol"]).alias("Proto"),
		F.lit("Allow").alias("Action")
	)

	# 2. Generate Block Rules (Including Noise Cluster)
	block_df = df_classified.filter((F.col("prediction") == 1.0) | (F.col("clusterId") == -1))	
	block_rules = block_df.select(
		F.col("clusterId").alias("Micro-segment"),
		F.col(config.network_features["srcIp"]).alias("SIP"),
		F.col(config.network_features["srcPort"]).alias("SPort"),
		F.col(config.network_features["dstIp"]).alias("DIP"),
		F.col(config.network_features["dstPort"]).alias("DPort"),
		F.col(config.network_features["protocol"]).alias("Proto"),
		F.lit("Block").alias("Action")
	)

	# remove redundant links 
	allow_rules = allow_rules.distinct()
	block_rules = block_rules.distinct()

	# 3. Combine them into one master table
	all_rules_df = allow_rules.union(block_rules).orderBy("Micro-segment", "Action")
	
	# Print a preview to the terminal
	print("\nPreview of generated firewall policies:")
	all_rules_df.show(20, truncate=False)


	# 4. Save to a CSV using Pandas
	output_path = f"{config.path}/firewall_rules.csv"
	
	# Convert PySpark DF to Pandas DF
	pandas_df = all_rules_df.toPandas() 
	pandas_df.to_csv(output_path, index=False)
	
	print(f"Successfully saved {len(pandas_df)} distinct firewall rules to: {output_path}")

def per_attack_analysis(predictions: DataFrame, config: DatasetConfig):
	""" Per attack analysis to determine the detection rate of each attack """
	# check if a detailed label column exists
	if len(config.label_features) < 2:
		print("No detailed label column defined, skipping per-attack analysis.")
		return

	detailed_label_col = config.label_features[1]
	label_col = config.label_features[0]

	# check the column actually exists in the DataFrame
	if detailed_label_col not in predictions.columns:
		print(f"Column '{detailed_label_col}' not found, skipping per-attack analysis.")
		return

	print("\n--- Per-Attack Type Analysis ---")

	# 1. Benign flow prediction breakdown
	print("\nBenign flows (label=0) prediction breakdown:")
	predictions.filter(F.col(label_col) == 0) \
		.groupBy("prediction").count() \
		.orderBy("prediction").show()

	# 2. Per attack type misclassification breakdown
	print("\nPer attack type classification breakdown:")
	
	# First, get the raw counts just like you did before
	attack_counts = predictions.filter(F.col(label_col) == 1) \
		.groupBy(detailed_label_col, "prediction") \
		.count()

	# Define a window partitioned by the attack type
	window_spec = Window.partitionBy(detailed_label_col)

	# Calculate the total per attack, compute the percentage, and sort
	attack_counts.withColumn("total_for_attack", F.sum("count").over(window_spec)) \
		.withColumn("percentage (%)", F.round((F.col("count") / F.col("total_for_attack")) * 100, 2)) \
		.orderBy(detailed_label_col, "prediction") \
		.show(50, truncate=False)
		
	# 3. Feature means — single groupBy pass, one Spark job for everything
	candidate_features = ["duration", "orig_bytes", "history",
						  "orig_pkts", "tunnel_parents", "conn_state"]
	feature_cols = [f for f in candidate_features if f in predictions.columns]

	if not feature_cols:
		print("No raw feature columns found in predictions, skipping means comparison.")
		return

	print("\nFeature means per group:")

	# Tag each row with its analysis group in a single pass.
	# "AAA_" and "AAB_" prefixes force the benign reference rows to sort first.
	labeled = predictions.withColumn(
		"analysis_group",
		F.when(
			(F.col(label_col) == 0) & (F.col("prediction") == 1.0),
			F.lit("AAA_FP-Benign (misclassified)")
		).when(
			(F.col(label_col) == 0) & (F.col("prediction") == 0.0),
			F.lit("AAB_TN-Benign (correct)")
		).when(
			(F.col(label_col) == 1) & (F.col("prediction") == 0.0),
			F.concat(F.lit("FN-"), F.col(detailed_label_col))
		).otherwise(
			F.concat(F.lit("TP-"), F.col(detailed_label_col))
		)
	).filter(F.col("analysis_group").isNotNull())

	# Single aggregation over all groups at once
	result = labeled.groupBy("analysis_group").agg(
		F.count("*").alias("flow_count"),
		*[F.mean(f).alias(f) for f in feature_cols]
	).orderBy("analysis_group")

	result.show(50, truncate=False)

def main():
	start = time.time()
	dataset_choice = sys.argv[1]
	config = DATASET_CONFIG[dataset_choice]
	spark = create_spark_session(f"{dataset_choice}")
	spark.sparkContext.setLogLevel("ERROR")
	print(f"Loading the dataset...")
	dataset_df = load_data(spark, config.saving_path) # load pca data
	optics_df = load_data(spark, config.optics_path)

	enriched_optics_dt = dataset_df.join(
		optics_df.select(config.uid, "clusterId"),
		on = config.uid,
		how = "inner"
	)
	# train decision tree
	trained_model = train_dt(dataset_df, config)
	# trained_model = train_dt_stratified_sampling(dataset_df, config)
	# trained_model = train_dt_weighted(dataset_df, config)
	# save the model
	save_dt_model(trained_model, config)

	optics_dt = dt_on_optics(trained_model, enriched_optics_dt, config)
	
	generate_firewall_rules(optics_dt, config)

	elapsed_time = time.time() - start
	print(f"Code duration: {elapsed_time:.2f}s")

if __name__ == "__main__":
	main()