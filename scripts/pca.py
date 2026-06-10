from load_dataset import save_data
from config import DatasetConfig, SEED
from pyspark.ml.feature import PCA, PCAModel
from pyspark.sql import DataFrame 


def find_optimal_PCA_nb(full_pca_model: PCAModel, pca_expl_var: float) -> int:
	""" Calculates the minimum number of components (k) required to reach the reach the variance threshold """
	# Vector of proportions of variance explained by each principal component
	explained = full_pca_model.explainedVariance.toArray()
	optimal_k = len(explained)
	# Determine optimal k from variance
	cumvar = explained.cumsum()
	print(f"[DEBUG] cumvar : {cumvar}")
	for i in range(len(cumvar)):
		if cumvar[i] >= pca_expl_var:
			print(f"[DEBUG] Optimal k found (PCA): {i}")
			print(f"[DEBUG] Explained Variance: {cumvar[i]}")
			return i + 1
	# return k=len(explained) if the threshold is never reached
	return optimal_k

def apply_pca(df: DataFrame, nb_col: int, config: DatasetConfig):
	sample_df = df.sample(fraction=0.2, seed=SEED) # do the pca on 
	full_pca = PCA(
		k=nb_col,
		inputCol="scaled_features",
		outputCol="pca_full"  # Contains ALL components
	)
	full_pca_model = full_pca.fit(sample_df) 
	if config.name == "iot23":
		optimal_k = find_optimal_PCA_nb(full_pca_model, config.pca_expl_var)
	else:
		optimal_k = config.pca_expl_var
	# df_with_pca = full_pca_model.transform(df)
	final_pca = PCA(k=optimal_k, inputCol="scaled_features", outputCol="pca_features")
	final_model = final_pca.fit(df)
	df_final = final_model.transform(df)
	
	df_final = df_final.drop("pca_full", "scaled_features")
	save_data(df_final, config.pca_path)
	# Return label features + sliced PCA features
	print("[DEBUG] Schema in PCA")
	df_final.printSchema() 

	spark = df.sparkSession
	df_cut = spark.read.parquet(config.pca_path) # we re read
	return df_cut
