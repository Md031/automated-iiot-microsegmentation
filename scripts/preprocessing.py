from pyspark.sql import DataFrame 
from pyspark.ml.feature import StringIndexer, StandardScaler, Imputer, VectorAssembler, PCA, PCAModel, VectorSlicer
from pyspark.sql.types import StringType, NumericType
from pyspark.ml.stat import Correlation
from config import DatasetConfig

def get_str_cols(df: DataFrame, protected_features: list[str]) -> list[str]:
	"""Return list of str columns (StringType)"""
	str_cols = []
	for field in df.schema.fields:
		if isinstance(field.dataType, StringType) and field.name not in protected_features:
			str_cols.append(field.name) 
	return str_cols

def get_num_columns(df: DataFrame, protected_features: list[str]) -> list[str]:
	"""Return list of numeric columns (DoubleType or IntegerType) and a df with only the numeric columns"""
	num_col = []
	for field in df.schema.fields:
		if isinstance(field.dataType, NumericType) and field.name not in protected_features:
			num_col.append(field.name) 
	return num_col

def labelEncoding(df: DataFrame, config: DatasetConfig) -> DataFrame:
	""" Code inspired from here : https://stackoverflow.com/questions/30580410/how-to-do-labelencoding-or-categorical-value-in-apache-spark """
	str_cols = get_str_cols(df, config.label_features + config.noisy_features)
	print(f"String columns: {str_cols}")
	indexed_cols = []
	for elem in str_cols:
		indexed_cols.append(elem + "_index")
	indexer = StringIndexer(
		inputCols = str_cols,
		outputCols = indexed_cols,
		handleInvalid = "keep"
		)
	model = indexer.fit(df)
	df_encoded = model.transform(df)

	# rename the "_index" columns to their original name
	df_encoded = df_encoded.drop(*str_cols)
	for orig_name, new_name in zip(str_cols, indexed_cols):
		df_encoded = df_encoded.withColumnRenamed(new_name, orig_name)
	return df_encoded

def missing_values_handler(df: DataFrame, numeric_cols: list) -> DataFrame:
	""" Handle the missing value (previously set to None) and set them to them mean of the column """
	imputer = Imputer(
		inputCols=numeric_cols,
		outputCols=numeric_cols,
		strategy="mean"
	)
	model = imputer.fit(df)
	return model.transform(df)

def standardScaling(df: DataFrame, numeric_cols: list) -> DataFrame:
	"""	
	Standardize the numerics columns of the DataFrame
	Returns:
		Dataframe: a new dataframe with a new column "scaled_features" 
	"""
	df_standardized = missing_values_handler(df, numeric_cols)

	# Create a new columns containing all the numeric columns
	assembler = VectorAssembler(
		inputCols=numeric_cols,
		outputCol="features",
	)
	df_vector = assembler.transform(df_standardized)

	# Standardize the value of the new column created
	scaler = StandardScaler(
		inputCol="features",
		outputCol="scaled_features",
		withStd=True,
		withMean=True
	)
	scaler_model = scaler.fit(df_vector)
	df_standardized = scaler_model.transform(df_vector)
	return df_standardized

def compute_corr_matrix(df: DataFrame):
	# Compute correlation matrix
	corr_matrix = Correlation.corr(
		df.select("scaled_features"),
		"scaled_features",
		method="pearson"
		).head()[0].toArray()
	return corr_matrix

def correlation(df: DataFrame, corr_matrix, numeric_cols, threshold=0.95):
	"""
	Computes absolute correlation between all numeric columns efficiently using Spark ML.
	Prints pairs above the threshold.
	"""
	high_corr_pairs = []
	for i in range(len(numeric_cols)):
		for j in range(i + 1, len(numeric_cols)):
			corr_value = abs(corr_matrix[i, j])
			if corr_value > threshold:
				high_corr_pairs.append((numeric_cols[i], numeric_cols[j], corr_value))
	high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
	print(f"Number of highly correlated pairs (> {threshold}): {len(high_corr_pairs)}\n")
	for col1, col2, corr_value in high_corr_pairs:
		print(f"{col1:25} {col2:25} corr = {corr_value:.4f}")

def drop_perfectly_correlated(df: DataFrame, corr_matrix, numeric_cols) -> DataFrame:
	"""
	Drops one column from each pair with correlation >= threshold using Spark ML.
	"""
	threshold = 0.99
	to_drop = set()
	for i in range(len(numeric_cols)):
		for j in range(i + 1, len(numeric_cols)):
			if abs(corr_matrix[i, j]) >= threshold:
				to_drop.add(numeric_cols[i])  # drop first column of the pair
	if to_drop:
		print(f"\nDropping {len(to_drop)} perfectly correlated features (corr >= {threshold}):")
		print(to_drop)
		df = df.drop(*to_drop)
	else:
		print("\nNo perfectly correlated features found.")
	return df

def preprocessing(df: DataFrame, config: DatasetConfig) -> DataFrame:
	print(f"Starting pre-processing data")
	
	# save the uid columns and drop the noisy features
	# df_preprocessing = df.drop(*config.noisy_features) # drop the unecessary features

	df_preprocessing = labelEncoding(df, config)
	numeric_cols = get_num_columns(df_preprocessing, config.label_features + config.noisy_features)
	df_preprocessing = standardScaling(df_preprocessing, numeric_cols)
	df_preprocessing = df_preprocessing.drop("features")
	corr_matrix = compute_corr_matrix(df_preprocessing)
	# correlation(df_preprocessing, corr_matrix, numeric_cols)
	df_corr = drop_perfectly_correlated(df_preprocessing, corr_matrix, numeric_cols)
	
	return df_corr, numeric_cols
