# Phase 2 Day 6 — Spark MLlib Advanced
# Date: April 16, 2026
# MLlib Pipeline = Databricks ML interview topic!

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler,
    StringIndexer, OneHotEncoder,
    PCA, ChiSqSelector,
    MinMaxScaler, Normalizer,
    Tokenizer, HashingTF, IDF,
    Word2Vec
)
from pyspark.ml.classification import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    LogisticRegression
)
from pyspark.ml.regression import (
    LinearRegression,
    RandomForestRegressor
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    RegressionEvaluator
)
from pyspark.ml.tuning import (
    CrossValidator, ParamGridBuilder,
    TrainValidationSplit
)
import mlflow

spark = SparkSession.builder \
    .appName("MLlibAdvanced") \
    .getOrCreate()

print("="*60)
print("Spark MLlib — Advanced Pipeline")
print("="*60)

# 1. Create rich dataset
data = [
    (1, "XGBoost", "tabular", 92.5,
     1.2, 100, True),
    (2, "RF", "tabular", 88.0,
     0.8, 50, True),
    (3, "NN", "image", 95.5,
     5.4, 200, True),
    (4, "SVM", "tabular", 78.0,
     0.5, 10, False),
    (5, "LR", "text", 82.0,
     0.3, 5, True),
    (6, "CNN", "image", 91.0,
     8.2, 300, True),
    (7, "BERT", "text", 94.5,
     12.1, 500, True),
    (8, "KNN", "tabular", 71.0,
     0.2, 1, False),
] * 25  # replicate for more data

df = spark.createDataFrame(data, [
    "id", "model_type", "data_type",
    "accuracy", "train_time",
    "n_estimators", "label"
]).withColumn("label",
    F.col("label").cast("double"))

train, test = df.randomSplit([0.8, 0.2], seed=42)
print(f"Train: {train.count()} | "
      f"Test: {test.count()}")

# 2. Feature engineering pipeline
print("\n=== FULL ML PIPELINE ===")

# Categorical encoding
model_indexer = StringIndexer(
    inputCol="model_type",
    outputCol="model_type_idx",
    handleInvalid="keep"
)
data_indexer = StringIndexer(
    inputCol="data_type",
    outputCol="data_type_idx",
    handleInvalid="keep"
)
model_encoder = OneHotEncoder(
    inputCols=["model_type_idx"],
    outputCols=["model_type_vec"]
)
data_encoder = OneHotEncoder(
    inputCols=["data_type_idx"],
    outputCols=["data_type_vec"]
)

# Assemble all features
assembler = VectorAssembler(
    inputCols=[
        "model_type_vec", "data_type_vec",
        "accuracy", "train_time",
        "n_estimators"
    ],
    outputCol="features_raw"
)

# Scale features
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withStd=True, withMean=True
)

# Classifier
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=20,
    seed=42
)

# Build complete pipeline
pipeline = Pipeline(stages=[
    model_indexer, data_indexer,
    model_encoder, data_encoder,
    assembler, scaler, rf
])

# 3. Train pipeline
print("Training pipeline...")
mlflow.set_experiment("spark_mllib_advanced")

with mlflow.start_run(
        run_name="RF_full_pipeline"):
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("numTrees", 20)
    mlflow.log_param("pipeline_stages", 7)

    model = pipeline.fit(train)
    predictions = model.transform(test)

    evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    auc = evaluator.evaluate(predictions)

    acc_evaluator = \
        MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy"
        )
    acc = acc_evaluator.evaluate(predictions)

    mlflow.log_metric("auc", auc)
    mlflow.log_metric("accuracy", acc)
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")

# 4. Hyperparameter tuning
print("\n=== CROSS VALIDATION ===")
rf_tune = RandomForestClassifier(
    featuresCol="features",
    labelCol="label"
)
pipeline_tune = Pipeline(stages=[
    model_indexer, data_indexer,
    model_encoder, data_encoder,
    assembler, scaler, rf_tune
])

paramGrid = (ParamGridBuilder()
    .addGrid(rf_tune.numTrees, [10, 20])
    .addGrid(rf_tune.maxDepth, [3, 5])
    .build()
)
cv = CrossValidator(
    estimator=pipeline_tune,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=3,
    seed=42
)
cv_model = cv.fit(train)
best_auc = evaluator.evaluate(
    cv_model.transform(test)
)
print(f"Best CV AUC: {best_auc:.4f}")
print(f"Best params: "
      f"{cv_model.bestModel.stages[-1].extractParamMap()}")

# 5. Save and load pipeline
pipeline_path = "/tmp/mllib_pipeline"
cv_model.bestModel.write().overwrite() \
        .save(pipeline_path)
print(f"\nPipeline saved to {pipeline_path}")

# Load back
from pyspark.ml import PipelineModel
loaded = PipelineModel.load(pipeline_path)
loaded_preds = loaded.transform(test)
loaded_auc = evaluator.evaluate(loaded_preds)
print(f"Loaded pipeline AUC: {loaded_auc:.4f}")

# 6. Feature importance
rf_model = cv_model.bestModel.stages[-1]
importances = rf_model.featureImportances
print(f"\nFeature importances: {importances}")

# 7. Text features with TF-IDF
print("\n=== TEXT FEATURES (TF-IDF) ===")
text_data = spark.createDataFrame([
    (0, "spark mllib pipeline machine learning"),
    (1, "databricks delta lake mlflow"),
    (2, "pytorch neural network deep learning"),
    (3, "spark sql dataframe optimization"),
], ["id", "text"])

tokenizer = Tokenizer(
    inputCol="text", outputCol="words"
)
hashingTF = HashingTF(
    inputCol="words", outputCol="rawFeatures",
    numFeatures=20
)
idf = IDF(
    inputCol="rawFeatures",
    outputCol="features"
)

text_pipeline = Pipeline(
    stages=[tokenizer, hashingTF, idf]
)
text_model = text_pipeline.fit(text_data)
text_features = text_model.transform(text_data)
text_features.select(
    "text", "features"
).show(truncate=50)

mlflow.log_metric("text_pipeline_rows",
                   text_features.count())
print("\nSparkMLlib Advanced complete!")
