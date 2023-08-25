## About the project

# Stock Data Clustering Using PySpark K-means
This project demonstrates how to utilize PySpark for K-means clustering to predict stock data values. The objective is to create a model that predicts stock trends using supervised learning techniques. We'll perform the following steps within a Google Colab notebook:

### `live PDF:` [Open](https://github.com/kaiodeodato/Unsupervised_learning45/blob/main/M5_U7_kaio_deodato.pdf).
### `My LinkedIn:` [Go](https://www.linkedin.com/in/kaio-viana-6ab42016b/).

### Build with:

 • Python
 
 • Apache Spark
 
 • SQL

 • Spark Machile Learnig libraries


# Stock Data Clustering Using PySpark K-means


## Steps:

## 1. Create a New Notebook

Generate a new notebook in your Google Colab environment or in local work machine.

## 2. Install PySpark
```
Install the PySpark module in your Colab environment using the following command:

```

# 3. Create Spark Session

Set up the Spark session to work with PySpark:

```
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("StockPrediction").getOrCreate()
```

## 4. Load Data into DataFrame

Load the dataset `stocks_2021.csv` into a DataFrame:

```
df = spark.read.csv("stocks_2021.csv", header=True, inferSchema=True)
```
## 5. Import Required Modules

Import the necessary modules for the analysis:

```
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
```
## 6. Convert Data Types

Convert the data types of the 'open', 'low', and 'close' columns to float:

```
df = df.withColumn("open", df["open"].cast("float"))
df = df.withColumn("low", df["low"].cast("float"))
df = df.withColumn("close", df["close"].cast("float"))
```
## 7. Create Features Column

Use the VectorAssembler to create a 'features' column combining 'open', 'low', and 'close':

```
feature_cols = ["open", "low", "close"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)
```
## 8. Configure K-means

Create a K-means model object and configure it for 5 clusters:

```
kmeans = KMeans().setK(5).setSeed(42)
```
## 9. Train the Model

Train the K-means model using the 'features' column:

```
model = kmeans.fit(df)
```
## 10. Display Results

Show the results of the clustering:

```
results = model.transform(df)
results.show()
```
 
