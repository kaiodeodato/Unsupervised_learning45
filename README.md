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

This project demonstrates how to utilize PySpark for K-means clustering to predict stock data values. The objective is to create a model that predicts stock trends using supervised learning techniques. We'll perform the following steps within a Google Colab notebook.

## Steps:

### 1. Create a New Notebook

Generate a new notebook in your Google Colab environment.

```
# 2. Install PySpark

Install the PySpark module in your Colab environment using the following command:

```

# 3. Create Spark Session

Set up the Spark session to work with PySpark:

```
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("StockPrediction").getOrCreate()
```



 
