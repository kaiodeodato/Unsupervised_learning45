from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt

spark = SparkSession.builder.appName("app_model").getOrCreate()

df = spark.read.option("delimiter", ",").option("header", "true").csv("stocks_2021.csv")

df = df.withColumn('open', df.open.cast('float'))
df = df.withColumn('low', df.low.cast('float'))
df = df.withColumn('close', df.close.cast('float'))

va = VectorAssembler(inputCols=['open','low','close'], outputCol='features')

va_df = va.transform(df)


# construo um gráfico de 'inercia' em função do numero de clusters
inertias = []

for k in range(2, 11):  # Experimente k de 2 a 10
    kmeans = KMeans(k=k, seed=42)
    model = kmeans.fit(va_df.select('features'))
    inertias.append(model.summary.trainingCost)

# plt.plot(x_values, y_values, marker='o', markersize=8, markerfacecolor='red', markeredgecolor='black')

# dados importantes
plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), inertias, marker='o')
# informações gráfico
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
# mostrar
plt.show()


optimal_k = 4  # Escolha o valor de k com base na análise do gráfico
kmeans = KMeans(k=optimal_k, seed=42)

model = kmeans.fit(va_df.select('features'))

transformed = model.transform(va_df)

transformed.show(truncate=False)

# obtenho os centros dos clusters e mostro esses pontos
cluster_centers = model.clusterCenters()
for i, center in enumerate(cluster_centers):
    print(f"Cluster {i + 1} Center: {center}")