from sklearn.cluster import AgglomerativeClustering
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt

# Inicializa a sessão do Spark
spark = SparkSession.builder.appName("app_model").getOrCreate()

# Lê o arquivo com o delimitador "," e cabeçalho
df = spark.read.option("delimiter", ",").option("header", "true").csv("stocks_2021.csv")

# Converte as colunas em float
df = df.withColumn('open', df.open.cast('float'))
df = df.withColumn('low', df.low.cast('float'))
df = df.withColumn('close', df.close.cast('float'))

# Cria um VectorAssembler para agrupar as colunas 'open', 'low' e 'close' em uma única coluna 'features'
va = VectorAssembler(inputCols=['open', 'low', 'close'], outputCol='features')

# Aplica o VectorAssembler ao DataFrame
va_df = va.transform(df)

# Converte o DataFrame do PySpark para um Pandas DataFrame
pandas_df = va_df.toPandas()

# Cria um modelo AgglomerativeClustering com linkage 'ward' e n_clusters=5 clusters
agglomerative = AgglomerativeClustering(linkage='ward', n_clusters=5)

# Aplica o modelo AgglomerativeClustering aos dados Pandas
pandas_df['prediction'] = agglomerative.fit_predict(pandas_df[['open', 'low', 'close']])

# Gráficos de Dispersão (Scatter Plots)
# Cria um gráfico de dispersão colorindo os pontos de acordo com os clusters
plt.scatter(pandas_df['open'], pandas_df['low'], c=pandas_df['prediction'])
plt.xlabel('Open')
plt.ylabel('Low')
plt.title('Scatter Plot with Clusters')
plt.show()

# Cria outro gráfico de dispersão colorindo os pontos de acordo com os clusters
plt.scatter(pandas_df['low'], pandas_df['close'], c=pandas_df['prediction'])
plt.xlabel('Low')
plt.ylabel('Close')
plt.title('Scatter Plot with Clusters')
plt.show()


plt.scatter(pandas_df['open'], pandas_df['close'], c=pandas_df['prediction'])
plt.xlabel('open')
plt.ylabel('Close')
plt.title('Scatter Plot with Clusters')
plt.show()