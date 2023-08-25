from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

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

# Cria uma matriz de ligação usando o método 'ward'
Z = linkage(pandas_df[['open', 'low', 'close']], method='ward')

# Cria o dendrograma
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title('Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
