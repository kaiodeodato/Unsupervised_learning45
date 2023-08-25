from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram

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

# Cria o heatmap da matriz de distância agrupada
sns.set(font_scale=1)
sns.clustermap(pandas_df[['open', 'low', 'close']], method='ward', cmap='coolwarm', figsize=(8, 6))
plt.show()
