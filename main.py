# Importando os recursos necessários
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

# inicío a sessão do spark
spark = SparkSession.builder.appName("app_model").getOrCreate()
# importo o arquivo com o delimitador "," e com cabeçalho 
df = spark.read.option("delimiter", ",").option("header", "true").csv("stocks_2021.csv")

# converto as colunas em float
df = df.withColumn('open', df.open.cast('float'))
df = df.withColumn('low', df.low.cast('float'))
df = df.withColumn('close', df.close.cast('float'))

# Cria um VectorAssembler para agrupar as colunas 'open', 'low' e 'close' em uma única coluna 'features' 
va = VectorAssembler(inputCols=['open','low','close'], outputCol='features')
# Aplico o VectorAssembler ao DataFrame
va_df = va.transform(df)
# Crio um modelo KMeans com k=5 clusters

kmeans = KMeans(k=5)

# Treino o modelo KMeans com as features
model = kmeans.fit(va_df.select('features'))
# Aplico o modelo KMeans ao DataFrame
transformed = model.transform(va_df)
# mostro os resultados sem serem truncados
transformed.show(truncate=False)

# Como os primeiros 20 valores mostrados possuem todos prediction 0
# visualizo a distribuição entre os clusters
transformed.groupBy('prediction').count().orderBy('prediction').show()
