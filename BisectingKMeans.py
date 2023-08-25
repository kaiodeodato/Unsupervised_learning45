from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

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

# Cria um modelo BisectingKMeans com k=5 clusters
bkm = BisectingKMeans(seed=41)

# Treina o modelo BisectingKMeans com as features
model = bkm.fit(va_df)

# Aplica o modelo BisectingKMeans ao DataFrame
transformed = model.transform(va_df)

# Mostra os resultados sem truncamento
transformed.show(truncate=False)

# Visualiza a distribuição entre os clusters
transformed.groupBy('prediction').count().orderBy('prediction').show()
