from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
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

# Cria um modelo KMeans com k=5 clusters
kmeans = KMeans(k=4, seed=42)

# Treina o modelo KMeans com as features
model = kmeans.fit(va_df.select('features'))

# Aplica o modelo KMeans ao DataFrame
transformed = model.transform(va_df)

# Avalia a qualidade dos clusters usando o coeficiente de silhueta
evaluator = ClusteringEvaluator()
silhouette_score = evaluator.evaluate(transformed)
print("Silhouette Score:", silhouette_score)
