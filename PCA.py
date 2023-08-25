from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import PCA
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

# Cria o modelo PCA com 2 componentes principais
pca = PCA(k=2, inputCol='features', outputCol='pca_features')
model = pca.fit(va_df)

# Transforma os dados originais usando o modelo PCA
pca_df = model.transform(va_df)

# Converte o DataFrame do Spark para um Pandas DataFrame
pandas_df = pca_df.select('pca_features').toPandas()

# Extrai os componentes principais
pca_components = model.pc.toArray().reshape(3, 2)

# Plota os dados no espaço de componentes principais
plt.figure(figsize=(8, 6))
plt.scatter(pandas_df['pca_features'].apply(lambda x: x[0]), pandas_df['pca_features'].apply(lambda x: x[1]), c='b', marker='o')
plt.title('PCA: Principal Component Analysis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Plota os vetores de componente principal
for i, (comp1, comp2) in enumerate(pca_components):
    plt.arrow(0, 0, comp1, comp2, color='r', alpha=0.5, head_width=0.2)
    plt.text(comp1, comp2, f'Feature {i+1}', color='r')

plt.grid()
plt.show()

