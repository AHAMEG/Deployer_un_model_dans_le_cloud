# Databricks notebook source
from pyspark import SQLContext
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.image import ImageSchema

from pyspark.sql.functions import udf
from pyspark.sql.functions import input_file_name
from pyspark.sql.types import *

# COMMAND ----------

def preprocess_data(dataframe):
    '''Renvoie le résultat de l'avant dernière couche de chaque image du dataframe via transform du ResNet50
    return un df contenant des vecteurs de dimension 1x2048 '''
    
    from sparkdl import DeepImageFeaturizer
    # DeepImageFeaturizer Applies the model specified by its popular name, 
    # with its prediction layer(s) chopped off
    featurizer = DeepImageFeaturizer(inputCol="image",outputCol="image_preprocessed", modelName="ResNet50")
    output = featurizer.transform(dataframe).select(['path', 'categorie', 'image_preprocessed'])
    del featurizer
    return output

# COMMAND ----------

# check the contents in tables folder
display(dbutils.fs.ls("/FileStore/tables"))

# COMMAND ----------

# pyspark functions
from pyspark.sql.functions import *

# URL processing "Package de gestion des url"
import urllib

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Lecture du fichier csv contenant les clés AWS databricks.
# MAGIC Nous spécifions que le type de fichier => "csv".
# MAGIC 
# MAGIC Nous indiquons que le fichier a la première ligne comme en-tête et une virgule comme délimiteur. 
# MAGIC 
# MAGIC Le chemin du fichier "csv" a été transmis pour charger le fichier.

# COMMAND ----------

# Define file type
file_type ="csv"

# Whether the file has a header 
first_row_is_header = "true"

# Delimiter used in the file 
delimiter =","

# Read the csv file to spark dataframe 
aws_keys_df = spark.read.format(file_type)\
.option("header", first_row_is_header)\
.option("sep", delimiter)\
.load("/FileStore/tables/AHAMEG_accessKeys.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Obtenez la clé d'accès et la clé secrète à partir de la trame de données Spark. 
# MAGIC 
# MAGIC => La clé secrète a été encodée à l'aide de urllib.parse.quote à des fins de sécurité. 
# MAGIC 
# MAGIC => safe="" signifie que chaque caractère de la clé secrète est encodé.

# COMMAND ----------

# Get the AWS access key and secret key from the spark dataframe 
ACCESS_KEY = aws_keys_df.select('Access key ID').collect()[0]['Access key ID'] # where(col('AHAMEG')=='AHAMEG_accessKeys')
SECRET_KEY = aws_keys_df.select('Secret access key').collect()[0]['Secret access key'] # where(col('AHAMEG')=='AHAMEG_accessKeys').

# Encode the secrete key
ENCODED_SECRET_KEY = urllib.parse.quote(string=SECRET_KEY, safe="")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Connexion de AWS_S3 à Databricks

# COMMAND ----------

# AWS s3 bucket name
AWS_S3_BUCKET = "db-ab98764274fceb6cdf2ec80674550417-s3-root-bucket"

# Mount name for the bucket
MOUNT_NAME = "/mnt/db-ab98764274fceb6cdf2ec80674550417-s3-root-bucket"

# Source url
SOURCE_URL = f"s3n://{ACCESS_KEY}:{ENCODED_SECRET_KEY}@{AWS_S3_BUCKET}"

## Mount the drive
#dbutils.fs.mount(SOURCE_URL, MOUNT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Vérification du contenu du compartiment AWS_S3

# COMMAND ----------

display(dbutils.fs.ls("/mnt/db-ab98764274fceb6cdf2ec80674550417-s3-root-bucket/ireland-prod/649518621194849/tmp/Photos/"))

# COMMAND ----------

import glob
import time
import io
import os
#import numpy as np
from PIL import Image

# COMMAND ----------

# MAGIC %md #### Lecture des données AWS_S3 à partir de Databricks. 
# MAGIC Affichage de la catégorie de chaque image et son chemin 

# COMMAND ----------

# File location and type
path ="/mnt/db-ab98764274fceb6cdf2ec80674550417-s3-root-bucket/ireland-prod/649518621194849/tmp/Photos/*"
file_type ="image"

#compteur
start = time.time()

# Read data from S3 to DataBricks
df = spark.read.format(file_type).load(path)
print ('loading done')

# recover path from images
df = df.withColumn("path", input_file_name())

def parse_categorie(path):
    '''Renvoie la catégorie d\'une image à partir de son chemin'''
    if len(path) > 0:
        #catégorie de l'image
        return path.split('/')[-2]
    else:
        return ''

#image categorization
udf_categorie = udf(parse_categorie, StringType())
df = df.withColumn('categorie', udf_categorie('path'))
print('Temps de chargement des images : {} secondes'.format(time.strftime('%S', time.gmtime(time.time()-start))))
      
# Take a look at the data 
df.show()

# COMMAND ----------

pip install tensorflow

# COMMAND ----------

import pandas as pd
from PIL import Image
import numpy as np
import io

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

from pyspark.sql.functions import col, pandas_udf, PandasUDFType

# COMMAND ----------

# MAGIC %md
# MAGIC ## Charger les données
# MAGIC Chargez des images à l'aide de la source de données de fichier binaire de Spark. On peut également utiliser la source de données d'image de Spark, mais la source de données de fichier binaire offre plus de flexibilité dans la façon dont on prétraite les images.

# COMMAND ----------

images = spark.read.format("binaryFile") \
  .option("pathGlobFilter", "*.jpg") \
  .option("recursiveFileLookup", "true") \
  .load("/mnt/db-ab98764274fceb6cdf2ec80674550417-s3-root-bucket/ireland-prod/649518621194849/tmp/Photos/*")

display(images.limit(50))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Préparation du modèle
# MAGIC Téléchargez le modèle ResNet50 en tronquant la dernière couche. 

# COMMAND ----------

model = ResNet50(include_top=False)
model.summary()  # vérifier que la dernière couche est supprimée

# COMMAND ----------

bc_model_weights = sc.broadcast(model.get_weights())

def model_fn():
  """
  Returns a ResNet50 model with top layer removed and broadcasted pretrained weights.
  """
  model = ResNet50(weights=None, include_top=False)
  model.set_weights(bc_model_weights.value)
  return model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Définition de la logique de chargement et de featurisation des images dans une UDF Pandas
# MAGIC pandas UDF
# MAGIC 
# MAGIC On utilise la nouvelle UDF Scalar Iterator pandas pour amortir le coût du chargement de modèles volumineux sur les nœuds de calcul.

# COMMAND ----------

def preprocess(content):
  """
  Preprocesses raw image bytes for prediction.
  """
  img = Image.open(io.BytesIO(content)).resize([224, 224])
  arr = img_to_array(img)
  return preprocess_input(arr)

def featurize_series(model, content_series):
  """
  Featurize a pd.Series of raw images using the input model.
  :return: a pd.Series of image features
  """
  input = np.stack(content_series.map(preprocess))
  preds = model.predict(input)
  # For some layers, output features will be multi-dimensional tensors.
  # We flatten the feature tensors to vectors for easier storage in Spark DataFrames.
  output = [p.flatten() for p in preds]
  return pd.Series(output)

# COMMAND ----------

@pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
def featurize_udf(content_series_iter):
  '''
  This method is a Scalar Iterator pandas UDF wrapping our featurization function.
  The decorator specifies that this returns a Spark DataFrame column of type ArrayType(FloatType).
  
  :param content_series_iter: This argument is an iterator over batches of data, where each batch
                              is a pandas Series of image data.
  '''
  # With Scalar Iterator pandas UDFs, we can load the model once and then re-use it
  # for multiple data batches.  This amortizes the overhead of loading big models.
  model = model_fn()
  for content_series in content_series_iter:
    yield featurize_series(model, content_series)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Appliquer la featurisation au DataFrame des images

# COMMAND ----------

# Pandas UDFs on large records (e.g., very large images) can run into Out Of Memory (OOM) errors.
# If you hit such errors in the cell below, try reducing the Arrow batch size via `maxRecordsPerBatch`.
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1024")

# COMMAND ----------

# We can now run featurization on our entire Spark DataFrame.
# NOTE: This can take a long time (about 10 minutes) since it applies a large model to the full dataset.
features_df = images.repartition(16).select(col("path"), featurize_udf("content").alias("features"))
features_df.write.mode("overwrite").parquet("dbfs:/mnt/db-ab98764274fceb6cdf2ec80674550417-s3-root-bucket/ireland-prod/649518621194849/tmp/photos_features/")

# COMMAND ----------

def write_dataframe_parquet(dataframe, path_parquet):
    '''Enregistrement du spark dataframe au format parquet au chemin specifie'''
    try:
        start = time.time()
        dataframe.write.format("parquet").mode('overwrite').save(path_parquet)
        print('Enregistrement effectué.')
            

        print('Temps de sauvegarde : {} secondes'.format(time.strftime('%S', time.gmtime(time.time()-start))))
    except:
        print('L\'Enregistrement a échoué.')
    
    return True

def load_dataframe_parquet(path):
    '''chargement du dataframe : entree parquet / sortie dataframe'''
    return spark.read.format('parquet').load('path')

# COMMAND ----------

#enregistrement des données (format parquet)
print('Enregistrement distant S3')
path_parquet="/mnt/db-ab98764274fceb6cdf2ec80674550417-s3-root-bucket/ireland-prod/649518621194849/tmp/photos_features/"
write_dataframe_parquet(features_df, path_parquet)   

# COMMAND ----------

features_df.show()

# COMMAND ----------

features_df.columns

# COMMAND ----------

from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors, VectorUDT
to_vector = udf(lambda x: Vectors.dense(x), VectorUDT())
features_df = features_df.withColumn('features_Vector', to_vector(features_df["features"]))



# COMMAND ----------

pca = PCA(k=3, inputCol='features_Vector', outputCol='features_pca')
model_pca = pca.fit(features_df)


# COMMAND ----------

SIZE=20
cumValues = model_pca.explainedVariance.cumsum() # get the cumulative values
import matplotlib.pyplot as plt
# plot the graph 
plt.figure(figsize=(10,8))
plt.plot(range(1,5), cumValues, marker = 'o', linestyle='--')
plt.title('Variance par composante', size=20)
plt.xlabel('Nombre de composantes')
plt.ylabel('Variancecumulée')
plt.rc('axes', labelsize=SIZE) 
plt.rc('xtick', labelsize=SIZE)
plt.rc('ytick', labelsize=SIZE) 
plt.rc('legend', fontsize=SIZE)

# COMMAND ----------

df_features_image_pca = model_pca.transform(features_df)

# COMMAND ----------

df_features_image_pca.show()

# COMMAND ----------

cols_to_drop = ['features', 'features_Vector'] 
final_df_features_image_pca = df_features_image_pca.drop(*cols_to_drop)
final_df_features_image_pca.show()

# COMMAND ----------

#enregistrement des données réduite avec un PCA
PCA_path="/mnt/db-ab98764274fceb6cdf2ec80674550417-s3-root-bucket/ireland-prod/649518621194849/tmp/photos_features_PCA/"
final_df_features_image_pca.write.mode('overwrite').save(PCA_path)

# COMMAND ----------

#enregistrement des données réduite avec un PCA (format parquet)
print('Enregistrement distant S3')
path_parquet="/mnt/db-ab98764274fceb6cdf2ec80674550417-s3-root-bucket/ireland-prod/649518621194849/tmp/photos_features/"
write_dataframe_parquet(features_df, path_parquet)   

# COMMAND ----------

#lecture des données réduites enregistrées (format parquet)
print('lecture distante S3')
df_parquet = spark.read.format('parquet').load(path_parquet) 
df_parquet.count()

# COMMAND ----------

df_parquet.show()

# COMMAND ----------

#lecture des données réduites enregistrées (format parquet)
print('lecture distante S3')
PCA_df_parquet = spark.read.load(PCA_path) 
PCA_df_parquet.count()

# COMMAND ----------

PCA_df_parquet.show()
