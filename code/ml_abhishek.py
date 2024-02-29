# Databricks notebook source
from pyspark.sql.functions import *
import pandas as pd
import numpy as np
import json
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from sparknlp.annotator import *
from sparknlp.base import *
import sparknlp
from sparknlp.pretrained import PretrainedPipeline
from pyspark.ml.feature import Tokenizer,StopWordsRemover,CountVectorizer,IDF,StringIndexer
from pyspark.ml.classification import LogisticRegression

# COMMAND ----------

submissions = spark.read.parquet("abfss://anly502@marckvaismanblob.dfs.core.windows.net/reddit/parquet/submissions")

# COMMAND ----------

politics_sub = submissions.filter(submissions.subreddit == "politics")

# COMMAND ----------

politics_sub.printSchema()

# COMMAND ----------

comments = spark.read.parquet("abfss://anly502@marckvaismanblob.dfs.core.windows.net/reddit/parquet/comments")
politics_com = comments.filter(comments.subreddit == "politics")

# COMMAND ----------

politics_com.printSchema()

# COMMAND ----------

politics_com.groupBy('controversiality').count().show()

# COMMAND ----------

politics_com_limit=politics_com.limit(10000)

# COMMAND ----------

politics_com_limit=politics_com_limit.filter(col("body") != "[removed]")
politics_com_limit=politics_com_limit.filter(col("body") != "[deleted]")

# COMMAND ----------

politics_com_limit = politics_com_limit.dropDuplicates()

# COMMAND ----------

politics_com_limit = politics_com_limit.select("body", "controversiality")

# COMMAND ----------

tokenizer = Tokenizer(inputCol='body',outputCol='mytokens')
stopwords_remover = StopWordsRemover(inputCol='mytokens',outputCol='filtered_tokens')
vectorizer = CountVectorizer(inputCol='filtered_tokens',outputCol='rawFeatures')
idf = IDF(inputCol='rawFeatures',outputCol='vectorizedFeatures')

# COMMAND ----------

(trainDF,testDF) = politics_com_limit.randomSplit((0.7,0.3),seed=100)

# COMMAND ----------

lr = LogisticRegression(featuresCol='vectorizedFeatures',labelCol='controversiality')

# COMMAND ----------

pipeline = Pipeline(stages=[tokenizer,stopwords_remover,vectorizer,idf,lr])

# COMMAND ----------

lr_model = pipeline.fit(trainDF)

# COMMAND ----------

predictions = lr_model.transform(testDF)

# COMMAND ----------

predictions

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol='controversiality',predictionCol='prediction',metricName='accuracy')
accuracy = evaluator.evaluate(predictions)
print("the accuracy of the model is: ", accuracy*100,"%")

# COMMAND ----------

evaluator = MulticlassClassificationEvaluator(labelCol='controversiality',predictionCol='prediction',metricName='f1')
f1 = evaluator.evaluate(predictions)
print("the f1 score of the model is: ", f1)

# COMMAND ----------

evaluator = MulticlassClassificationEvaluator(labelCol='controversiality',predictionCol='prediction',metricName='precisionByLabel')
precision = evaluator.evaluate(predictions)
print("the precision of the model is: ", precision)

# COMMAND ----------

evaluator = MulticlassClassificationEvaluator(labelCol='controversiality',predictionCol='prediction',metricName='recallByLabel')
recall = evaluator.evaluate(predictions)
print("the recall of the model is: ", recall)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

evaluator = BinaryClassificationEvaluator(labelCol='controversiality',rawPredictionCol='prediction',metricName='areaUnderROC')
areaROC = evaluator.evaluate(predictions)

print("the Area under the ROC: ", areaROC)

# COMMAND ----------

y_true = predictions.select("controversiality")
y_true = y_true.toPandas()

y_pred = predictions.select("prediction")
y_pred = y_pred.toPandas()

# COMMAND ----------

len(y_true)

# COMMAND ----------

len(y_pred)

# COMMAND ----------

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
cm=confusion_matrix(y_true['controversiality'], y_pred['prediction'])
cmp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["0","1"])
fig, ax = plt.subplots(figsize=(10,10))
cmp.plot(ax=ax)

# COMMAND ----------

cmp.figure_.savefig('/Workspace/Repos/ad1728@georgetown.edu/fall-2022-reddit-big-data-project-project-group-30/data/plots/confusion_matrix_abhishek.png')
