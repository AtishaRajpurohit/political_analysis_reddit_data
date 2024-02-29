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

# COMMAND ----------

comments = spark.read.parquet("abfss://anly502@marckvaismanblob.dfs.core.windows.net/reddit/parquet/comments")

# COMMAND ----------

demo_comments = comments.filter(comments.subreddit == "democrats")
rep_comments = comments.filter(comments.subreddit == "Conservative")

# COMMAND ----------

demo_comments=demo_comments.select("body","subreddit")
rep_comments=rep_comments.select("body","subreddit")
demo_comments=demo_comments.filter(col("body") != "[removed]")
rep_comments=rep_comments.filter(col("body") != "[removed]")
demo_comments=demo_comments.filter(col("body") != "[deleted]")
rep_comments=rep_comments.filter(col("body") != "[deleted]")

# COMMAND ----------

#df = rep_comments.union(demo_comments)
#df = df.dropDuplicates()

# COMMAND ----------

#df = df.cache()

# COMMAND ----------

#df.groupBy('subreddit').count().orderBy('count').show()

# COMMAND ----------

rep_comments=rep_comments.limit(40000)

# COMMAND ----------

demo_comments=demo_comments.limit(40000)

# COMMAND ----------

df = rep_comments.union(demo_comments)
df = df.dropDuplicates()

# COMMAND ----------

#data is imbalanced
df = df.sample(.3)

# COMMAND ----------

from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
# regular expression tokenizer
regexTokenizer = RegexTokenizer(inputCol="body", outputCol="words", pattern="\\W")
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered")
# bag of words count
countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
label_stringIdx = StringIndexer(inputCol = "subreddit", outputCol = "label")
pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, label_stringIdx])
# Fit the pipeline to training documents.
pipelineFit = pipeline.fit(df)
dataset = pipelineFit.transform(df)
#dataset.show(5)

# COMMAND ----------

dataset = dataset.cache()

# COMMAND ----------

# set seed for reproducibility
(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed = 100)

# COMMAND ----------

lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
lrModel = lr.fit(trainingData)
predictions = lrModel.transform(testData)
predictions.filter(predictions['prediction'] == "0") \
    .select("body","subreddit","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='label')
evaluator.evaluate(predictions)

# COMMAND ----------

train_fit_lr = predictions.select('label','prediction')
train_fit_lr.groupBy('label','prediction').count().show()

# COMMAND ----------

evaluatorRF = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC")
roc_result = evaluatorRF.evaluate(predictions)
roc_result

# COMMAND ----------

import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
plt.plot([0, 1], [0, 1], 'r--')
plt.plot(lrModel.summary.roc.select('FPR').collect(),
         lrModel.summary.roc.select('TPR').collect())
plt.xlabel('False Positive Rare')
plt.ylabel('True Positive Rate')
plt.title("ROC Curve")
plt.show()
plt.savefig('/Workspace/Repos/jpp82@georgetown.edu/fall-2022-reddit-big-data-project-project-group-30/data/plots/ROC_curveJack.png')
# plt.savefig('../data/plots/ROC_curveJack.png')
# ../data/plots/

# COMMAND ----------

import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification


fpr = lrModel.summary.roc.select('FPR').rdd.flatMap(lambda x: x).collect()
tpr = lrModel.summary.roc.select('TPR').rdd.flatMap(lambda x: x).collect()

fig = px.area(
    x=fpr, y=tpr,
#     title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.update_layout(title_text=f'ROC Curve (AUC={auc(fpr, tpr):.4f})', title_x=0.5)
fig.show()
fig.write_html("../data/plots/ROC_curveJack.html")

# COMMAND ----------

y_true = predictions.select(['label']).collect()
y_pred = predictions.select(['prediction']).collect()

# COMMAND ----------

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_true, y_pred))

# COMMAND ----------

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm=confusion_matrix(y_true, y_pred)
cmp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Conservative","democrat"])
fig, ax = plt.subplots(figsize=(10,10))
cmp.plot(ax=ax)

# COMMAND ----------

display_labels=["Conservative","democrat"]

# COMMAND ----------

cmp.figure_.savefig('/Workspace/Repos/jpp82@georgetown.edu/fall-2022-reddit-big-data-project-project-group-30/data/plots/confusion_matrix_jack.png')

# COMMAND ----------

# import plotly.express as px
# data=[[2554, 1021],[1630, 1761]]
# fig = px.imshow(data,
#                 labels=dict(x="Day of Week", y="Time of Day", color="Productivity"),
#                 x=["Conservative","democrat"],
#                 y=["Conservative","democrat"]
#                )
# # fig.update_xaxes(side="bottom")
# fig.show()

# COMMAND ----------

# import plotly.graph_objects as go
# def plot_confusion_matrix(cm, labels, title):
#     # cm : confusion matrix list(list)
#     # labels : name of the data list(str)
#     # title : title for the heatmap
#     data = go.Heatmap(z=cm, y=labels, x=labels)
#     annotations = []
#     for i, row in enumerate(cm):
#         for j, value in enumerate(row):
#             annotations.append(
#                 {
#                     "x": labels[i],
#                     "y": labels[j],
#                     "font": {"color": "white"},
#                     "text": str(value),
#                     "xref": "x1",
#                     "yref": "y1",
#                     "showarrow": False
#                 }
#             )
#     layout = {
#         "title": title,
#         "xaxis": {"title": "Predicted value"},
#         "yaxis": {"title": "Real value"},
#         "annotations": annotations
#     }
#     fig = go.Figure(data=data, layout=layout)
#     return fig

# plot_confusion_matrix(cm, ["Conservative","democrat"], 'Confusion ')

# COMMAND ----------

#change maxIter to 100 and regParam to .8
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(maxIter=100, regParam=0.8, elasticNetParam=0)
lrModel = lr.fit(trainingData)
predictions = lrModel.transform(testData)
predictions.filter(predictions['prediction'] == "0") \
    .select("body","subreddit","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)

# COMMAND ----------

#accuracy increased by .001
evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='label')
evaluator.evaluate(predictions)

# COMMAND ----------

evaluatorRF = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC")
roc_result = evaluatorRF.evaluate(predictions)
roc_result

# COMMAND ----------

y_true = predictions.select(['label']).collect()
y_pred = predictions.select(['prediction']).collect()

# COMMAND ----------

cm=confusion_matrix(y_true, y_pred)
cmp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Conservative","democrat"])
fig, ax = plt.subplots(figsize=(10,10))
cmp.plot(ax=ax)

# COMMAND ----------

cmp.figure_.savefig('/Workspace/Repos/jpp82@georgetown.edu/fall-2022-reddit-big-data-project-project-group-30/data/plots/confusion_matrix2_jack.png')

# COMMAND ----------

fpr = lrModel.summary.roc.select('FPR').rdd.flatMap(lambda x: x).collect()
tpr = lrModel.summary.roc.select('TPR').rdd.flatMap(lambda x: x).collect()

fig = px.area(
    x=fpr, y=tpr,
#     title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.update_layout(title_text=f'ROC Curve (AUC={auc(fpr, tpr):.4f})', title_x=0.5)
fig.show()
fig.write_html("../data/plots/ROC_curve2Jack.html")

# COMMAND ----------


