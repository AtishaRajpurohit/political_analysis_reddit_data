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

#read external dataset in
#approval = pd.read_csv("/Workspace/Repos/jpp82@georgetown.edu/fall-2022-reddit-big-data-project-project-group-30/data/csv/approval_topline.csv")
approval = pd.read_csv("data/csv/approval_topline.csv")
#convert df to spark df
approval=spark.createDataFrame(approval) 

# COMMAND ----------

politics_comments = comments.filter(comments.subreddit == "politics")

# COMMAND ----------

#only keep a few cols
politics_comments=politics_comments.select("author", "body",'created_utc','id')

# COMMAND ----------

politics_comments.show(3)

# COMMAND ----------

from pyspark.sql.types import StringType,BooleanType,DateType
#convert unixtime to datetime
politics_comments=politics_comments.withColumn("created_utc", from_unixtime(col("created_utc"),"M/d/yyyy"))
#convert to DateType
politics_comments=politics_comments.withColumn("created_utc",to_date(col("created_utc"),"M/d/yyyy"))

# COMMAND ----------

#convert to Datetype
approval=approval.withColumn("date",to_date(col("modeldate"),"M/d/yyyy"))

# COMMAND ----------

 approval=approval.filter(approval.subgroup == "All polls")

# COMMAND ----------

approval.show(5)

# COMMAND ----------

#app=approval.toPandas()

# COMMAND ----------

#app.to_csv("/Workspace/Repos/jpp82@georgetown.edu/fall-2022-reddit-big-data-project-project-group-30/data/csv/approval.csv",index=False)

# COMMAND ----------

#join approval rating dataset with comments dataset on date
approval_df=approval.join(politics_comments,approval.date ==  politics_comments.created_utc,"inner")

# COMMAND ----------

#drop dups
approval_df=approval_df.drop_duplicates()

# COMMAND ----------

approval_df=approval_df.select("author", "body",'date','id','approve_estimate')

# COMMAND ----------

spark = SparkSession.builder \
        .appName("SparkNLP") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:4.2.1") \
    .master('yarn') \
    .getOrCreate()

# COMMAND ----------

MODEL_NAME='sentimentdl_use_twitter'
documentAssembler = DocumentAssembler()\
    .setInputCol("body")\
    .setOutputCol("document")
    
use = UniversalSentenceEncoder.pretrained(name="tfhub_use", lang="en")\
 .setInputCols(["document"])\
 .setOutputCol("sentence_embeddings")


sentimentdl = SentimentDLModel.pretrained(name=MODEL_NAME, lang="en")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("sentiment")

nlpPipeline = Pipeline(
      stages = [
          documentAssembler,
          use,
          sentimentdl
      ])

# COMMAND ----------

pipelineModel = nlpPipeline.fit(approval_df)
result = pipelineModel.transform(approval_df)

# COMMAND ----------

result.select('body','author','date','approve_estimate','id', explode('sentiment.result')).show(20)

# COMMAND ----------

sent_df = result.select('body','author','id','date', 'approve_estimate', explode('sentiment.result'))

# COMMAND ----------

sent_df=sent_df.filter(col("author") != "[deleted]")
sent_df=sent_df.filter(col("body") != "[removed]")

# COMMAND ----------

sent_df = sent_df.filter((sent_df.col == "positive") | (sent_df.col == "negative"))

# COMMAND ----------

#count number of comments each day
sent_df=sent_df.groupBy("date","approve_estimate","col").agg(count("id").alias("num_comms"))

# COMMAND ----------

from pyspark.sql.window import Window
sent_df = sent_df.withColumn('com_pct', col('num_comms')/sum('num_comms').over(Window.partitionBy(['Date']))*100)

# COMMAND ----------

sent_df = sent_df.cache()

# COMMAND ----------

sent_df.filter((sent_df.date == "positive")).show()

# COMMAND ----------

sent_df.show()

# COMMAND ----------

sent_df2=sent_df.toPandas()

# COMMAND ----------

sent_df2.head()

# COMMAND ----------

sent_df2 = sent_df2[sent_df2['col']=="positive"]

# COMMAND ----------

sent_df2.head()

# COMMAND ----------

#sent_df2.to_csv("/Workspace/Repos/jpp82@georgetown.edu/fall-2022-reddit-big-data-project-project-group-30/data/csv/sent_df.csv",index=False)
sent_df2.to_csv("data/csv/sent_df.csv",index=False)

# COMMAND ----------

sent_df2 = pd.read_csv("/Workspace/Repos/jpp82@georgetown.edu/fall-2022-reddit-big-data-project-project-group-30/data/csv/sent_df.csv")

# COMMAND ----------

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(figsize=(10, 6), dpi=80)
ax = plt.gca()
ax.set_title('Approval Rating Correlation to Positive Comments')
sent_df2.plot(x="date", y="com_pct",ax=ax,label="Number of Positive Comments")
sent_df2.plot(x="date", y="approve_estimate",ax=ax,label="Approval Rating")
ax.set_xlabel("Date")
#plt.savefig('/Workspace/Repos/jpp82@georgetown.edu/fall-2022-reddit-big-data-project-project-group-30/data/plots/time_series_correlation_pos.png')
#plt.savefig('data/plots/time_series_correlation_pos.png')

# COMMAND ----------

# Using graph_objects
import plotly.graph_objects as go

import pandas as pd
fig = go.Figure([go.Scatter(x=sent_df2['date'], y=sent_df2['com_pct'],name="Calculated Approval Rating")])
fig.add_trace(go.Scatter(x=sent_df2['date'], y=sent_df2['approve_estimate'],name="Approval Rating"))
fig.update_layout(
        title='Approval Rating Compared to Calculated Approval Rating',
        yaxis_title='Approval Rating',
        xaxis_title='Time')
fig.show()
fig.write_html("../data/plots/time_series_correlation.html")

# COMMAND ----------


