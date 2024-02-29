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

# retrieve all comments from the r/politics subreddit
politics_comments = comments.filter(comments.subreddit == "politics")

# retrieve all comments from the r/Conservative subreddit
conservative_comments = comments.filter(comments.subreddit == "Conservative")

# retrieve all submissions from the r/democrats subreddit
democrats_comments = comments.filter(comments.subreddit == "democrats")

# COMMAND ----------

#only keep a few cols
politics_comments=politics_comments.select("author", "body",'created_utc','id').cache()
conservative_comments=conservative_comments.select("author", "body",'created_utc','id').cache()
democrats_comments=democrats_comments.select("author", "body",'created_utc','id').cache()

# COMMAND ----------

politics_comments.show(3)

# COMMAND ----------

conservative_comments.show(3)

# COMMAND ----------

democrats_comments.show(3)

# COMMAND ----------

def remove_deleted_comments(df):
    # deleting rows that don't have a body anymore
    df = df.filter(col("body") != "[deleted]")
    df = df.filter(col("body") != "[removed]")
    return df

# COMMAND ----------

politics_comments = remove_deleted_comments(politics_comments)
conservative_comments = remove_deleted_comments(conservative_comments)
democrats_comments = remove_deleted_comments(democrats_comments)

# COMMAND ----------

# print("Rows:",
#      '\npolitics =',politics_comments.count(),
#      '\nconservative =',conservative_comments.count(),
#      '\ndemocrat =',democrats_comments.count())

# COMMAND ----------

democrats_comments.show(3)

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

pol_pipelineModel = nlpPipeline.fit(politics_comments)
pol_result = pol_pipelineModel.transform(politics_comments)
pol_result = pol_result.select('body','author','id', explode('sentiment.result'))
pol.result.show(20)

# COMMAND ----------

con_pipelineModel = nlpPipeline.fit(conservative_comments)
con_result = con_pipelineModel.transform(conservative_comments)
con_result = con_result.select('body','author','id', explode('sentiment.result'))
con_result.show(20)

# COMMAND ----------

dem_pipelineModel = nlpPipeline.fit(democrats_comments)
dem_result = dem_pipelineModel.transform(democrats_comments)
dem_result = dem_result.select('body','author','id', explode('sentiment.result'))
dem_result.show(20)

# COMMAND ----------

dem_df = dem_result.groupBy("col").agg(count("col")).show().alias('count').toPandas()
#         .withColumn("day_night", F.when(col("day_night")=="true", "Day").otherwise("Night")).show()
con_df = con_result.groupBy("col").agg(count("col")).show().alias('count').toPandas()
pol_df = pol_result.groupBy("col").agg(count("col")).show().alias('count').toPandas()

# COMMAND ----------

# # GET TOTAL COMMENT COUNTS
# dem_count = dem_result.count()
# pol_count = pol_result.count()
# con_count = con_result.count()

# COMMAND ----------

# # GET NEGATIVE COUNTS
# dem_neg_count = dem_result.filter(dem_result.col == "negative").count()
# pol_neg_count = pol_result.filter(pol_result.col == "negative").count()
# con_neg_count = con_result.filter(con_result.col == "negative").count()


# # GET POSITIVE COUNTS
# dem_pos_count = dem_result.filter(dem_result.col == "positive").count()
# pol_pos_count = pol_result.filter(pol_result.col == "positive").count()
# con_pos_count = con_result.filter(con_result.col == "positive").count()


# # GET NEUTRAL COUNTS
# dem_neutral_count = dem_result.filter(dem_result.col == "neutral").count()
# pol_neutral_count = pol_result.filter(pol_result.col == "neutral").count()
# con_neutral_count = con_result.filter(con_result.col == "neutral").count()

# COMMAND ----------

print("TOTAL Comment count:",
     '\npolitics =',dem_count,
     '\nconservative =',pol_count,
     '\ndemocrat =',con_count)
print('---------------------------------------------------------------------\n')
print("NEGATIVE Comment count:",
     '\npolitics =',dem_neg_count,
     '\nconservative =',pol_neg_count,
     '\ndemocrat =',con_neg_count)
print('---------------------------------------------------------------------\n')
print("POSITIVE Comment count:",
     '\npolitics =',dem_pos_count,
     '\nconservative =',pol_pos_count,
     '\ndemocrat =',con_pos_count)
print('---------------------------------------------------------------------\n')
print("NEUTRAL Comment count:",
     '\npolitics =',dem_neutral_count,
     '\nconservative =',pol_neutral_count,
     '\ndemocrat =',con_neutral_count)
print('---------------------------------------------------------------------\n\n\n')

# GET PERCENT BREAKDOWN ACCORDING TO TOTAL COMMENT COUNT
print('--PERCENT BREAKDOWN PER SUBREDDIT--')
print("NEGATIVE Comment percent:",
     '\npolitics =',dem_neg_count/dem_count,
     '\nconservative =',pol_neg_count/pol_count,
     '\ndemocrat =',con_neg_count/con_count)
print('---------------------------------------------------------------------\n')
print("POSITIVE Comment percent:",
     '\npolitics =',dem_pos_count/dem_count,
     '\nconservative =',pol_pos_count/pol_count,
     '\ndemocrat =',con_pos_count/con_count)
print('---------------------------------------------------------------------\n')
print("NEUTRAL Comment percent:",
     '\npolitics =',dem_neutral_count/dem_count,
     '\nconservative =',pol_neutral_count/pol_count,
     '\ndemocrat =',con_neutral_count/con_count)
print('---------------------------------------------------------------------\n')

# COMMAND ----------

import plotly.graph_objects as go
from plotly.subplots import make_subplots

colors = ['gold', 'mediumturquoise', 'darkorange']
labels = ['Negative', 'Positive', 'Neutral']

fig = make_subplots(1, 3, specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]],
                    subplot_titles=['r/Politics', 'r/Conservative', 'r/Democrat'])
fig.add_trace(go.Pie(labels=labels, values=[139792,105612,13493], scalegroup='one',
                     name="r/Politics",marker_colors=['lightcyan','cyan','royalblue']), 1, 1)
fig.add_trace(go.Pie(labels=labels, values=[12396063,8993337,1172907], scalegroup='one',
                     name="r/Conservative"), 1, 2)
fig.add_trace(go.Pie(labels=labels, values=[2294712,1838989,232664], scalegroup='one',
                     name="r/Democrat"), 1, 3)

fig.update_layout(title_text='Sentiment of Comments in Political Subreddits',autosize=False,width=1000,height=600)
fig.show()
fig.write_html("../data/plots/SubSentimentBreakdown.html")

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

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
#normalized rating and comments to have a uniform scale
figure(figsize=(10, 6), dpi=80)
ax = plt.gca()
ax.set_title('Approval Rating Correlation to Positive Comments')
sent_df2.plot(x="date", y="com_pct",ax=ax,label="Number of Positive Comments")
sent_df2.plot(x="date", y="approve_estimate",ax=ax,label="Approval Rating")
ax.set_xlabel("Date")
#plt.savefig('/Workspace/Repos/jpp82@georgetown.edu/fall-2022-reddit-big-data-project-project-group-30/data/plots/time_series_correlation_pos.png')
plt.savefig('data/plots/time_series_correlation_pos.png')
