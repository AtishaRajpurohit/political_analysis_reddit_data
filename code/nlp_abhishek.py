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
import pyspark.sql.functions as f

# COMMAND ----------

comments = spark.read.parquet("abfss://anly502@marckvaismanblob.dfs.core.windows.net/reddit/parquet/comments")
politics_comments = comments.filter(comments.subreddit == "politics")


# COMMAND ----------

def remove_deleted_comments(df):
    # deleting rows that don't have authors or body anymore
    df = df.filter(col("author") != "[deleted]")
    df = df.filter(col("body") != "[deleted]")
    df = df.filter(col("body") != "[removed]")
    return df


# COMMAND ----------

converted_political_comments = remove_deleted_comments(politics_comments)
converted_political_comments = converted_political_comments.select("*", f.lower("body"))

# COMMAND ----------

converted_political_comments.show()

# COMMAND ----------

converted_political_comments=converted_political_comments.select("lower(body)")

# COMMAND ----------

converted_political_comments.show()

# COMMAND ----------

converted_political_comments_2 = converted_political_comments.withColumn("Biden", when(col('lower(body)')\
        .rlike("(?i)joe|(?i)biden|(?i)joe biden"), lit(True)) \
        .otherwise(False))\
        .withColumn("Trump", when(col('lower(body)')\
        .rlike('(?i)trump|(?i)donald|(?i)donald trump'), True)\
        .otherwise(False))\
        .withColumn("Sanders", when(col('lower(body)')\
        .rlike('(?i)bernie|(?i)sanders'), True)\
        .otherwise(False))\
        .withColumn("Buttigieg", when(col('lower(body)')\
        .rlike('(?i)buttigieg|(?i)pete'), True)\
        .otherwise(False))\
        .withColumn("AOC", when(col('lower(body)')\
        .rlike('(?i)aoc|(?i)ocasio-cortez|(?i)alexandria'), True)\
        .otherwise(False))\
        .withColumn("Harris", when(col('lower(body)')\
        .rlike('(?i)kamala|(?i)harris'), True)\
        .otherwise(False))\
        .withColumn("DeSantis", when(col('lower(body)')\
        .rlike('(?i)ron|(?i)desantis'), True)\
        .otherwise(False))\
        .withColumn("Youngkin", when(col('lower(body)')\
        .rlike('(?i)glenn|(?i)youngkin'), True)\
        .otherwise(False))\
        .withColumn("Scott", when(col('lower(body)')\
        .rlike('(?i)tim|(?i)scott'), True)\
        .otherwise(False))\

converted_political_comments_2.show()

# COMMAND ----------

spark = SparkSession.builder \
        .appName("SparkNLP") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:4.2.1") \
    .master('yarn') \
    .getOrCreate()



MODEL_NAME='sentimentdl_use_twitter'



documentAssembler = DocumentAssembler()\
    .setInputCol("lower(body)")\
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

pipelineModel = nlpPipeline.fit(converted_political_comments_2)
result = pipelineModel.transform(converted_political_comments_2)

# COMMAND ----------

output = result.select('lower(body)','Biden','Trump','Sanders','Buttigieg','AOC','Harris','DeSantis','Youngkin','Scott', explode('sentiment.result'))

# COMMAND ----------

biden_comments = output.filter(output.Biden == True)
trump_comments = output.filter(output.Trump == True)
sanders_comments = output.filter(output.Sanders == True)
buttigieg_comments = output.filter(output.Buttigieg == True)
aoc_comments = output.filter(output.AOC == True)
harris_comments = output.filter(output.Harris == True)
desantis_comments = output.filter(output.DeSantis == True)
youngkin_comments = output.filter(output.Youngkin == True)
scott_comments = output.filter(output.Scott == True)



# COMMAND ----------

# GET NEGATIVE COUNTS
biden_neg_count = biden_comments.filter(biden_comments.col == "negative").count()
trump_neg_count = trump_comments.filter(trump_comments.col == "negative").count()
sanders_neg_count = sanders_comments.filter(sanders_comments.col == "negative").count()
buttigieg_neg_count = buttigieg_comments.filter(buttigieg_comments.col == "negative").count()
aoc_neg_count = aoc_comments.filter(aoc_comments.col == "negative").count()
harris_neg_count = harris_comments.filter(harris_comments.col == "negative").count()
desantis_neg_count = desantis_comments.filter(desantis_comments.col == "negative").count()
youngkin_neg_count = youngkin_comments.filter(youngkin_comments.col == "negative").count()
scott_neg_count = scott_comments.filter(scott_comments.col == "negative").count()


# GET POSITIVE COUNTS
biden_pos_count = biden_comments.filter(biden_comments.col == "positive").count()
trump_pos_count = trump_comments.filter(trump_comments.col == "positive").count()
sanders_pos_count = sanders_comments.filter(sanders_comments.col == "positive").count()
buttigieg_pos_count = buttigieg_comments.filter(buttigieg_comments.col == "positive").count()
aoc_pos_count = aoc_comments.filter(aoc_comments.col == "positive").count()
harris_pos_count = harris_comments.filter(harris_comments.col == "positive").count()
desantis_pos_count = desantis_comments.filter(desantis_comments.col == "positive").count()
youngkin_pos_count = youngkin_comments.filter(youngkin_comments.col == "positive").count()
scott_pos_count = scott_comments.filter(scott_comments.col == "positive").count()


# GET NUETRAL COUNTS
biden_neu_count = biden_comments.filter(biden_comments.col == "neutral").count()
trump_neu_count = trump_comments.filter(trump_comments.col == "neutral").count()
sanders_neu_count = sanders_comments.filter(sanders_comments.col == "neutral").count()
buttigieg_neu_count = buttigieg_comments.filter(buttigieg_comments.col == "neutral").count()
aoc_neu_count = aoc_comments.filter(aoc_comments.col == "neutral").count()
harris_neu_count = harris_comments.filter(harris_comments.col == "neutral").count()
desantis_neu_count = desantis_comments.filter(desantis_comments.col == "neutral").count()
youngkin_neu_count = youngkin_comments.filter(youngkin_comments.col == "neutral").count()
scott_neu_count = scott_comments.filter(scott_comments.col == "neutral").count()

# COMMAND ----------


data = {'biden': [biden_neg_count, biden_pos_count, biden_neu_count],
        'trump': [trump_neg_count, trump_pos_count, trump_neu_count],
        'sanders':[sanders_neg_count, sanders_pos_count, sanders_neu_count],
        'buttigieg':[buttigieg_neg_count, buttigieg_pos_count, buttigieg_neu_count],
        'aoc':[aoc_neg_count, aoc_pos_count, aoc_neu_count],
        'harris':[harris_neg_count, harris_pos_count, harris_neu_count],
        'desantis':[desantis_neg_count, desantis_pos_count, desantis_neu_count],
        'youngkin':[youngkin_neg_count, youngkin_pos_count, youngkin_neu_count],
        'scott':[scott_neg_count, scott_pos_count, scott_neu_count],
       }
df = pd.DataFrame.from_dict(data)
df.to_csv('out_candidates.csv', index=False)

# COMMAND ----------

import plotly.graph_objects as go
from plotly.subplots import make_subplots

colors = ['gold', 'mediumturquoise', 'darkorange']
labels = ['Negative', 'Positive', 'Neutral']

fig = make_subplots(2, 3, specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]],
                    subplot_titles=['Biden (Democrat)', 'Harris (Democrat)', 'Sanders (Democrat)','Trump (Republican)','DeSantis (Republican)','Youngkin (Republican)'])
fig.add_trace(go.Pie(labels=labels, values=[499652,314771,46445], scalegroup='one',
                     name="Biden",marker_colors=['lightcyan','cyan','royalblue']), 1, 1)
fig.add_trace(go.Pie(labels=labels, values=[31571,21508,47384], scalegroup='one',
                     name="Harris"), 1, 2)
fig.add_trace(go.Pie(labels=labels, values=[78441,48138,10340], scalegroup='one',
                     name="Sanders"), 1, 3)
fig.add_trace(go.Pie(labels=labels, values=[1182654,563074,90344], scalegroup='one',
                     name="Trump",marker_colors=['lightcyan','cyan','royalblue']), 2, 1)
fig.add_trace(go.Pie(labels=labels, values=[587163,361944,47384], scalegroup='one',
                     name="DeSantis"), 2, 2)
fig.add_trace(go.Pie(labels=labels, values=[4833,2618,501], scalegroup='one',
                     name="Youngkin"), 2, 3)

fig.update_layout(title_text='Sentiment of Comments for Popular Candidates',autosize=False,width=1000,height=600)
fig.show()
fig.write_html("../data/plots/PopularCandidates.html")

# COMMAND ----------


