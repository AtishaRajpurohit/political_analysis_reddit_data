# Databricks notebook source
pip install wordcloud

# COMMAND ----------

#Question - To find out the the differences in reactions and sentiments of democrats and republicans over some of the important political events over the year
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
from pyspark.sql.functions import lit
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# COMMAND ----------

comments = spark.read.parquet("abfss://anly502@marckvaismanblob.dfs.core.windows.net/reddit/parquet/comments")
submissions = spark.read.parquet("abfss://anly502@marckvaismanblob.dfs.core.windows.net/reddit/parquet/submissions")

# COMMAND ----------

comments=comments.filter(col("author") != "[deleted]")
comments=comments.filter(col("body") != "[removed]")

# COMMAND ----------

##Creating subsets for demorats and republicans
#Politics Subreddit
politics_comments=comments.filter(comments.subreddit=="politics")
politics_submissions = submissions.filter(submissions.subreddit == "politics")

#Conservative Subreddits
conservative_comments = comments.filter(comments.subreddit == "Conservative")
conservative_submissions = submissions.filter(submissions.subreddit == "Conservative")

#Democratic Subreddits
democrats_comments = comments.filter(comments.subreddit == "democrats")
democrats_submissions = submissions.filter(submissions.subreddit == "democrats")

# COMMAND ----------

#Filtering the columns
politics_submissions_base=politics_submissions.select("author","title","num_comments","score","created_utc").cache()
politics_comments_base=politics_comments.select("author","body","controversiality","score","created_utc").cache()

#Conservative
conservative_submissions_base=conservative_submissions.select("author","title","num_comments","score","created_utc").cache()
conservative_comments_base=conservative_comments.select("author","body","controversiality","score","created_utc").cache()

#Democrats
democrats_submissions_base=democrats_submissions.select("author","title","num_comments","score","created_utc").cache()
democrats_comments_base=democrats_comments.select("author","body","controversiality","score","created_utc").cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## To determine the public’s opinion about how president Biden handles important issues
# MAGIC 
# MAGIC ### Create a subset of the dataset which includes the words “biden”, “POTUS” and so on. Then create further subsets based on each of the issues above such as covid19, gun control, inflation, ukraine,etc and i and assess the sentiment scores. This can be further extended to the democrat and republican subreddits separately as well.

# COMMAND ----------

#Getting subsets for President Biden
#Politics
politics_comments_biden=politics_comments_base.withColumn('Biden', when(((col('body').rlike("(?i)biden|(?i)Biden|(?i)POTUS|(?i)Joe Biden|(?i)potus"))), 1).otherwise(0))
politics_biden_comments=politics_comments_biden.filter(col('Biden') == 1)
politics_biden_comments = politics_biden_comments.select("body", f.lower("body"))

#Subset2
politics_biden_comments2=politics_comments_biden.filter(col('Biden') == 1)


# COMMAND ----------

politics_biden_comments2=politics_biden_comments2.cache()

# COMMAND ----------

politics_biden_comments2.show()

# COMMAND ----------

#Conservative
conservative_comments_biden=conservative_comments_base.withColumn('Biden', when(((col('body').rlike("(?i)biden|(?i)Biden|(?i)POTUS|(?i)Joe Biden|(?i)potus"))), 1).otherwise(0))
conservative_biden_comments=conservative_comments_biden.filter(col('Biden') == 1)
conservative_biden_comments = conservative_biden_comments.select("body", f.lower("body"))

# COMMAND ----------

# #Democrats
democrats_comments_biden=democrats_comments_base.withColumn('Biden', when(((col('body').rlike("(?i)biden|(?i)Biden|(?i)POTUS|(?i)Joe Biden|(?i)potus"))),1).otherwise(0))
democrats_biden_comments=democrats_comments_biden.filter(col('Biden') == 1)
democrats_biden_comments = democrats_biden_comments.select("body", f.lower("body"))

# COMMAND ----------

politics_biden_comments.show()

# COMMAND ----------

#Functions to create datasets for all the issues

#1. Gun Control
def gun_control(df):
    df = df.withColumn("gun_control", when(col('lower(body)')\
        .rlike("(?i)gun|(?i)rifle|(?i)firearm|(?i)assualt weapon|(?i)weapon|(?i)semiautomatic weapon|(?i)weapons"), lit(True)) \
        .otherwise(False))
    df=df.filter(col('gun_control') == 1)
    return df

#2. Birth Rights
def birth_rights(df):
    df = df.withColumn("birth_rights", when(col('lower(body)')\
        .rlike('(?i)abortion|(?i)birth control|(?i)pro-life|(?i)anti-abortion|(?i)reproductive'), True)\
        .otherwise(False))
    df=df.filter(col('birth_rights') == 1)
    return df

#3. Covid
def covid(df):
    df = df.withColumn("covid", when(col('lower(body)')\
        .rlike('(?i)covid|(?i)corona|(?i)virus|(?i)mask|(?i)masks|(?i)masking|(?i)covid-19'), True)
        .otherwise(False))
    df=df.filter(col('covid') == 1)
    return df

#4. Ukraine
def ukraine(df):
    df = df.withColumn("ukraine", when(col('lower(body)')\
        .rlike('(?i)ukraine|(?i)russia'), True)
        .otherwise(False))
    df=df.filter(col('ukraine') == 1)
    return df

#5. Taxation
def taxation(df):
    df = df.withColumn("taxation", when(col('lower(body)')\
        .rlike('(?i)tax|(?i)taxes|(?i)taxation|(?i)taxtherich'), True)
        .otherwise(False))
    df=df.filter(col('taxation') == 1)
    return df

# COMMAND ----------

politics_gun_biden=gun_control(politics_biden_comments)
politics_birth_biden=birth_rights(politics_biden_comments)
politics_covid_biden=covid(politics_biden_comments)
politics_ukraine_biden=ukraine(politics_biden_comments)
politics_taxation_biden=taxation(politics_biden_comments)

# COMMAND ----------

politics_gun_biden.show()

# COMMAND ----------

politics_birth_biden.show()

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

def issue_sentiment(df,issue):
    pipelineModel = nlpPipeline.fit(df)
    result_ = pipelineModel.transform(df)
    result_ = result_.select('body', explode('sentiment.result'))
    result_ = result_.filter((result_.col == "positive") | (result_.col == "negative"))
    final_df=result_.groupBy("col").agg(count("body").alias("Sentiment_Count"))
    final_df=final_df.withColumnRenamed("col","Sentiment")
    final_df=final_df.cache()
    final_df=final_df.withColumn("Issue", lit(issue))
    return final_df

# COMMAND ----------

politics_gun_biden_sentiment=issue_sentiment(politics_gun_biden,"Gun_Control")
politics_birth_biden_sentiment=issue_sentiment(politics_birth_biden,"Birth_Rights")
politics_covid_biden_sentiment=issue_sentiment(politics_covid_biden,"Covid")
politics_ukraine_biden_sentiment=issue_sentiment(politics_ukraine_biden,"Ukraine")
politics_taxation_biden_sentiment=issue_sentiment(politics_taxation_biden,"Taxation")

# COMMAND ----------

politics_gun_biden_sentiment.show()

# COMMAND ----------

politics_birth_biden_sentiment.show()

# COMMAND ----------

politics_covid_biden_sentiment.show()

# COMMAND ----------

politics_ukraine_biden_sentiment.show()

# COMMAND ----------

politics_taxation_biden_sentiment.show()

# COMMAND ----------

sentiment_summary=politics_gun_biden_sentiment.union(politics_birth_biden_sentiment)
sentiment_summary=sentiment_summary.union(politics_covid_biden_sentiment)
sentiment_summary=sentiment_summary.union(politics_ukraine_biden_sentiment)
sentiment_summary=sentiment_summary.union(politics_taxation_biden_sentiment)

# COMMAND ----------

sentiment_summary=sentiment_summary.cache()

# COMMAND ----------

sentiment_summary.show()

# COMMAND ----------

sentiment_pandas=sentiment_summary.toPandas()

# COMMAND ----------

sentiment_pandas

# COMMAND ----------

sentiment_pandas.to_csv('/Workspace/Repos/ar1710@georgetown.edu/fall-2022-reddit-big-data-project-project-group-30/data/csv/biden_issues_pd.csv',index=False)

# COMMAND ----------

sentiment_df=pd.read_csv("/Workspace/Repos/ar1710@georgetown.edu/fall-2022-reddit-big-data-project-project-group-30/data/csv/biden_issues_pd.csv")
df2=sentiment_df.pivot(index=['Issue'], columns='Sentiment', values='Sentiment_Count').reset_index()

# COMMAND ----------

df2

# COMMAND ----------

# MAGIC %md
# MAGIC ### Getting the sentiment for Democrats and Republicans

# COMMAND ----------

#Conservative
conservative_gun_biden=gun_control(conservative_biden_comments)
conservative_birth_biden=birth_rights(conservative_biden_comments)
conservative_covid_biden=covid(conservative_biden_comments)
conservative_ukraine_biden=ukraine(conservative_biden_comments)
conservative_taxation_biden=taxation(conservative_biden_comments)

# COMMAND ----------

#Democrats
democrats_gun_biden=gun_control(democrats_biden_comments)
democrats_birth_biden=birth_rights(democrats_biden_comments)
democrats_covid_biden=covid(democrats_biden_comments)
democrats_ukraine_biden=ukraine(democrats_biden_comments)
democrats_taxation_biden=taxation(democrats_biden_comments)

# COMMAND ----------

#Transforming the dataset for conservative
conservative_gun_biden_sentiment=issue_sentiment(conservative_gun_biden,"Gun_Control")
conservative_birth_biden_sentiment=issue_sentiment(conservative_birth_biden,"Birth_Rights")
conservative_covid_biden_sentiment=issue_sentiment(conservative_covid_biden,"Covid")
conservative_ukraine_biden_sentiment=issue_sentiment(conservative_ukraine_biden,"Ukraine")
conservative_taxation_biden_sentiment=issue_sentiment(conservative_taxation_biden,"Taxation")

# COMMAND ----------

conservative_taxation_biden_sentiment.show()

# COMMAND ----------

#Transforming the dataset for democrats
democrats_gun_biden_sentiment=issue_sentiment(democrats_gun_biden,"Gun_Control")
democrats_birth_biden_sentiment=issue_sentiment(democrats_birth_biden,"Birth_Rights")
democrats_covid_biden_sentiment=issue_sentiment(democrats_covid_biden,"Covid")
democrats_ukraine_biden_sentiment=issue_sentiment(democrats_ukraine_biden,"Ukraine")
democrats_taxation_biden_sentiment=issue_sentiment(democrats_taxation_biden,"Taxation")

# COMMAND ----------

#Sentiment Summary conservative
sentiment_summary2=conservative_gun_biden_sentiment.union(conservative_birth_biden_sentiment)
sentiment_summary2=sentiment_summary2.union(conservative_covid_biden_sentiment)
sentiment_summary2=sentiment_summary2.union(conservative_ukraine_biden_sentiment)
sentiment_summary2=sentiment_summary2.union(conservative_taxation_biden_sentiment)

# COMMAND ----------

sentiment_summary2=sentiment_summary2.cache()
sentiment_pandas2=sentiment_summary2.toPandas()

# COMMAND ----------

sentiment_pandas2

# COMMAND ----------

sentiment_pandas2.to_csv('/Workspace/Repos/ar1710@georgetown.edu/fall-2022-reddit-big-data-project-project-group-30/data/csv/biden_issues_pd2.csv',index=False)

# COMMAND ----------

#Sentiment Summary democrats
sentiment_summary3=democrats_gun_biden_sentiment.union(democrats_birth_biden_sentiment)
sentiment_summary3=sentiment_summary3.union(democrats_covid_biden_sentiment)
sentiment_summary3=sentiment_summary3.union(democrats_ukraine_biden_sentiment)
sentiment_summary3=sentiment_summary3.union(democrats_taxation_biden_sentiment)

# COMMAND ----------

sentiment_summary3=sentiment_summary3.cache()
sentiment_pandas3=sentiment_summary3.toPandas()

# COMMAND ----------

sentiment_pandas3

# COMMAND ----------

sentiment_pandas3.to_csv('/Workspace/Repos/ar1710@georgetown.edu/fall-2022-reddit-big-data-project-project-group-30/data/csv/biden_issues_pd3.csv',index=False)

# COMMAND ----------

sentiment_df2=pd.read_csv("/Workspace/Repos/ar1710@georgetown.edu/fall-2022-reddit-big-data-project-project-group-30/data/csv/biden_issues_pd2.csv")
df3=sentiment_df2.pivot(index=['Issue'], columns='Sentiment', values='Sentiment_Count').reset_index()

# COMMAND ----------

sentiment_df3=pd.read_csv("/Workspace/Repos/ar1710@georgetown.edu/fall-2022-reddit-big-data-project-project-group-30/data/csv/biden_issues_pd3.csv")
df4=sentiment_df3.pivot(index=['Issue'], columns='Sentiment', values='Sentiment_Count').reset_index()

# COMMAND ----------

fig = make_subplots(rows=3, cols=1,subplot_titles=("r/Politics","r/Conservatives","r/Democrats"))

colors = {'negative': '#941E05',
          'positive': '#05946D'}

fig.add_trace(go.Bar(x=-df2.negative.values,
                         y=df2['Issue'],
                         orientation='h',                    
                         name="Negative",
                         marker={'color': colors['negative']},
        
                         customdata=df2.negative,
                         hovertemplate = "%{y}: %{customdata}"),
                         row=1, col=1)

fig.add_trace(go.Bar(x=df2.positive.values,
                         y=df2['Issue'],
                         orientation='h',                    
                         name="Positive",
                         marker={'color': colors['positive']},
        
                         customdata=df2.positive,
                         hovertemplate = "%{y}: %{customdata}"),
                         row=1, col=1)


fig.add_trace(go.Bar(x=-df3.negative.values,
                         y=df3['Issue'],
                         orientation='h',                    
                         name="Negative",
                         marker={'color': colors['negative']},
        
                         customdata=df3.negative,
                         hovertemplate = "%{y}: %{customdata}"),
                         row=2, col=1)

fig.add_trace(go.Bar(x=df3.positive.values,
                         y=df3['Issue'],
                         orientation='h',                    
                         name="Positive",
                         marker={'color': colors['positive']},
        
                         customdata=df3.positive,
                         hovertemplate = "%{y}: %{customdata}"),
                         row=2, col=1)

fig.add_trace(go.Bar(x=-df4.negative.values,
                         y=df4['Issue'],
                         orientation='h',                    
                         name="Negative",
                         marker={'color': colors['negative']},
        
                         customdata=df4.negative,
                         hovertemplate = "%{y}: %{customdata}"),
                         row=3, col=1)

fig.add_trace(go.Bar(x=df4.positive.values,
                         y=df4['Issue'],
                         orientation='h',                    
                         name="Positive",
                         marker={'color': colors['positive']},
        
                         customdata=df4.positive,
                         hovertemplate = "%{y}: %{customdata}"),
                         row=3, col=1)

fig=fig.update_layout(barmode='overlay', 
                  height=800, 
                  width=800, 
                  yaxis_autorange='reversed',
                  bargap=0.05,
                  legend_orientation ='h',
                  legend_x=-0.05, legend_y=1.1,
                    autosize=False,
        showlegend=False,
    
        title=("Sentiment of issues handled by President Biden"),
        title_font_color='darkblue',
        title_font_size=22,
        paper_bgcolor='White',
        plot_bgcolor='White')
fig

# COMMAND ----------

fig.write_html("../data/plots/sentiment_combined.html")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exploring the wordclouds for positive and negative sentiments of democrats and conservative
# MAGIC 
# MAGIC Cleaning the data by tokenising, normalising, lemmatising,stopwords removal

# COMMAND ----------

conservative_biden_comments.show()

# COMMAND ----------

# import nltk
# nltk.download('stopwords')

# COMMAND ----------

#pip install nltk

# COMMAND ----------

# from nltk.corpus import stopwords
# eng_stopwords = stopwords.words('english')
# eng_stopwords.append('xxxx')

# COMMAND ----------

documentAssembler = DocumentAssembler() \
     .setInputCol('body') \
     .setOutputCol('document')\
     .setCleanupMode("shrink")

tokenizer = Tokenizer() \
     .setInputCols(['document']) \
     .setOutputCol('token')

normalizer = Normalizer() \
     .setInputCols(['token']) \
     .setOutputCol('normalized') \
     .setLowercase(True)

lemmatizer = LemmatizerModel.pretrained() \
     .setInputCols(['normalized']) \
     .setOutputCol('lemma')

# stopwords_cleaner = StopWordsCleaner() \
#      .setInputCols(['lemma']) \
#      .setOutputCol('clean_lemma') \
#      .setCaseSensitive(False) \
#      .setStopWords(eng_stopwords)

stopwords_cleaner = StopWordsCleaner.pretrained("stopwords_en", "en") \
     .setInputCols(["lemma"]) \
     .setOutputCol("clean_lemma")

finisher = Finisher() \
     .setInputCols(['clean_lemma']) \
     .setCleanAnnotations(False)



# COMMAND ----------

cleaning_pipeline = Pipeline() \
     .setStages([
           documentAssembler,    
           tokenizer,
           normalizer,
           lemmatizer,
           stopwords_cleaner,
           finisher
     ])

# COMMAND ----------

data = politics_biden_comments.select('body')
cleaned_body= cleaning_pipeline.fit(data).transform(data)

# COMMAND ----------

# pipelineModel = nlpPipeline.fit(df)
# result_ = pipelineModel.transform(df)
# result_ = result_.select('body', explode('sentiment.result'))
# result_ = result_.filter((result_.col == "positive"))
# result_ = result_.filter((result_.col == "negative"))

# COMMAND ----------

cleaned_body.show()

# COMMAND ----------

def get_words(u):
    u_words = (u.select(f.explode(u.finished_clean_lemma).alias('word')))
    u_words = u_words.where(u_words.word != '')
    u_words=u_words.groupBy('word').count()
    u_words = u_words.orderBy("count", ascending=0)
    u_df=u_words.toPandas()
    word_0=u_df[150:]
    k=word_0.set_index('word').squeeze()
    return k

# COMMAND ----------

cleanded_body_word=get_words(cleaned_body)

# COMMAND ----------

import matplotlib as mpl

# COMMAND ----------

cmap = mpl.cm.Blues(np.linspace(0,1,20)) 
wc = WordCloud(min_word_length =10,background_color='white',mode="RGBA",width=1600, height=800,colormap='Greys').generate_from_frequencies(cleanded_body_word)
plt.figure(figsize=(16,8))
plt.imshow(wc)
plt.axis('off')
plt.savefig("../data/plots/wordcloud_politics.png")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Negative Conservative Comments

# COMMAND ----------

#Conservative Negative
pipelineModel = nlpPipeline.fit(conservative_biden_comments)
result_cons = pipelineModel.transform(conservative_biden_comments)
result_cons = result_cons.select('body', explode('sentiment.result'))
result_neg_cons = result_cons.filter((result_cons.col == "negative"))

# COMMAND ----------

result_neg_cons.show()

# COMMAND ----------

data_cons = result_neg_cons.select('body')
cleaned_body_cons= cleaning_pipeline.fit(data_cons).transform(data_cons)
cleanded_body_word_cons=get_words(cleaned_body_cons)

# COMMAND ----------

wc2 = WordCloud(min_word_length =10,background_color='white',mode="RGBA",width=1600, height=800,colormap='Reds').generate_from_frequencies(cleanded_body_word_cons)
plt.figure(figsize=(16,8))
plt.imshow(wc2)
plt.axis('off')
plt.savefig("../data/plots/wordcloud_cons.png")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Negative Democrat Comments

# COMMAND ----------

#Democrat Negative
pipelineModel = nlpPipeline.fit(democrats_biden_comments)
result_dem = pipelineModel.transform(democrats_biden_comments)
result_dem = result_dem.select('body', explode('sentiment.result'))
result_neg_dem = result_dem.filter((result_dem.col == "negative"))

# COMMAND ----------

data_dem = result_neg_dem.select('body')
cleaned_body_dem= cleaning_pipeline.fit(data_dem).transform(data_dem)
cleanded_body_word_dem=get_words(cleaned_body_dem)

# COMMAND ----------

wc3 = WordCloud(min_word_length =10,background_color='white',mode="RGBA",width=1600, height=800,colormap='Blues').generate_from_frequencies(cleanded_body_word_dem)
plt.figure(figsize=(16,8))
plt.imshow(wc3)
plt.axis('off')
plt.savefig("../data/plots/wordcloud_dem.png")
plt.show()

# COMMAND ----------


