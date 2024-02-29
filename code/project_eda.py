# Databricks notebook source
# MAGIC %md
# MAGIC # Project EDA Assignment

# COMMAND ----------

# MAGIC %md
# MAGIC ## Questions to be answered
# MAGIC 
# MAGIC ### Question 1
# MAGIC <h5>Business goal: Determine the most popular leaders from both the republican and the democratic party to see who is in line to get the 2024 Presidential Election bid.
# MAGIC   
# MAGIC Technical proposal: Use NLP to find out the posts that mention the names of the popular leaders. Conduct counts and conduct sentimental analysis to assign negative or positive values to the post. Then present a list of the most popular leaders amongst positive values and the most popular leaders amongst negative values. 
# MAGIC </h5>
# MAGIC 
# MAGIC ### Question 2
# MAGIC <h5>Business goal: In order to make some changes to the reddit algorithm to increase screen time of people using reddit, predict the controversiality of the post so that the most controversial posts can be placed above lesser controversial posts in r/politics.
# MAGIC   
# MAGIC Technical proposal: Use NLP to create a few dummy variables that pick up key words from the text, create more dummy variables like time of the post etc. Train the machine learning algorithm on 80% of the data and then test the accuracy of algorithm on the remaining 20% of the data. Then optimize the algorithm to get better results. 
# MAGIC </h5>
# MAGIC 
# MAGIC ### Question 3
# MAGIC 
# MAGIC <h5>Business goal: To find out what topics should the candidate's election promise cover by looking at the popular topics
# MAGIC   
# MAGIC Technical proposal: Create different dummy variables for various topics, involving “economy”, “medicare”, “gun-control” , “terrorism”, “abortion” and other common topics. Take a subset of the data from r/politics to only look at recent posts. Plot the number of posts that refer to these topics.
# MAGIC </h5>
# MAGIC 
# MAGIC ### Question 4
# MAGIC 
# MAGIC <h5>Business goal:  To find out the the differences in reactions and sentiments of democrats and republicans over some of the important political events over the year
# MAGIC   
# MAGIC Technical proposal: To display the word clouds of the submissions and comments from the democrat and republican subreddits. This will be achieved using subsets generated based on the timeline of some of the political events over the year. Comparisons will also be made by analyzing the submission and controversiality scores.
# MAGIC </h5>
# MAGIC 
# MAGIC 
# MAGIC ### Question 5
# MAGIC 
# MAGIC <h5>Business goal: To determine the public’s opinion about how president Biden handles important issues.
# MAGIC 
# MAGIC Technical proposal: Create a subset of the dataset which includes the words “biden”, “POTUS” and so on. Then create further subsets based on each of the issues above such as covid19, gun control, inflation, ukraine,etc and i and assess the sentiment scores. This can be further extended to the democrat and republican subreddits separately as well. 
# MAGIC </h5>
# MAGIC 
# MAGIC 
# MAGIC ### Question 6
# MAGIC 
# MAGIC <h5>Business goal: Determine how socially and culturally different people with opposing political beliefs are. 
# MAGIC 
# MAGIC   Technical proposal: What are the top secondary subreddits for users that post in liberal or conservative subreddits? What can these top secondary subreddits tell us about liberals and conservatives? Get separate lists that include the usernames of accounts that follow the subreddits. Retrieve list of all subreddits that these accounts follow. Find out which other subreddits, besides r/Conservative and r/democrats are most likely to be followed by accounts in each group.
# MAGIC   </h5>
# MAGIC   
# MAGIC   
# MAGIC 
# MAGIC ### Question 7
# MAGIC 
# MAGIC <h5>Business goal: Find out which political group contains the angriest and most emotional followers.
# MAGIC   
# MAGIC Technical proposal: Find the top submissions from both subreddits and collect the comments. Conduct sentiment analysis on both sets of comments to find out if there is any difference between the sentiment scores. Do subreddits for different political groups contain different sentiment scores? I.e., are posts/comments in contrasting political subreddits angrier than their counterparts? 
# MAGIC </h5>
# MAGIC 
# MAGIC ### Question 8
# MAGIC 
# MAGIC <h5>Business goal: Determine which political group is most likely to consider arguments from an opposing viewpoint.
# MAGIC   
# MAGIC Technical proposal: Who would be more likely to post/comment in a politically neutral subreddit: an account that posts mainly in conservative or liberal forums? Find out which group of authors (r/Conservatives and r/democrats) have the larger portion of their users posting/commenting in r/politics. Get unique users from both subreddits and use an inner join to get the number of posts/comments coming from democrats and republicans. Check if the authors from r/democrats are also subscribed to r/Conservative and vice-versa.
# MAGIC </h5>
# MAGIC 
# MAGIC ### Question 9
# MAGIC 
# MAGIC <h5>Business goal: Determine what it means for a person to be politically neutral in today's political climate.
# MAGIC   
# MAGIC Technical proposal: Determine what people are posting about if they are commenting or submitting in both conservative and liberal subreddits. Is there a common theme? Check if the author from r/politics is in conservative or liberal subreddits by joining on author.  Conduct sentiment analysis to see if the posts of that author are positive or negative.
# MAGIC </h5>
# MAGIC 
# MAGIC 
# MAGIC ### Question 10
# MAGIC 
# MAGIC <h5>Business goal: Is the discussion around politics online balanced? Are people that are discussing politics online more liberal or conservative?
# MAGIC   
# MAGIC Technical proposal: Check if the author from r/politics is in conservative or liberal subreddits by joining on author. Conduct sentiment analysis to see if the posts of that author are positive or negative to determine what they agree with.
# MAGIC   </h5>
# MAGIC 
# MAGIC 
# MAGIC ### Question 11
# MAGIC 
# MAGIC <h5>
# MAGIC   Business goal: Is the president's approval rating accurate? Is the way people discuss the president online representative of his approval rating?
# MAGIC   
# MAGIC Technical proposal: Join the r/democrat reddit dataset with an approval rating dataset on the date. Create a time series that displays the number of positive posts a day and the approval rating. See if the approval rating is correlated to the number of positive comments.
# MAGIC </h5>
# MAGIC 
# MAGIC 
# MAGIC ### Question 12
# MAGIC 
# MAGIC <h5>
# MAGIC   Business goal: Are there certain words or phrases that align with political bias? Write a speech based on topics that are most important to the party using language that is popular among the party.
# MAGIC   
# MAGIC Technical proposal: Train a model on the r/democrat and r/republican subreddits using SparkML. Perform sentiment analysis to only keep the positive posts from each subreddit. By only keeping positive posts, the subreddits will be void of rival party comments.
# MAGIC </h5>

# COMMAND ----------

# MAGIC %md
# MAGIC ## EDA

# COMMAND ----------

# import libraries
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql.types import StringType,BooleanType,DateType
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pyspark.sql.functions as f
from pyspark.sql.functions import col, when
import plotly.express as px

# COMMAND ----------

# getting data
dbutils.fs.ls("abfss://anly502@marckvaismanblob.dfs.core.windows.net/reddit/parquet")
comments = spark.read.parquet("abfss://anly502@marckvaismanblob.dfs.core.windows.net/reddit/parquet/comments")
submissions = spark.read.parquet("abfss://anly502@marckvaismanblob.dfs.core.windows.net/reddit/parquet/submissions")

# COMMAND ----------

# filtering data using only r/politics
politics_submissions = submissions.filter(submissions.subreddit == "politics")
politics_comments = comments.filter(comments.subreddit == "politics")

# COMMAND ----------

# printing the schema
politics_submissions.printSchema()

# COMMAND ----------

# printing the schema
politics_comments.printSchema()

# COMMAND ----------

print("Number of submissions (rows) in r/politics = ",politics_submissions.count())

# COMMAND ----------

print("Number of comments (rows) in r/politics = ",politics_comments.count())

# COMMAND ----------

politics_comments.groupBy('controversiality').count().show()  # controversiality is flagged when the number of upvotes and downvotes are similar

# COMMAND ----------

# cleaning and adding new columns

def remove_deleted_comments(df):
    # deleting rows that don't have authors or body anymore
    df = df.filter(col("author") != "[deleted]")
    df = df.filter(col("body") != "[deleted]")
    df = df.filter(col("body") != "[removed]")
    return df

def remove_deleted_submissions(df):
    # deleting rows that don't have authors or body anymore
    df = df.filter(col("author") != "[deleted]")
    df = df.filter(col("title") != "[deleted]")
    df = df.filter(col("title") != "[removed]")
    return df
    

def convert_utc(df):
    df = df.withColumn("created_month", from_unixtime(col("created_utc"),"M"))  # adding a new column (month of creation)
    df = df.withColumn("created_utc", from_unixtime(col("created_utc"),"M/d/yyyy"))  # converting into m/d/yyyy format
    return df
  
def convert_utc_date(df):
    df = df.withColumn("created_utc", to_date(col("created_utc"),"M/d/yyyy"))  # converting into m/d/yyyy format
    return df

# COMMAND ----------

converted_political_comments = remove_deleted_comments(convert_utc(politics_comments))

# COMMAND ----------

print("Number of comments left after deleting deleted users and comments =",converted_political_comments.count())

# COMMAND ----------

#Total number of comments discussing president Biden
# converted_political_submissions = remove_deleted_submissions(convert_utc(politics_submissions))
# politics_submissions_base=converted_political_submissions.select("author","title","num_comments","score","created_utc")
# politics_submissions_base = politics_submissions_base.withColumn('Biden', when(((col('title').rlike("(?i)biden|(?i)Biden|(?i)POTUS|(?i)Joe Biden|(?i)potus"))), 1).otherwise(0))
# biden=politics_submissions_base.filter(col('Biden') == 1)
# biden.cache()
# biden=convert_utc_date(biden)
# biden_sum= biden.groupBy('created_utc').agg(f.sum('num_comments').alias('sum_comments'))
# biden_sum_sort=biden_sum.sort(biden_sum.created_utc.asc())
# biden_sum_sort_df.to_csv('/dbfs/FileStore/data/biden_submission_tot_comments.csv')


# COMMAND ----------

#Visualisation 3
biden_sum_sort_df = pd.read_csv('/dbfs/FileStore/data/biden_submission_tot_comments.csv')
#Using plotly
fig = px.line(biden_sum_sort_df, x="created_utc", y="sum_comments",color_discrete_sequence=["#0938FA"])
fig.update_layout(
    title="Daily total number of comments about president Biden in r/Politics",
    xaxis_title="Date",
    yaxis_title="Total Comments",
    legend_title=None,
    xaxis_showgrid=False, yaxis_showgrid=False,
    width=700)

fig.show()
fig.write_html("../data/plots/biden_sum_comments.html")

# COMMAND ----------

#Contrversiality of Bidens comments and scores for these comments
# politics_comments_base=converted_political_comments.select("author","body","controversiality","score","created_utc")
# politics_comments_base = politics_comments_base.withColumn('Biden', when(((col('body').rlike("(?i)biden|(?i)Biden|(?i)POTUS|(?i)Joe Biden|(?i)potus"))), 1).otherwise(0))
# biden_comments=politics_comments_base.filter(col('Biden') == 1)
# biden_comments.cache()
# biden_comments=convert_utc_date(biden_comments)

#Controversiality table
# biden_controversiality=biden_comments.groupBy(col('controversiality')).count()
# biden_controversiality=biden_controversiality.toPandas()
# biden_controversiality.to_csv('/dbfs/FileStore/data/biden_controversiality_general.csv')

# COMMAND ----------

#Visualisation 4
biden_controversiality=pd.read_csv('/dbfs/FileStore/data/biden_controversiality_general.csv')
#Using plotly
fig = px.bar(biden_controversiality, x="controversiality", y="count",color="controversiality",color_continuous_scale=px.colors.sequential.Cividis)
fig.update_layout(
    title="Controversiality of comments about president Biden in r/politics",
    xaxis_title="Not Controversial/Controversial",
    yaxis_title="Count",
    legend_title=None,
    width=700,
    xaxis_showgrid=False, yaxis_showgrid=False,
    coloraxis_showscale=False
)
fig.update_xaxes(nticks=3)
fig.show()
fig.write_html("../data/plots/biden_controversial_comments.html")

# COMMAND ----------

#Visualisation 5
#Scores for President biden
#biden_comments=biden_comments.toPandas()


# COMMAND ----------


# biden_comments=pd.read_csv('/dbfs/FileStore/data/biden_comments.csv')
# fig = px.scatter(biden_comments, x="created_utc", y="score",color_discrete_sequence=["#042E5E"])
# fig.update_layout(
#     title="Comment scores about president Biden in r/politics",
#     xaxis_title="Date",
#     yaxis_title="Score",
#     legend_title=None,
#     width=700,
#     xaxis_showgrid=False, yaxis_showgrid=False
# )
# fig.show()


# COMMAND ----------

# pol_com_created=politics_comments.select("created_utc", "body")
# pol_com_created=pol_com_created.withColumn("created_utc", from_unixtime(lit(col("created_utc").cast("long")),"M/yyyy"))


# date_created_counts = pol_com_created.groupby('created_utc').count().collect()
# date_created_counts_df = spark.createDataFrame(date_created_counts)
# date_created_counts_df = date_created_counts_df.toPandas()
# date_created_counts_df['created_utc']= pd.to_datetime(date_created_counts_df['created_utc']).dt.to_period('M')
# date_created_counts_df = date_created_counts_df.sort_values(by='created_utc',ascending=True)


# COMMAND ----------


#date_created_counts_df.to_csv('/dbfs/FileStore/data/politics_comments_groupby_yyyy-mm.csv')
date_created_counts_df = pd.read_csv('/dbfs/FileStore/data/politics_comments_groupby_yyyy-mm.csv')
fig = px.bar(date_created_counts_df, x="created_utc", y="count",color_discrete_sequence=["#02597F"])
fig.update_layout(
    title="Counts of comments by year and month in r/politics",
    xaxis_title="Timeline",
    yaxis_title="Comment Count",
    legend_title=None,
    xaxis_showgrid=False, yaxis_showgrid=False,
    width=700
)
fig.show()
fig.write_html("../data/plots/count_comments.html")

# COMMAND ----------

# politics_comments_topic_model = politics_comments.select("body", f.lower("body"))


# politics_comments_topic_model = politics_comments_topic_model.withColumn("gun_control", when(col('lower(body)')\
#         .rlike("(?i)gun|(?i)rifle|(?i)firearm|(?i)assualt weapon|(?i)weapon|(?i)semiautomatic weapon|(?i)weapons"), lit(True)) \
#         .otherwise(False))\
#         .withColumn("birth_rights", when(col('lower(body)')\
#         .rlike('(?i)abortion|(?i)birth control|(?i)pro-life|(?i)anti-abortion|(?i)reproductive'), True)\
#         .otherwise(False))\
#         .withColumn("terrorism", when(col('lower(body)')\
#         .rlike('(?i)terrorism|(?i)terrorist|(?i)isis|(?i)al-qaeda|(?i)air-strike|(?i)airstrike'), True)\
#         .otherwise(False))\
#         .withColumn("health_care", when(col('lower(body)')\
#         .rlike('(?i)health|(?i)medicare|(?i)medicaid|(?i)healthcare|(?i)health insurance|(?i)obamacare'), True)
#         .otherwise(False))\
#         .withColumn("covid", when(col('lower(body)')\
#         .rlike('(?i)covid|(?i)corona|(?i)virus|(?i)mask|(?i)masks|(?i)masking|(?i)covid-19'), True)
#         .otherwise(False))\
#         .withColumn("taxation", when(col('lower(body)')\
#         .rlike('(?i)tax|(?i)taxes|(?i)taxation|(?i)taxtherich'), True)
#         .otherwise(False))

# gun_control = politics_comments_topic_model.groupby('gun_control').count().collect()

# birth_rights = politics_comments_topic_model.groupby('birth_rights').count().collect()

# terrorism = politics_comments_topic_model.groupby('terrorism').count().collect()

# health_care = politics_comments_topic_model.groupby('health_care').count().collect()

# covid = politics_comments_topic_model.groupby('covid').count().collect()

# taxation = politics_comments_topic_model.groupby('taxation').count().collect()

# big_list =[gun_control, birth_rights, terrorism, health_care, covid, taxation]
# big_list_str =['gun_control', 'birth_rights', 'terrorism', 'health_care', 'covid', 'taxation']
# big_dict = []
# for i,j in zip(big_list,big_list_str):
#     big_dict.append({'topic':j, 'count':i[0][1]})
    
# politics_comments_topic_df = pd.DataFrame(big_dict)

# politics_comments_topic_df.to_csv('/dbfs/FileStore/data/politics_topic_count.csv')

politics_comments_topic_df = pd.read_csv('/dbfs/FileStore/data/politics_topic_count.csv')

# COMMAND ----------

#Visualisation 2
# politics_comments_topic_df.plot.bar(x='topic', y='count', rot=30, legend=None)
# plt.title("Counts of topics in r/politics")
# plt.xlabel("Topics")
# plt.ylabel("Counts")

fig = px.bar(politics_comments_topic_df, x="topic", y="count",color_discrete_sequence=["#656362","#AA3279","#741917","#A0BE77","#3C3ACB","#205B0E"],color="topic")
fig.update_layout(
    title="Counts of topics in r/politics",
    xaxis_title="Topics",
    yaxis_title="Count",
    legend_title=None,
    width=700,
    xaxis_showgrid=False, yaxis_showgrid=False
)
fig.show()
fig.write_html("../data/plots/count_topics.html")

# COMMAND ----------

# MAGIC %md
# MAGIC d # r/politics

# COMMAND ----------

# retrieve all submissions from the r/politics subreddit
politics_submissions = submissions.filter(submissions.subreddit == "politics")

# subset dataset to contain only the "author" and "title" columns
politics_submissions=politics_submissions.select("author", "title")

# view the current table 
# politics_submissions.show()

# COMMAND ----------

# get count of total submissions in the r/politics subreddit
politics_submissions.count()

# COMMAND ----------

# create new DF that contains all unique authors and the number of submission posts they have contributed 
pol_author_submission_count = politics_submissions.select('author').groupBy('author').count()
# pol_author_submission_count.show()

# COMMAND ----------

# MAGIC %md
# MAGIC This cell collects all unique authors that are posting submissions in r/politics

# COMMAND ----------

politics_submissions = politics_submissions.cache()
unique_politic_authors = politics_submissions.select('author').distinct()
print('Number of unique authors posting submissions in r/politics:',politics_submissions.select('author').distinct().count())

# COMMAND ----------

# MAGIC %md
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC # r/Conservative

# COMMAND ----------

# retrieve all submissions from the r/Conservative subreddit
conservative_submissions = submissions.filter(submissions.subreddit == "Conservative")

# subset dataset to contain only the "author" and "title" columns
conservative_submissions=conservative_submissions.select("author", "title")

# view the current table
# conservative_submissions.show()

# COMMAND ----------

# create new DF that contains all unique authors and the number of submission posts they have contributed 
cons_author_submission_count = conservative_submissions.select('author').groupBy('author').count()
# cons_author_submission_count.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Get unique authors that are posting submissions in r/Conservative

# COMMAND ----------

conservative_submissions = conservative_submissions.cache()
unique_conservative_authors = conservative_submissions.select('author').distinct()
print('Number of unique authors posting submissions in r/Conservative:',conservative_submissions.select('author').distinct().count())

# COMMAND ----------

# MAGIC %md
# MAGIC Number of authors that post in both **r/politics** and **r/Conservative**:

# COMMAND ----------

unique_politic_authors.join(unique_conservative_authors, unique_politic_authors.author == unique_conservative_authors.author, "inner").count()

# COMMAND ----------

# MAGIC %md
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC # r/democrats

# COMMAND ----------

# retrieve all submissions from the r/democrats subreddit
democrats_submissions = submissions.filter(submissions.subreddit == "democrats")

# subset dataset to contain only the "author" and "title" columns
democrats_submissions=democrats_submissions.select("author", "title").cache()

# view the current table
# democrats_submissions.show()

# COMMAND ----------

# create new DF that contains all unique authors and the number of submission posts they have contributed 
dem_author_submission_count = democrats_submissions.select('author').groupBy('author').count()
# dem_author_submission_count.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Get unique authors that are posting submissions in **r/democrats**.

# COMMAND ----------

democrats_submissions = democrats_submissions.cache()
unique_democrats_authors = democrats_submissions.select('author').distinct()
print('Number of unique authors posting submissions in r/democrats:',democrats_submissions.select('author').distinct().count())

# COMMAND ----------

# MAGIC %md
# MAGIC Number of authors that post in both **r/politics** and **r/democrats**:

# COMMAND ----------

unique_politic_authors.join(unique_democrats_authors, unique_politic_authors.author == unique_democrats_authors.author, "inner").count()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC 
# MAGIC # How many users are viewing both sides of the argument?
# MAGIC 
# MAGIC In order to answer this question we will want to see which authors are posting submissions in politically opposing subreddits. Here, we perform an inner join on 'author' to see which users are posting in both **r/Conservatives** and **r/democrats**.
# MAGIC 
# MAGIC Number of authors that post in both **r/Conservatives** and **r/democrats**:

# COMMAND ----------

unique_conservative_authors.join(unique_democrats_authors, unique_conservative_authors.author == unique_democrats_authors.author, "inner").count()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC 
# MAGIC # Directly comparing r/politics to r/Conservative and r/democrats

# COMMAND ----------

# change the name of the columns to be unique to which subreddit they are found in
pol_author_submission_count = pol_author_submission_count.selectExpr("count as submissions_politics", "author as p_author")
cons_author_submission_count = cons_author_submission_count.selectExpr("count as submissions_Conservative", "author as c_author")
dem_author_submission_count = dem_author_submission_count.selectExpr("count as submissions_democrats", "author as d_author")

# COMMAND ----------

# perform an outer merge on the authors in r/politics and r/Conservative
merged = pol_author_submission_count.join(cons_author_submission_count, pol_author_submission_count.p_author == cons_author_submission_count.c_author, "fullouter")


# COMMAND ----------

# Fill 'author' == null with the author 
merged = merged.withColumn("c_author", when(merged["c_author"].isNull(), merged["p_author"]).otherwise(merged["c_author"]))


# COMMAND ----------

# merge the previously joined table (politics + conservative) with the democrats table
merged = merged.join(dem_author_submission_count, merged.c_author == dem_author_submission_count.d_author, "fullouter")
# merged.show()

# COMMAND ----------

# Fill 'author' == null with the author
merged = merged.withColumn("c_author", when(merged["c_author"].isNull(), merged["d_author"]).otherwise(merged["c_author"]))
# merged.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Here, we fill all author submission counts that are `null` with `0` and then select only the necessary columns.

# COMMAND ----------

# convert `null` to 0
merged = merged.withColumn("submissions_politics", when(merged["submissions_politics"].isNull(), 0).otherwise(merged["submissions_politics"]))
merged = merged.withColumn("submissions_Conservative", when(merged["submissions_Conservative"].isNull(), 0).otherwise(merged["submissions_Conservative"]))
merged = merged.withColumn("submissions_democrats", when(merged["submissions_democrats"].isNull(), 0).otherwise(merged["submissions_democrats"]))

# select necessary columns
merged=merged.select("c_author", "submissions_politics", "submissions_Conservative", "submissions_democrats")

# COMMAND ----------

# MAGIC %md
# MAGIC ## __Create a new column__
# MAGIC 
# MAGIC In this cell we create a new boolean column called `centrist` that tells us whether an author has posted submissions in both the **r/Conservative** and **r/democrats** subreddits

# COMMAND ----------

merged = merged.withColumn("centrist", when((merged["submissions_Conservative"] != 0) & (merged["submissions_democrats"] != 0) , True).otherwise(False))
# merged.show(100)

# COMMAND ----------

# MAGIC %md
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC # Create summary tables to help explain the data

# COMMAND ----------

# MAGIC %md
# MAGIC Table 1: Number of submissions in each of the three subreddits

# COMMAND ----------

merged.agg(sum(merged.submissions_politics), sum(merged.submissions_Conservative), sum(merged.submissions_democrats)).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Table 2: Average number of submissions for all authors in all three subreddits

# COMMAND ----------

merged.agg(avg(merged.submissions_politics), avg(merged.submissions_Conservative), avg(merged.submissions_democrats)).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Table 3: Maximum number of submissions from a single author
# MAGIC 
# MAGIC This is misleading since these numbers are likely from `[deleted]` authors.

# COMMAND ----------

merged.agg(max(merged.submissions_politics), max(merged.submissions_Conservative), max(merged.submissions_democrats)).show()

# COMMAND ----------

# MAGIC %md
# MAGIC The following cells show the average number of submissions from each unique author that posts in one of the three subreddits.

# COMMAND ----------

dem_author_submission_count.agg({'submissions_democrats':'avg'}).show()

# COMMAND ----------

cons_author_submission_count.agg({'submissions_Conservative':'avg'}).show()

# COMMAND ----------

pol_author_submission_count.agg({'submissions_politics':'avg'}).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC # Visualizing the Data

# COMMAND ----------

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC In the cell below we create pandas dataframes for the unique authors and their count of submissions.

# COMMAND ----------

# add submission count data to dataframe
demDF = dem_author_submission_count.toPandas()
conDF = cons_author_submission_count.toPandas()
polDF = pol_author_submission_count.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC The datasets contain many outliers, so a logarithmic transformation is performed and added as a new column in each of the dataframes.

# COMMAND ----------

demDF['log_submissions'] = np.log(demDF['submissions_democrats'])
conDF['log_submissions'] = np.log(conDF['submissions_Conservative'])
polDF['log_submissions'] = np.log(polDF['submissions_politics'])

# COMMAND ----------

# MAGIC %md
# MAGIC In the cell below, boxplots of the log(number of submissions per unique author) are created for the three subreddits.

# COMMAND ----------

fig = go.Figure()
fig.add_trace(go.Box(x=demDF['log_submissions'], name='r/democrat'))
fig.add_trace(go.Box(x=conDF['log_submissions'], name='r/Conservative'))
fig.add_trace(go.Box(x=polDF['log_submissions'], name='r/politics'))
# fig.add_trace(go.Histogram(x=x1))

# The three histograms are drawn on top of another
fig.update_layout(barmode='overlay',autosize=False,width=1000,height=600,title_text='Boxplots of log(number of submissions per user)')
fig.show()
fig.write_html("../data/plots/log_submissions.html")

# COMMAND ----------

# MAGIC %md
# MAGIC In this cell we collect the averages of the log(number of submissions per user) and convert it back to the number of submissions

# COMMAND ----------

# collect the averages of the log(number of submissions per user)
submission_log_avg = [demDF['log_submissions'].mean(), conDF['log_submissions'].mean(), polDF['log_submissions'].mean()]

# convert the average log(number of submissions per user) back to actual number of submissions
submission_avg = np.exp(submission_log_avg)
submission_avg

# COMMAND ----------

subreddit=['r/democrat', 'r/Conservative', 'r/politics']

fig = go.Figure([go.Bar(x=subreddit, y=submission_avg, marker_color=['#636EFA','#EF553B','#00CC96'])])
fig.update_layout(barmode='overlay',autosize=False,width=1000,height=600,title_text='Average Number of Submissions per User <br><sup>Averages obtained after calculating the average of log(number of submissions per user) and converting back to number of submission.</sup>')
fig.show()
fig.write_html("../data/plots/avg_submissions.html")

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

dem_author_count = dem_author_submission_count.select('d_author').count()
con_author_count = cons_author_submission_count.select('c_author').count()
pol_author_count = pol_author_submission_count.select('p_author').count()

unique_author_count = [dem_author_count,con_author_count,pol_author_count]

# COMMAND ----------

subreddit=['r/democrat', 'r/Conservative', 'r/politics']

fig = go.Figure([go.Bar(x=subreddit, y=unique_author_count, marker_color=['#636EFA','#EF553B','#00CC96'])])
fig.update_layout(barmode='overlay',autosize=False,width=1000,height=600,title_text='Author Count <br><sup>Number of unique authors posting submissions.</sup>')
fig.show()
fig.write_html("../data/plots/author_count_submissions.html")

# COMMAND ----------

merged_new = merged
merged_new.cache()
# merged_new.show(100)

# COMMAND ----------

# look at percent of users in r/democrat and r/Conservative that are 'centrist'
centrist_count = merged_new.filter(merged_new.centrist).count()

centrist_count

# COMMAND ----------

percent_centrist = [centrist_count/dem_author_count,centrist_count/con_author_count]

subreddit=['r/democrat', 'r/Conservative']

fig = go.Figure([go.Bar(x=subreddit, y=percent_centrist, marker_color=['#636EFA','#EF553B'])])
fig.update_layout(barmode='overlay',yaxis_tickformat=".2%",autosize=False,width=1000,height=600,title_text='Percent of Authors that are Centrist <br><sup>Percent of users that are posting submissions in an opposing viewpoints subreddit.</sup>')
fig.show()
fig.write_html("../data/plots/centrist_submissions.html")

# COMMAND ----------

#Jack's Code
from pyspark.sql.functions import *

# COMMAND ----------

#reload data
politics_submissions = submissions.filter(submissions.subreddit == "politics")
politics_submissions=politics_submissions.select("author", "title")
demo_submissions = submissions.filter(submissions.subreddit == "democrats")
demo_submissions=demo_submissions.select("author", "title")
conservative_submissions = submissions.filter(submissions.subreddit == "Conservative")
conservative_submissions=conservative_submissions.select("author", "title")

# COMMAND ----------

#remove deleted authors and posts
demo_submissions=remove_deleted_submissions(demo_submissions)
conservative_submissions=remove_deleted_submissions(conservative_submissions)

# COMMAND ----------

#join conservative submission with politics submissions to see who posts in both subreddits 
c_pol=conservative_submissions.join(politics_submissions,conservative_submissions.author ==  politics_submissions.author,"inner")
c_pol.columns

# COMMAND ----------

#rename columns to drop
Data_list = ['author', 'title', 'author2', 'title2']
c_pol = c_pol.toDF(*Data_list)
#drop duplicate columns
c_pol=c_pol.drop("author2","title2")

# COMMAND ----------

#authors who post on both r/Conservative and r/politics 
#summary table
c_pol=c_pol.select(countDistinct("author").alias("count"))

# COMMAND ----------

d_pol=demo_submissions.join(politics_submissions,demo_submissions.author ==  politics_submissions.author,"inner")
Data_list = ['author', 'title', 'author2', 'title2']
d_pol = d_pol.toDF(*Data_list)
d_pol=d_pol.drop("author2","title2")

# COMMAND ----------

#authors who post on both r/democrat and r/politics 
#summary table
d_pol=d_pol.select(countDistinct("author").alias("count"))

# COMMAND ----------

#add number of conservative authors and number of liberal authors to one df
result = c_pol.union(d_pol)

# COMMAND ----------

#convert df to pandas for vizualization
df = result.toPandas()

# COMMAND ----------

df

# COMMAND ----------

# ax = df.plot(kind="bar",title='r/politics Political Afflications',legend=False)
# ax.set_xlabel("Political Affiliation")
# ax.set_ylabel("Authors")
# ax.set_xticklabels(['Conserative','Liberal'])  
# # plt.savefig('data/plots/political_affiliation.png')
# plt.show()

#Using Plotly
fig = px.bar(df,y="count")
fig.update_layout(
    title="r/politics Political Afflications",
    xaxis_title="Political Affiliation",
    yaxis_title="Authors",
    legend_title=None,
    width=700,
    xaxis_showgrid=False, yaxis_showgrid=False,
    coloraxis_showscale=False,
    xaxis=dict(
        tickvals = [0,1],
        ticktext = ['Conserative','Liberal'])
)
fig.update_xaxes(nticks=3)
fig.show()
fig.write_html("../data/plots/political_affiliations.html")

# COMMAND ----------

import pandas as pd
#read external dataset in
#approval = pd.read_csv("data/csv/approval_topline.csv")
approval = pd.read_csv("../data/csv/approval_topline.csv")

#convert df to spark df
approval=spark.createDataFrame(approval)

# COMMAND ----------

#only keep a few cols
politics_comments=politics_comments.select("author", "body",'created_utc','id')

# COMMAND ----------

#convert unixtime to datetime
politics_comments=politics_comments.withColumn("created_utc", from_unixtime(col("created_utc"),"M/d/yyyy"))
politics_comments.printSchema()

# COMMAND ----------

from pyspark.sql.types import StringType,BooleanType,DateType
#convert to DateType
politics_comments=politics_comments.withColumn("created_utc",to_date(col("created_utc"),"M/d/yyyy"))

# COMMAND ----------

#check date range
# summary table
col_earlist_date = min('created_utc').alias('earliest')
col_latest_date = max('created_utc').alias('latest')
df_result = politics_comments.select(col_earlist_date, col_latest_date)
df_result.show()

# COMMAND ----------

#convert to Datetype
# creating new variable
approval=approval.withColumn("date",to_date(col("modeldate"),"M/d/yyyy"))

# COMMAND ----------

#check date range
# summary table
col_earlist_date = min('date').alias('earliest')
col_latest_date = max('date').alias('latest')
df_result = approval.select(col_earlist_date, col_latest_date)
df_result.show()

# COMMAND ----------

#join approval rating dataset with comments dataset on date
approval_df=approval.join(politics_comments,approval.date ==  politics_comments.created_utc,"inner")

# COMMAND ----------

#drop dups
approval_df=approval_df.drop_duplicates()

# COMMAND ----------

#only keep results from all polls
approval_df = approval_df.filter(approval_df.subgroup == "All polls")
approval_df=approval_df.select("author", "body",'date','id','approve_estimate')

# COMMAND ----------

approval_df = approval_df.cache()

# COMMAND ----------

#count number of comments each day
approval_df2=approval_df.groupBy("date","approve_estimate").agg(count("id").alias("num_coms"))

# COMMAND ----------

#summary table
approval_df2.show()

# COMMAND ----------

approval_df3=approval_df2.toPandas()


# COMMAND ----------

#normalize approval rating
approval_df3["norm_rating"] = approval_df3["approve_estimate"] /approval_df3["approve_estimate"].abs().max()
#normalize approval rating
approval_df3["norm_coms"] = approval_df3["num_coms"] /approval_df3["num_coms"].abs().max()

# COMMAND ----------

# from matplotlib.pyplot import figure
# #normalized rating and comments to have a uniform scale
# figure(figsize=(10, 6), dpi=80)
# ax = plt.gca()
# ax.set_title('Approval Rating and Comments Correlation')
# approval_df3.plot(x="date", y="norm_coms",ax=ax,label="Number of Comments")
# approval_df3.plot(x="date", y="norm_rating",ax=ax,label="Approval Rating")
# ax.set_xlabel("Date")
# # plt.savefig('data/plots/time_series_correlation.png')
# plt.show()



# COMMAND ----------

#reload data
politics_submissions = submissions.filter(submissions.subreddit == "politics")
politics_submissions=politics_submissions.select("author", "title")
#create new boolean variable 'Democrat' 
#true if author that posts in r/politics also posts in r/democrat
politics_submissions=politics_submissions.join(demo_submissions.withColumn('Democrat', lit(True)), on="author", how='left')
#rename columns to drop
Data_list = ['author', 'title', 'title2', 'Democrat']
politics_submissions = politics_submissions.toDF(*Data_list)
politics_submissions = politics_submissions.drop("title2")

# COMMAND ----------

#create new boolean variable 'Republican' 
#true if author that posts in r/politics also posts in r/Conservative
politics_submissions=politics_submissions.join(conservative_submissions.withColumn('Republican', lit(True)), on="author", how='left')
#rename columns to drop
Data_list = ['author', 'title', 'Democrat', 'title2','Republican']
politics_submissions = politics_submissions.toDF(*Data_list)
politics_submissions = politics_submissions.drop("title2")
#drop duplicate rows
politics_submissions = politics_submissions.dropDuplicates(["author", "title"])

# COMMAND ----------

#save data
#comment out because cannot save to same location twice
#politics_submissions.select("author", "title", "Democrat","Republican").write.format('com.databricks.spark.csv').save("/FileStore/data")
