"""
Twitter Sentiment Analysis

This text data has been used to evaluate the performance of topic models on short text data.
However, it can be used for other tasks such as clustering. Implement topic modeling and sentiment analysis
"""

__date__ = "2022-10-31"
__author__ = "NedeeshaWeerasuriya"
__version__ = "0.1"


# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# %% --------------------------------------------------------------------------
# Load Twitter Data
# -----------------------------------------------------------------------------
df = pd.read_csv('data/textoutput',header=0)
df = df.drop(['Tweet'],axis=1)
df

# %% --------------------------------------------------------------------------
# Group data
# -----------------------------------------------------------------------------
year_df = df.groupby('year').agg({'Preprocessed Tweet':'sum'})
year_df

# %%
channel_df= df.groupby('channel').agg({'Preprocessed Tweet':'sum'})
channel_df

# %% --------------------------------------------------------------------------
# Run VADER Sentiment Analysis for Channel grouping
# -----------------------------------------------------------------------------
sent = SentimentIntensityAnalyzer()
all_scores = {}
for i in range(0,channel_df.size):
    score = sent.polarity_scores(channel_df.iloc[i]['Preprocessed Tweet'])
    all_scores[channel_df.index[i]] = score

# %%
score_df = pd.DataFrame(all_scores).T
score_df['channel'] = score_df.index
score_df.to_csv('vader_scores_channels.csv',index=False)

# %% --------------------------------------------------------------------------
# Join with sentiment word count dataframe
# -----------------------------------------------------------------------------
word_count = pd.read_csv('normalised_sentiment_word_count.csv')
word_count.index = word_count['Channel']
join_df = word_count.join(score_df)
join_df = join_df.drop(['channel','Channel'],axis=1).reset_index()
join_df.to_csv('channel_analysis.csv',index=False)

# %% --------------------------------------------------------------------------
# # Run VADER Sentiment Analysis for Year grouping
# -----------------------------------------------------------------------------
sent = SentimentIntensityAnalyzer()
year_scores = {}
for i in range(0,year_df.size):
    score = sent.polarity_scores(year_df.iloc[i]['Preprocessed Tweet'])
    year_scores[year_df.index[i]] = score

year_scores
# %%
year_score_df = pd.DataFrame(year_scores).T
year_score_df['year'] = year_score_df.index
year_score_df.to_csv('vader_year.csv',index=False)
# %%
