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
import glob
import os
from wordcloud import WordCloud
# %% --------------------------------------------------------------------------
# Load Twitter Data
# -----------------------------------------------------------------------------
df = pd.read_csv('data/ldaoutput',header=0)
df = df.drop(['Tweet'],axis=1)


# %% --------------------------------------------------------------------------
# Find assigned Topic and Probability
# -----------------------------------------------------------------------------



def find_topic(df, num):
    num_list = list(range(1,num+1))
    column_dict = {}
    for n in num_list:
        column_dict[f'Topic {n}'] = n
    df1 = df.rename(column_dict,axis=1)
    df1['Topic_prob'] = df1[num_list].max(axis=1)
    df1['overall'] = df1[num_list].idxmax(axis=1)
    return df1.drop(num_list,axis=1)

df2 = find_topic(df,3)
# # %%
# df1 = df.rename({'Topic 1':1,'Topic 2':2,'Topic 3':3,'Topic 4':4},axis=1)
# df1['overall'] = df1[[1,2,3,4]].idxmax(axis=1)
# df2 = df1.rename({1:'Topic 1',2:'Topic 2',3:'Topic 3',4:'Topic 4'},axis=1)
# df2
# %%

# %% --------------------------------------------------------------------------
# Group by topic
# -----------------------------------------------------------------------------

topic_df = df2.groupby('overall')['Preprocessed Tweet'].sum()

wc = WordCloud(
    width=1000, height=800, min_font_size=10, collocations=False)
fig, axes = plt.subplots(2,2,figsize=(16,12))



for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = topic_df.iloc[i]['Preprocessed Tweet']
    wc.generate(topic_words)
    plt.gca().imshow(wc)
    plt.gca().set_title('Topic ' + str(i+1), fontdict=dict(size=16),c='w')
    plt.gca().axis('off')

plt.savefig('topic_wordcloud.png')
plt.tight_layout()
plt.show()
# %%
plt.savefig('topic_wordcloud.jpg')
# %%
pd.read_csv('topic_output')
# %%
df2.to_csv('topic_alldf.csv',index=False)
# %%
