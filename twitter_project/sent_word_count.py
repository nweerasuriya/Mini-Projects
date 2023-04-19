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


# %% --------------------------------------------------------------------------
# Load Twitter Data
# -----------------------------------------------------------------------------
df = pd.read_csv('data/textoutput',header=0)
df

# %% --------------------------------------------------------------------------
# Get Sentiment Dictionary
# -----------------------------------------------------------------------------
def get_sentiment_word_dict():
    #Read csv dictionary
    loc = "data/Loughran-McDonald_MasterDictionary_1993-2021.csv"
    df = pd.read_csv(loc)
    df = df[['Word','Negative','Positive','Uncertainty','Litigious','Constraining']]
    df.set_index('Word', inplace = True)
    
    my_dict = {}
    sentiment = ['Negative','Positive','Uncertainty','Litigious','Constraining']
    
    for sent in sentiment:
        neg_df = df.drop(columns = [x for x in sentiment if x != sent])[df[sent]!=0]
        neg_df.loc[neg_df[sent] !=0 , sent] = sent
        neg_df = neg_df.reset_index()
        neg_words = neg_df['Word'].tolist()
        my_dict[sent] = neg_words
    
    return my_dict


# %% --------------------------------------------------------------------------
# Sentiment Word Count
# -----------------------------------------------------------------------------

def count_sentiment(input_data,channel):
    sentiment_dict = get_sentiment_word_dict()
    index_count = len(input_data)
    print(index_count)
    temporary_dict = {  'Channel': channel,
                        'Negative': 0, 
                        'Positive': 0, 
                        'Uncertainty': 0, 
                        'Litigious': 0, 
                        'Constraining': 0,
                        'NormNegative': 0,
                        'NormPositive': 0,
                        'NormUncertainty': 0,
                        'NormLitigious': 0,
                        'NormConstraining': 0}

    # the following block creates an individual dictionary for each 10k file, and then appends it to list_of_dicts
    for sentiment in sentiment_dict.keys(): 
        for sentiment_word in sentiment_dict[sentiment]:
            count = sum(input_data.str.count(sentiment_word).fillna(0))
            count += sum(input_data.str.count(sentiment_word.lower()).fillna(0))
            count += sum(input_data.str.count(sentiment_word.title()).fillna(0))
            temporary_dict[sentiment] += count
        normalised_sent = 1000*(temporary_dict[sentiment]/index_count) # Multiply by 1000 to give nice values
        temporary_dict['Norm'+ sentiment] = normalised_sent
    return temporary_dict

# %% --------------------------------------------------------------------------
# Run Word count for each channel
# -----------------------------------------------------------------------------
channel_list = df['channel'].unique()
sent_count = []
for i in channel_list:
    channel_df = df.loc[df['channel']==i]
    sent_count.append(count_sentiment(channel_df['Preprocessed Tweet'],i))

sent_df = pd.DataFrame(sent_count)
sent_df


# %% --------------------------------------------------------------------------
# Save data as csv
# -----------------------------------------------------------------------------
sent_df.to_csv('normalised_sentiment_word_count_2.csv',index=False)


# %%
# %%

