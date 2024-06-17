# Demonstrating Topic Modelling and Sentiment Analysis
## Topic Modelling (Azure ML)

Used the LDA Module in Azure Designer to split tweets into 4 topics 

![image](https://user-images.githubusercontent.com/65176466/233098511-f37e91ce-6d17-4cef-a42c-fa6df59ff5f5.png)




![image](https://user-images.githubusercontent.com/65176466/233098615-7b6af1f6-a347-4685-ad89-b9fa8d11f3a1.png)



## Sentiment Analysis

#### Sentiment Wordcount
-Loughran-McDonald master dictionary ['Negative','Positive','Uncertainty','Litigious','Constraining']

-Count number of words matching dictionary

-Normalise over the number of words in the tweet


##### VADER Sentiment Analysis
-Group dataset by channel and year independently

-Run VADER analysis on grouped tweets to produce sentiment/polarity score

-Sentiment score is between -1 (negative sentiment) and +1 (positive sentiment)



![image](https://user-images.githubusercontent.com/65176466/233106098-cbaf0091-3912-480a-a178-faf8e8d6387d.png)



![image](https://user-images.githubusercontent.com/65176466/233106189-5ccb545d-df5b-47ef-b0c6-2456d62f9c4a.png)



