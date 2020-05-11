import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import CleanData, SaveState
import importlib
importlib.reload(CleanData)
importlib.reload(SaveState)
from CleanData import clean_comment, get_bag_of_words
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import random
from random import sample

from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn import metrics

import time
start_time = time.time()

extreme_frac = 0.2   # This extreme_frac stands for the percentage of media to be selected as left/right, high/low media. i.e. _extreme_frac_ leftmost media are selected as left media
training_frac = 0.5  # This sample_frac stands for the percentage of (left/right, high/low) media to be sampled as training data


# Load all article reviews from MediaBiasChart V5.0:
media_bias = pd.read_csv('../data/MediaBiasChart.csv')
media_bias = media_bias.groupby('Source').mean()
media_bias.reset_index(level=0, inplace=True)

MediaBiasChart_to_Tweet = {'ABC':'ABC','AP':'AP','Axios':'axios','CNN':'CNN','Wall Street Journal':'WSJ',\
    'The Atlantic':'TheAtlantic','The Hill':'thehill', 'BBC':'BBC', 'Think Progress':'thinkprogress',\
    'MSNBC':'MSNBC','The Nation':'thenation','Daily Beast':'thedailybeast','Mother Jones':'MotherJones',\
    'CNSNews':'cnsnews','Fox News':'FoxNews', 'The Federalist':'FDRLST','Breitbart':'BreitbartNews',\
    'Daily Caller':'DailyCaller','The Blaze':'theblaze','Business Insider':'businessinsider',\
    'CBS':'CBSNews','The Economist':'TheEconomist','BuzzFeed':'BuzzFeed','Daily Signal':'DailySignal',\
    'New Republic':'newrepublic','Foreign Policy':'ForeignPolicy','IJR':'TheIJR','National Review':'NRO',\
    'National Public Radio':'NPR','New York Post':'nypost','New York Times':'nytimes','The New Yorker':'NewYorker',\
    'NewsMax':'newsmax','One America News Network':'OANN','Politico':'politico','Quartz':'qz',\
    'Reason':'reason','Reuters':'Reuters','Slate':'Slate','Talking Points Memo':'TPM','Vanity Fair':'VanityFair',\
    'Vox':'voxdotcom','Washington Examiner':'dcexaminer','Washington Free Beacon':'FreeBeacon',\
    'Washington Post':'washingtonpost','Washington Times':'WashTimes','The Week':'TheWeek','Bloomberg':'Bloomberg',\
    'Christian Science Monitor':'csmonitor', 'Democracy Now':'democracynow','Financial Times':'FT',\
    'Fiscal Times':'TheFiscalTimes','Forbes':'Forbes','Fortune':'FortuneMagazine','Forward':'jdforward',\
    'FreeSpeech TV':'freespeechtv','Huffington Post':'HuffPost','LA Times':'latimes','Marketwatch':'MarketWatch',\
    'OZY':'ozy','PBS':'PBS','ProPublica':'propublica','Time':'TIME','USA Today':'USATODAY',\
    'Weather.com':'weatherchannel'}

media_bias['Source'] = media_bias.Source.map(MediaBiasChart_to_Tweet)
media_bias = media_bias.dropna()
media_bias = media_bias.reset_index(drop=True)

left_bound = media_bias.Bias.quantile(extreme_frac)
right_bound = media_bias.Bias.quantile(1-extreme_frac)
low_bound = media_bias.Quality.quantile(extreme_frac)
high_bound = media_bias.Quality.quantile(1-extreme_frac)
all_media = media_bias['Source'].tolist()
left_media = media_bias.loc[media_bias['Bias']<=left_bound]
left_media = left_media['Source'].tolist()
right_media = media_bias.loc[media_bias['Bias']>=right_bound]
right_media = right_media['Source'].tolist()
low_media = media_bias.loc[media_bias['Quality']<=low_bound]
low_media = low_media['Source'].tolist()
high_media = media_bias.loc[media_bias['Quality']>=high_bound]
high_media = high_media['Source'].tolist()


df1 = pd.read_csv('../data/filtered_part1.csv', skiprows=1)
df2 = pd.read_csv('../data/filtered_part2.csv')
df3 = pd.read_csv('../data/filtered_part3.csv')
df4 = pd.read_csv('../data/filtered_part4.csv',lineterminator='\n')
df6 = pd.read_csv('../data/filtered_part6.csv')
df7 = pd.read_csv('../data/filtered_part7.csv')
df8 = pd.read_csv('../data/filtered_part8.csv')
df = pd.concat([df1, df2, df3, df4, df6, df7, df8], sort = False)
df = df[['user_screen_name', 'text']]

all_media = media_bias['Source'].tolist()
df = df.loc[df['user_screen_name'].isin(all_media)]

print('Total number of tweets: ')
print(df.shape[0])

import preprocessor as p
df['text']  = df['text'].apply(p.clean)
bag_of_words, vectorizer = get_bag_of_words(df['text'],ngram_range=(1,3), min_df=0.0002)

print('Shape of bag_of_words: ')
print(bag_of_words.shape)



random.seed(0)
train_left = sample(left_media,int(len(left_media)*training_frac))
train_right = sample(right_media,int(len(right_media)*training_frac))
test_left = list(set(left_media)-set(train_left))
test_right = list(set(right_media)-set(train_right))
print('List of left_media in training set')
print(train_left)
print('List of right_media in training set')
print(train_right)

# Train binary multinomial Naive Bayes model

def get_binary_NB_model_LR(bag_of_words, df):
    # Training data:
    class1_words = bag_of_words[df['user_screen_name'].isin(train_left),:]
    class2_words = bag_of_words[df['user_screen_name'].isin(train_right),:]
    train_tweets = np.concatenate((class1_words,class2_words))
    labels = np.concatenate((np.zeros(class1_words.shape[0]),np.ones(class2_words.shape[0])))
    nb = ComplementNB()
    nb.fit(train_tweets, labels)
    # # Performance on training data
    predictions = nb.predict(train_tweets)
    print('Training Accuracy: ' + str(sum(labels==predictions)/len(labels)))
    # Compute the error.
    tn, fp, fn, tp = metrics.confusion_matrix(labels,predictions).ravel()
    print(tn, fp, fn, tp)
    return nb
nb_model = get_binary_NB_model_LR(bag_of_words, df)


predict_probs = nb_model.predict_proba(bag_of_words)
df['right_prob'] = predict_probs[:,1]
average_right_prob = df.groupby(['user_screen_name']).right_prob.mean()

media_bias = media_bias.sort_values(by = 'Source')
bias = media_bias.Bias
plt.figure(figsize=(13, 8))
plt.xlabel('Bias from MediaBiasChart', fontsize=24)
plt.ylabel('Mean right_probability from model', fontsize=24)
plt.scatter(bias.tolist(), average_right_prob.tolist())
plt.savefig('../results/half_media/biasvs_frac'+str(int(100*extreme_frac))+'.png')





print("---Execution time2: %s seconds ---" % (time.time() - start_time))