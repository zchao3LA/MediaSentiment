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
import scipy.stats
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

train_high = sample(high_media,int(len(high_media)/2))
train_low = sample(low_media,int(len(low_media)/2))
test_high = list(set(high_media)-set(train_high))
test_low = list(set(low_media)-set(train_low))
print('List of high_media in training set')
print(train_high)
print('List of low_media in training set')
print(train_low)



# Train binary multinomial Naive Bayes model

def NB_model_bias(bag_of_words, df):
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
nb_bias = NB_model_bias(bag_of_words, df)

print("---Execution time0.1: %s seconds ---" % (time.time() - start_time))

predict_bias = nb_bias.predict_proba(bag_of_words)


print("---Execution time0.2: %s seconds ---" % (time.time() - start_time))

df['right_prob'] = predict_bias[:,1]
average_right_prob = df.groupby(['user_screen_name']).right_prob.mean()

print("---Execution time0.3: %s seconds ---" % (time.time() - start_time))
media_bias = media_bias.sort_values(by = 'Source')
bias = media_bias.Bias

corr_bias = scipy.stats.pearsonr(average_right_prob.tolist(), bias.tolist())[0]

plt.figure(figsize=(13, 8))
plt.xlabel('Bias from MediaBiasChart', fontsize=24)	
plt.ylabel('Mean right_probability from model', fontsize=24)
plt.scatter(bias.tolist(), average_right_prob.tolist())
plt.title('Correlation between V5.0 bias and reconstructed bias: '+str(corr_bias))
plt.savefig('../results/half_media/biasvs_frac'+str(int(100*extreme_frac))+'.png')


print("---Execution time0.4: %s seconds ---" % (time.time() - start_time))


# Train binary multinomial Naive Bayes model. This one is for Low/High Quality
def NB_model_qual(bag_of_words, df):
    # Training data:
    class1_words = bag_of_words[df['user_screen_name'].isin(train_low),:]
    class2_words = bag_of_words[df['user_screen_name'].isin(train_high),:]
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
nb_qual = NB_model_qual(bag_of_words, df)
predict_qual = nb_qual.predict_proba(bag_of_words)
df['high_prob'] = predict_qual[:,1]

print("---Execution time0.5: %s seconds ---" % (time.time() - start_time))

average_high_prob = df.groupby(['user_screen_name']).high_prob.mean()
quality = media_bias.Quality
corr_quality = scipy.stats.pearsonr(average_high_prob.tolist(), quality.tolist())[0]

plt.figure(figsize=(13, 8))
plt.xlabel('Quality from MediaBiasChart', fontsize=24)
plt.ylabel('Mean high_quality_probability from model', fontsize=24)
plt.scatter(quality.tolist(), average_high_prob.tolist())
plt.title('Correlation between V5.0 quality and reconstructed quality: '+str(corr_quality))
plt.savefig('../results/half_media/qualvs_frac'+str(int(100*extreme_frac))+'.png')



testLR_correct = sum(df[df['user_screen_name'].isin(test_left)].right_prob <= 0.5) + \
                 sum(df[df['user_screen_name'].isin(test_right)].right_prob > 0.5)
testLH_correct = sum(df[df['user_screen_name'].isin(test_low)].high_prob <= 0.5) + \
                 sum(df[df['user_screen_name'].isin(test_high)].high_prob > 0.5)
print('Bias Testing Accuracy: ' + str(testLR_correct/sum(df['user_screen_name'].isin(test_left+test_right))))
print('Quality Testing Accuracy: ' + str(testLH_correct/sum(df['user_screen_name'].isin(test_low+test_high))))



print("---Execution time1: %s seconds ---" % (time.time() - start_time))



n_tweets = df.groupby(['user_screen_name']).size()
n_tweets = n_tweets.rename("n_tweets")
media_bias = media_bias.set_index('Source').join(average_right_prob).join(average_high_prob).join(n_tweets)
media_bias = media_bias.dropna()
media_bias = media_bias.reset_index()


def plotsubset(names, dir1, dir2):
    index = media_bias.Source.isin(names)

    names = media_bias[index].Source.tolist()
    x0 = [(x+30)/60 for x in media_bias[index].Bias.tolist()]
    y0 = [(x-20)/40 for x in media_bias[index].Quality.tolist()]
    x1 = media_bias[index].right_prob.tolist()
    y1 = media_bias[index].high_prob.tolist()
    size = [x**2 / 100000 for x in media_bias[index].n_tweets.tolist()]
    color = pd.Series(colors)[index].tolist()
    
    plt.figure(figsize=(16, 10))
    # pd.Series(size)
    plt.scatter(x0, y0, s=size, c=color, alpha=0.5)
    for i, name in enumerate(names):
        plt.annotate(name, (x0[i], y0[i]))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Original Bias', fontsize=24)
    plt.ylabel('Original Quality', fontsize=24)
    plt.savefig(dir1)
    

    plt.figure(figsize=(16, 10))
    for i, name in enumerate(names):
        plt.annotate(name, (x1[i], y1[i]))
    plt.scatter(x1, y1, s=size, c=color, alpha=0.5)
    plt.xlim(0.2, 0.8)
    plt.ylim(0.1, 0.9)
    plt.xlabel('Reconstructed Bias', fontsize=24)
    plt.ylabel('Reconstructed Quality', fontsize=24)
    plt.savefig(dir2)
    


names = media_bias.Source.tolist()
dir1 = '../results/half_media/media_all_origin.png'
dir2 = '../results/half_media/media_all_recon.png'
plotsubset(names, dir1, dir2)

names = train_left + train_right
dir1 = '../results/half_media/media_trainB_origin.png'
dir2 = '../results/half_media/media_trainB_recon.png'
plotsubset(names, dir1, dir2)

names = list(set(left_media + right_media) - set(train_left + train_right))
dir1 = '../results/half_media/media_testB_origin.png'
dir2 = '../results/half_media/media_testB_recon.png'
plotsubset(names, dir1, dir2)

names = train_high + train_low
dir1 = '../results/half_media/media_trainQ_origin.png'
dir2 = '../results/half_media/media_trainQ_recon.png'
plotsubset(names, dir1, dir2)

names = list(set(high_media + low_media) - set(train_high + train_low))
dir1 = '../results/half_media/media_testQ_origin.png'
dir2 = '../results/half_media/media_testQ_recon.png'
plotsubset(names, dir1, dir2)

names = list(set(media_bias['Source'].tolist()) - set(left_media + right_media + high_media + low_media))
dir1 = '../results/half_media/media_rest_origin.png'
dir2 = '../results/half_media/media_rest_recon.png'
plotsubset(names, dir1, dir2)