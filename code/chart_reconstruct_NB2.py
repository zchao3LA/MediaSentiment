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
import argparse
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn import metrics
import scipy.stats
import time
start_time = time.time()

parser = argparse.ArgumentParser(description='params')
parser.add_argument('--media', type=float, default=0.2, help='portion of media to be considered as left/right/low/high')
parser.add_argument('--train', type=float, default=0.8, help='portion of media for training')
parser.add_argument('--max_tweets', type=int, default=30000, help='portion of media for training')
args = parser.parse_args()
# extreme_frac = 0.2   # This extreme_frac stands for the percentage of media to be selected as left/right, high/low media. i.e. _extreme_frac_ leftmost media are selected as left media
# training_frac = 0.5  # This sample_frac stands for the percentage of (left/right, high/low) media to be sampled as training data
# training_frac = float(input("Enter a fraction for training set: (default = 0.5)") or '0.5')

max_tweets = args.max_tweets
extreme_frac = args.media
training_frac = args.train
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

### Take only _max_tweets_ tweets for each media
df = df.sample(frac=1).reset_index(drop=True)
df = df.groupby('user_screen_name').head(max_tweets).reset_index(drop=True)
###

print('Total number of tweets: ')
print(df.shape[0])

import preprocessor as p
df['text']  = df['text'].apply(p.clean)
bag_of_words, vectorizer = get_bag_of_words(df['text'],ngram_range=(1,3), min_df=0.0002)

print('Shape of bag_of_words: ')
print(bag_of_words.shape)



random.seed(0)
df['sampled'] = np.random.randint(2, size=df.shape[0])

# Train binary multinomial Naive Bayes model for Left/Right Bias
def NB_model_bias(bag_of_words, df):
    # Training data:
    class1_words = bag_of_words[(df['user_screen_name'].isin(left_media)) & (df['sampled'] == 1),:]
    class2_words = bag_of_words[(df['user_screen_name'].isin(right_media)) & (df['sampled'] == 1),:]
    train_tweets = np.concatenate((class1_words,class2_words))
    labels = np.concatenate((np.zeros(class1_words.shape[0]),np.ones(class2_words.shape[0])))
    nb = ComplementNB()
    nb.fit(train_tweets, labels)
    # # Performance on training data
    predictions = nb.predict(train_tweets)
    print('Bias Training Accuracy: ' + str(sum(labels==predictions)/len(labels)))
    # Compute the error.
    tn, fp, fn, tp = metrics.confusion_matrix(labels,predictions).ravel()
    print(tn, fp, fn, tp)
    return nb

nb_bias = NB_model_bias(bag_of_words, df)
predict_bias = nb_bias.predict_proba(bag_of_words)
df['right_prob'] = predict_bias[:,1]


average_right_prob = df[~(df['user_screen_name'].isin(left_media + right_media) & df['sampled'] == 1)].groupby(['user_screen_name']).right_prob.mean()
media_bias = media_bias.sort_values(by = 'Source')
bias = media_bias.Bias
plt.scatter(bias.tolist(), average_right_prob.tolist())

corr_bias = scipy.stats.pearsonr(average_right_prob.tolist(), bias.tolist())[0]

plt.figure(figsize=(13, 8))
plt.xlabel('Bias from MediaBiasChart', fontsize=24) 
plt.ylabel('Mean right_probability from model', fontsize=24)
plt.scatter(bias.tolist(), average_right_prob.tolist())
plt.title('Correlation between V5.0 bias and reconstructed bias: '+str(corr_bias))
plt.savefig('../results/half_tweet/vsbias_frac'+str(int(100*extreme_frac))+'.png')


# Train binary multinomial Naive Bayes model for Low/High Quality
def NB_model_qual(bag_of_words, df):
    # Training data:
    class1_words = bag_of_words[(df['user_screen_name'].isin(low_media)) & (df['sampled'] == 1),:]
    class2_words = bag_of_words[(df['user_screen_name'].isin(high_media)) & (df['sampled'] == 1),:]
    train_tweets = np.concatenate((class1_words,class2_words))
    labels = np.concatenate((np.zeros(class1_words.shape[0]),np.ones(class2_words.shape[0])))
    nb = ComplementNB()
    nb.fit(train_tweets, labels)
    # # Performance on training data
    predictions = nb.predict(train_tweets)
    print('Quality Training Accuracy: ' + str(sum(labels==predictions)/len(labels)))
    # Compute the error.
    tn, fp, fn, tp = metrics.confusion_matrix(labels,predictions).ravel()
    print(tn, fp, fn, tp)
    return nb

nb_qual = NB_model_qual(bag_of_words, df)
predict_qual = nb_qual.predict_proba(bag_of_words)
df['high_prob'] = predict_qual[:,1]


average_high_prob = df[~(df['user_screen_name'].isin(low_media + high_media) & df['sampled'] == 1)].groupby(['user_screen_name']).high_prob.mean()
quality = media_bias.Quality
corr_quality = scipy.stats.pearsonr(average_high_prob.tolist(), quality.tolist())[0]

plt.figure(figsize=(13, 8))
plt.xlabel('Quality from MediaBiasChart', fontsize=24)
plt.ylabel('Mean high_quality_probability from model', fontsize=24)
plt.scatter(quality.tolist(), average_high_prob.tolist())
plt.title('Correlation between V5.0 quality and reconstructed quality: '+str(corr_quality))
plt.savefig('../results/half_tweet/vsqual_frac'+str(int(100*extreme_frac))+'.png')



bias_correct = sum(df[(df['user_screen_name'].isin(left_media)) & (df['sampled'] == 0)].right_prob <= 0.5) + \
               sum(df[(df['user_screen_name'].isin(right_media)) & (df['sampled'] == 0)].right_prob > 0.5)
qual_correct = sum(df[(df['user_screen_name'].isin(low_media)) & (df['sampled'] == 0)].high_prob <= 0.5) + \
               sum(df[(df['user_screen_name'].isin(high_media)) & (df['sampled'] == 0)].high_prob > 0.5)

print('Bias Testing Accuracy: ' + str(bias_correct/sum((df['user_screen_name'].isin(left_media+right_media)) & (df['sampled'] == 0))))
print('Quality Testing Accuracy: ' + str(qual_correct/sum((df['user_screen_name'].isin(high_media+low_media)) & (df['sampled'] == 0))))

print('Bias Correlation: ' + str(corr_bias))
print('Quality Correlation: ' + str(corr_quality))


n_tweets = df.groupby(['user_screen_name']).size()  
n_tweets = n_tweets.rename("n_tweets")
N = media_bias.shape[0]
np.random.seed(1)
colors = np.random.rand(N)
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
    size = [x**2 / 900000 for x in media_bias[index].n_tweets.tolist()]
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
dir1 = '../results/half_tweet/media_all_origin.png'
dir2 = '../results/half_tweet/media_all_recon.png'
plotsubset(names, dir1, dir2)

names = left_media
dir1 = '../results/half_tweet/media_left_origin.png'
dir2 = '../results/half_tweet/media_left_recon.png'
plotsubset(names, dir1, dir2)

names = right_media
dir1 = '../results/half_tweet/media_right_origin.png'
dir2 = '../results/half_tweet/media_right_recon.png'
plotsubset(names, dir1, dir2)

names = high_media
dir1 = '../results/half_tweet/media_high_origin.png'
dir2 = '../results/half_tweet/media_high_recon.png'
plotsubset(names, dir1, dir2)

names = low_media
dir1 = '../results/half_tweet/media_low_origin.png'
dir2 = '../results/half_tweet/media_low_recon.png'
plotsubset(names, dir1, dir2)

names = list(set(media_bias['Source'].tolist()) - set(left_media + right_media + high_media + low_media))
dir1 = '../results/half_tweet/media_rest_origin.png'
dir2 = '../results/half_tweet/media_rest_recon.png'
plotsubset(names, dir1, dir2)


print("---Execution done in: %s seconds ---" % (time.time() - start_time))