import pandas as pd
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
import scipy.stats
import argparse

import time
start_time = time.time()

parser = argparse.ArgumentParser(description='params')
parser.add_argument('--media', type=float, default=0.2, help='portion of media to be considered as left/right/low/high, only for plotting')
parser.add_argument('--train', type=float, default=0.8, help='portion of training data')
args = parser.parse_args()
# extreme_frac = 0.2   # This extreme_frac stands for the percentage of media to be selected as left/right, high/low media. i.e. _extreme_frac_ leftmost media are selected as left media
# training_frac = 0.5  # This sample_frac stands for the percentage of (left/right, high/low) media to be sampled as training data
# training_frac = float(input("Enter a fraction for training set: (default = 0.5)") or '0.5')

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

### Take a subset; comment this piece for the formal test ###
# df = df.loc[((df['created_at']) >= '2018-01-01') & ((df['created_at']) <= '2018-11-31')]
# df = df.reset_index(drop = True)
###

df = df[['user_screen_name', 'text']]

all_media = media_bias['Source'].tolist()
df = df.loc[df['user_screen_name'].isin(all_media)]


df = df.sample(frac=1).reset_index(drop=True)

print('Total number of tweets: ')
print(df.shape[0])

# import preprocessor as p
# df['text']  = df['text'].apply(p.clean)

vocab_size = 5000
embedding_dim = 64
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = args.train

media2bias = dict(zip(media_bias.Source, media_bias.Bias))
df['bias'] = df['user_screen_name'].map(media2bias)
df['bias'] = df['bias'] - df['bias'].min()
df['bias'] = df['bias'] / df['bias'].max()
media2qual = dict(zip(media_bias.Source, media_bias.Quality))
df['quality'] = df['user_screen_name'].map(media2qual)
df['quality'] = df['quality'] - df['quality'].min()
df['quality'] = df['quality'] / df['quality'].max()


articles = []
labels = []

for index, row in df.iterrows():
    labels.append(row[['bias','quality']])
    article = row['text']
    for word in STOPWORDS:
        token = ' ' + word + ' '
        article = article.replace(token, ' ')
        article = article.replace(' ', ' ')
    articles.append(article)

print('Total number of tweets: ')
print(len(articles))


train_size = int(len(articles) * training_portion)

train_articles = articles[0: train_size]
train_labels = labels[0: train_size]

validation_articles = articles[train_size:]
validation_labels = labels[train_size:]

print('Training Size:')
print(len(train_articles))
print('Testing Size:')
print(len(validation_articles))


tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_articles)
word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(train_articles)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

validation_sequences = tokenizer.texts_to_sequences(validation_articles)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
training_label_seq = np.array(train_labels).astype('float32')
validation_label_seq = np.array(validation_labels).astype('float32')

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_article(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    # Add a Dense layer with 6 units and softmax activation.
    # When we have multiple outputs, softmax convert outputs layers into a probability distribution.
    tf.keras.layers.Dense(2)
])
print(model.summary())


model.compile(optimizer='adam', loss="mse")
num_epochs = 10
history = model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.savefig('../results/LSTM/loss_history.png')
  
plot_graphs(history, "loss")


print("---Training done in: %s seconds ---" % (time.time() - start_time))


test_predict = model.predict(validation_padded)
bias_pred = test_predict[:,0]
qual_pred = test_predict[:,1]

df_test = df.tail(df.shape[0]-train_size)
df_test['bias_pred'] = bias_pred
df_test['qual_pred'] = qual_pred

import scipy
average_bias = df_test.groupby(['user_screen_name']).bias_pred.mean()
average_qual = df_test.groupby(['user_screen_name']).qual_pred.mean()
bias = media_bias.Bias
quality = media_bias.Quality

corr_bias = scipy.stats.pearsonr(average_bias.tolist(), bias.tolist())[0]
corr_quality = scipy.stats.pearsonr(average_qual.tolist(), quality.tolist())[0]

plt.figure(figsize=(13, 8))
plt.xlabel('Bias from MediaBiasChart', fontsize=24) 
plt.ylabel('Mean right_probability from model', fontsize=24)
plt.scatter(bias.tolist(), average_bias.tolist())
plt.title('Correlation between V5.0 bias and reconstructed bias: '+str(corr_bias))
plt.savefig('../results/LSTM/biasvs.png')


plt.figure(figsize=(13, 8))
plt.xlabel('Quality from MediaBiasChart', fontsize=24)
plt.ylabel('Mean high_quality_probability from model', fontsize=24)
plt.scatter(quality.tolist(), average_qual.tolist())
plt.title('Correlation between V5.0 quality and reconstructed quality: '+str(corr_quality))
plt.savefig('../results/LSTM/qualvs.png')



n_tweets = df.groupby(['user_screen_name']).size()
n_tweets = n_tweets.rename("n_tweets")
N = media_bias.shape[0]
np.random.seed(1)
colors = np.random.rand(N)
media_bias = media_bias.set_index('Source').join(average_bias).join(average_qual).join(n_tweets)
media_bias = media_bias.dropna()
media_bias = media_bias.reset_index()

def plotsubset(names, dir1, dir2):
    index = media_bias.Source.isin(names)

    names = media_bias[index].Source.tolist()
    x0 = [(x+30)/60 for x in media_bias[index].Bias.tolist()]
    y0 = [(x-20)/40 for x in media_bias[index].Quality.tolist()]
    x1 = media_bias[index].bias_pred.tolist()
    y1 = media_bias[index].qual_pred.tolist()
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
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Reconstructed Bias', fontsize=24)
    plt.ylabel('Reconstructed Quality', fontsize=24)
    plt.savefig(dir2)

names = media_bias.Source.tolist()
dir1 = '../results/LSTM/media_all_origin.png'
dir2 = '../results/LSTM/media_all_recon.png'
plotsubset(names, dir1, dir2)

names = left_media
dir1 = '../results/LSTM/media_left_origin.png'
dir2 = '../results/LSTM/media_left_recon.png'
plotsubset(names, dir1, dir2)

names = right_media
dir1 = '../results/LSTM/media_right_origin.png'
dir2 = '../results/LSTM/media_right_recon.png'
plotsubset(names, dir1, dir2)

names = high_media
dir1 = '../results/LSTM/media_high_origin.png'
dir2 = '../results/LSTM/media_high_recon.png'
plotsubset(names, dir1, dir2)

names = low_media
dir1 = '../results/LSTM/media_low_origin.png'
dir2 = '../results/LSTM/media_low_recon.png'
plotsubset(names, dir1, dir2)

names = list(set(media_bias['Source'].tolist()) - set(left_media + right_media + high_media + low_media))
dir1 = '../results/LSTM/media_rest_origin.png'
dir2 = '../results/LSTM/media_rest_recon.png'
plotsubset(names, dir1, dir2)