import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
print('Loading and Processing Data ...')
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
media_bias.reset_index(drop=True)


# Load tweets and filter out a small portion which does not have a proper 'username'
df1 = pd.read_csv('../data/filtered_part1.csv', skiprows=1)
df2 = pd.read_csv('../data/filtered_part2.csv')
df3 = pd.read_csv('../data/filtered_part3.csv')
df4 = pd.read_csv('../data/filtered_part4.csv',lineterminator='\n')
df6 = pd.read_csv('../data/filtered_part6.csv')
df7 = pd.read_csv('../data/filtered_part7.csv')
df8 = pd.read_csv('../data/filtered_part8.csv')
df = pd.concat([df1, df2, df3, df4, df6, df7, df8], sort = False)

all_media = media_bias['Source'].tolist()
df = df.loc[df['user_screen_name'].isin(all_media)]
print('Loaded tweet size: ',df.shape)
print('Loading and Processing Complete !')

import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))



df = df.sample(frac=1).reset_index(drop=True)  #Shuffle
media2bias = dict(zip(media_bias.Source, media_bias.Bias))
df['bias'] = df['user_screen_name'].map(media2bias)
df['bias'] = df['bias']/df['bias'].abs().max()

media2qual = dict(zip(media_bias.Source, media_bias.Quality))
df['quality'] = df['user_screen_name'].map(media2qual)
df['quality'] = df['quality'] - df['quality'].abs().min()
df['quality'] = df['quality']/df['quality'].abs().max()

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
print(len(labels))
print(len(articles))
print((articles[0],labels[0]))


vocab_size = 5000
embedding_dim = 64
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .8


train_size = int(len(articles) * training_portion)
train_articles = articles[0: train_size]
train_labels = labels[0: train_size]

validation_articles = articles[train_size:]
validation_labels = labels[train_size:]


tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_articles)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_articles)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
validation_sequences = tokenizer.texts_to_sequences(validation_articles)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
training_label_seq = np.array(train_labels).astype('float32')
validation_label_seq = np.array(validation_labels).astype('float32')

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    # Add a Dense layer with 2 units and softmax activation.
    # When we have multiple outputs, softmax convert outputs layers into a probability distribution.
    tf.keras.layers.Dense(2, activation = 'softmax')
])
model.summary()

model.compile(optimizer='adam', loss="mse")
num_epochs = 5
history = model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)

train_predict = model.predict(train_padded)
test_predict = model.predict(validation_padded)
bias_pred = test_predict[:,0]
qual_pred = test_predict[:,1]
df_test = df.tail(df.shape[0]-train_size)
df_test['bias_pred'] = bias_pred
df_test['qual_pred'] = qual_pred

import scipy
media_bias = media_bias.sort_values(by = ['Source'])
average_bias = df_test.groupby(['user_screen_name']).bias_pred.mean()
average_qual = df_test.groupby(['user_screen_name']).qual_pred.mean()
bias = df_test.groupby(['user_screen_name']).bias.mean()
quality = df_test.groupby(['user_screen_name']).quality.mean()

corr_bias = scipy.stats.pearsonr(average_bias.tolist(), bias.tolist())[0]
corr_quality = scipy.stats.pearsonr(average_qual.tolist(), quality.tolist())[0]

print(corr_bias, corr_quality)

# plt.figure(figsize=(13, 8))
# plt.xlabel('Bias from MediaBiasChart', fontsize=24) 
# plt.ylabel('Mean Bias from model', fontsize=24)
# plt.scatter(bias.tolist(), average_bias.tolist())
# plt.title('Correlation between V5.0 bias and reconstructed bias: '+str(corr_bias))
# plt.savefig('../results/LSTM/biasvs.png')

# plt.figure(figsize=(13, 8))
# plt.xlabel('Quality from MediaBiasChart', fontsize=24)
# plt.ylabel('Mean Quality from model', fontsize=24)
# plt.scatter(quality.tolist(), average_qual.tolist())
# plt.title('Correlation between V5.0 quality and reconstructed quality: '+str(corr_quality))
# plt.savefig('../results/LSTM/qualvs.png')