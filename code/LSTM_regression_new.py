import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from random import sample
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

print('Loading and Processing Data ...')
df = pd.read_csv('../data/filtered_all.csv')
coor = pd.read_csv('../data/Media_coordinate.csv')

df2 = df.groupby('user_screen_name').head(1000).reset_index(drop=True)

def assign_coor(df, coor):
    media2bias = dict(zip(coor.Source, coor.Bias))
    df['bias'] = df['user_screen_name'].map(media2bias)
    df['bias'] = df['bias']/df['bias'].abs().max()
    media2qual = dict(zip(coor.Source, coor.Quality))
    df['quality'] = df['user_screen_name'].map(media2qual)
    df['quality'] = df['quality'] - df['quality'].abs().min()
    df['quality'] = df['quality']/df['quality'].abs().max()
    df = df.sample(frac=1).reset_index(drop=True)

    return df
    
df = assign_coor(df2, coor)

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
# print(len(labels))
# print(len(articles))
# print((articles[0],labels[0]))

print('Loading and Processing Data Complete!')
vocab_size = 5000
embedding_dim = 64
max_length = 70
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_size = .8


train_size = int(len(articles) * training_size)
train_articles = articles[0: train_size]
train_labels = labels[0: train_size]

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_articles)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_articles)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
validation_articles = articles[training_size:]
validation_labels = labels[training_size:]
validation_sequences = tokenizer.texts_to_sequences(validation_articles)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
training_label_seq = np.array(train_labels).astype('float32')
validation_label_seq = np.array(validation_labels).astype('float32')

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    tf.keras.layers.Dense(2, activation = 'softmax')
])
model.summary()

model.compile(optimizer='adam', loss="mse")
num_epochs = 10
history = model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)

train_predict = model.predict(train_padded)
test_predict = model.predict(validation_padded)
bias_pred = test_predict[:,0]
portionqual_pred = test_predict[:,1]
df_test = df.tail(int(df.shape[0]-training_size))
df_test['bias_pred'] = bias_pred
df_test['qual_pred'] = qual_pred

# coor = coor.sort_values(by = ['Source'])
average_bias = df_test.groupby(['user_screen_name']).bias_pred.mean()
average_qual = df_test.groupby(['user_screen_name']).qual_pred.mean()
bias = df.groupby(['user_screen_name']).bias.mean()
quality = df.groupby(['user_screen_name']).quality.mean()

corr_bias = scipy.stats.pearsonr(average_bias.tolist(), bias.tolist())[0]
corr_quality = scipy.stats.pearsonr(average_qual.tolist(), quality.tolist())[0]

print(corr_bias, corr_quality)