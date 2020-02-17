import pandas as pd
import string
from nltk.corpus import stopwords
import re # For regex processing
from sklearn.feature_extraction.text import CountVectorizer
# import numpy as np
# from sklearn.decomposition import NMF
from os.path import join
from SaveState import repo_root
data_path = join(repo_root,'data')

"""
Load datasets -
The dataset name needs to be the path to the data from the folder indicated by data_path
Returns a DataFrame with a column that contains cleaned preprocessed comments
"""
def load_dataset(dataset_name):
    df = pd.read_csv(join(data_path, dataset_name))

    columns = [ 'author',
           'author_fullname',  'banned_at_utc', 'body','created_utc',
           'distinguished', 'edited', 'gildings', 'id', 'is_submitter', 'link_id',
           'no_follow', 'parent_id', 'permalink', 'score',
           'send_replies', 'stickied', 'subreddit']
    df = df[columns]
    df = clean_dataframe(df)
    return df

def clean_dataframe(df):
    # Create column with cleaned comments
    df['clean_comment'] = df['body'].apply(clean_comment)

    # Remove unwanted rows from DataFrame
    df = df.loc[df['distinguished']!='moderator'] # moderator comments
    df = df.loc[df['body']!='[removed]'] # deleted comments
    df = df.loc[df['body']!='NaN']
    df = df.loc[~df['body'].str.contains('This comment has been removed')] # deleted (Neutral Politics)

    return df

def clean_comments(df):
    [clean_comment(comment) for comment in df['body']]
    return df

"""
Input a string to be cleaned.
"""
def clean_comment(comment):
    comment = comment.lower()
    to_replace = ['\n','\r','&gt;','&lt','&ge','&le','\'ll','\'ve', \
                  '\'t','\'d','\'s','\'re']
    for expr in to_replace: comment = comment.replace(expr,' ')
    comment = re.sub(r'http\S+', '', comment) # Remove links
    comment = re.sub(r'www\S+', '', comment) # Remove links
    comment = re.sub('[0-9]', '', comment) # Remove digits
    comment = comment.replace('[^\w\s]','')
    comment = comment.translate(str.maketrans('', '', string.punctuation))

    return comment


"""
Builds a bag of words representation of the cleaned comments in a dataframe.
Removes stop words included in nltk's 'english' stop word list,
as well as string representations of the numbers 1-10.
"""
def get_bag_of_words(comments,ngram_range=(1,2),min_df=2):
    # Load stop words
    stop_words = stopwords.words('english')
    number_strs = ['one', 'two', 'three', 'four', 'five', 'six', 'seven',\
                  'eight', 'nine', 'ten']
    stop_words.extend(number_strs)

    ngram_vectorizer = CountVectorizer(analyzer='word', ngram_range=ngram_range,
                                        min_df=min_df, stop_words=stop_words)
    
    try:
        bag_of_words = ngram_vectorizer.fit_transform(comments).toarray()
    except: # ValueError:
        print('Empty vocaulary')
        return [],[]

    # vocab = ngram_vectorizer.get_feature_names()
    return bag_of_words, ngram_vectorizer
    # Potentially useful info when exploring
#     word_counts = np.sum(bag_of_words,axis=0)
#     words_and_counts = list(zip(vocab, np.asarray(bag_of_words.sum(axis=0)).ravel()))
#     words_and_counts.sort(key = lambda x: x[1], reverse = True)
#     vocab.index("remove")





# Some comments are empty - or contain just special characters (eg "**!**")
# "No" was included in the stop words, but "yes" was not so I am not removing stop words
