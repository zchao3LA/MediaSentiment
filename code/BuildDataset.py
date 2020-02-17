from pandas import Series, DataFrame
import pandas as pd
import os
from SaveState import get_output_dir
from QueryRedditAPI import getDataFrame

"""
Lists of political subreddits.

The following page lists political subreddits according to their political bias:
https://www.reddit.com/r/politics/wiki/relatedsubs
"""

left_poli_subreddits = ['Democrats','Liberal']
# removed: 'AllTheLeft','Classical_Liberals','CornbreadLiberals','Leninism',
# 'GreenParty','Labor','Leftcommunism','Socialism','NeoProgs','Obama',
# 'Progressive','Demsocialist','SocialDemocracy'
right_poli_subreddits = [
    'Conservative','Conservatives','Republican','Republicans'
    ]
# removed:'Monarchism','New_Right','Objectivism','Paleoconservative',
# 'Romney','TrueObjectivism'
center_poli_subreddits = [
    'NeutralPolitics','Centrist','ModeratePolitics','PeoplesParty'
    ]
libert_poli_subreddits = [
    'Agorism','Anarcho_Capitalism','AnarchObjectivism','Christian_Ancaps','Libertarian',
    'LibertarianDebates','LibertarianLeft','LibertarianMeme',
    'LibertarianSocialism','LibertarianWomen','Paul','RonPaul','TrueLibertarian','Voluntarism'
    ]
socialist_subreddits = ['socialism']#,'SocialDemocracy','demsocialist']
capitalist_subreddits = ['Capitalism','Classical_Liberals','Libertarian','neoliberal']
# political subreddits for 20 most populous states.
state_poli_subreddits = ['CalPolitics','California_Politics','TexasPolitics',
    'nyspolitics','VirginiaPolititcs','FLgovernment','illinoispolitics',
    'Pennsylvania_Politics','New_Jersey_Politics','ohiopolitics','GAPol',
    'ncpolitics','michiganpolitics','Michigan_Politics','WAlitics',
    'arizonapolitics','MarylandPolitics','MissouriPolitics','wisconsinpolitics',
    'IndianaPolitics','TennesseePolitics','MassachusettsPolitics']

"""
Get comments for submissions with a specific keyword in the title for various subreddits
"""

"""
UPDATE KEYWORD AND query_dict
"""
keyword = 'shutdown' # put + between separate keywords
if ' ' in keyword: keyword = '\"' + keyword + '\"'

subreddits = left_poli_subreddits + right_poli_subreddits \
        + center_poli_subreddits + socialist_subreddits + capitalist_subreddits

query_dict = {
    'title' : keyword, # keyword must be contained in title
    'before' : '2019-01-24',
    'after' : '2018-12-22',
    'sort' : 'asc',
    'size' : '1000'
}

df = getDataFrame(query_dict, subreddits)
print(df[['body','subreddit']].groupby(['subreddit']).count())

# Save dataframe of comments
dirpath = get_output_dir(basename=keyword, dest_folder = "data")
dirpath.replace('\"','').replace(' ','_')
path = os.path.join(dirpath, keyword)
df.to_csv(path + 'duration_reddit_comments.csv')

# Save query info to .txt file
f = open(path + "_query_info.txt","w")
f.write( str(query_dict) )
f.write( "\n\n" + str(subreddits) )
f.close()
