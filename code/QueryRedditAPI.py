import certifi
import urllib3
import json
import numpy as np
import pandas as pd

"""
Get data using the pushshift API for reddit data.

See https://github.com/pushshift/api for more information on query options and
http://t-redactyl.io/blog/2015/11/analysing-reddit-data-part-2-extracting-the-data.html
for a nice tutorial using the Reddit API.
"""

"""
Lists of political subreddits.

The following page lists political subreddits according to their political bias:
https://www.reddit.com/r/politics/wiki/relatedsubs
"""

left_poli_subreddits = [
    'AllTheLeft','Classical_Liberals','CornbreadLiberals','Democrats','Demsocialist',
    'GreenParty','Labor','Leftcommunism','Leninism','Liberal','NeoProgs','Obama','Progressive',
    'SocialDemocracy','Socialism'
    ]
right_poli_subreddits = [
    'Conservative','Conservatives','Monarchism','New_Right','Objectivism','Paleoconservative',
    'Republican','Republicans','Romney','TrueObjectivism'
    ]
center_poli_subreddits = [
    'NeutralPolitics','Centrist','ModeratePolitics','PeoplesParty'
    ]
libert_poli_subreddits = [
    'Agorism','Anarcho_Capitalism','AnarchObjectivism','Christian_Ancaps','Libertarian',
    'LibertarianDebates','LibertarianLeft','LibertarianMeme',
    'LibertarianSocialism','LibertarianWomen','Paul','RonPaul','TrueLibertarian','Voluntarism'
    ]


"""
Gets dictionary of data given query url
"""
def get_data_dict_list(url):
    http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',ca_certs=certifi.where())
    data_req = http.request('GET', url.replace(' ','%20'))
    data = json.loads(data_req.data.decode('utf-8'))['data']
    return data

"""
Given a dictionary containing parameters for a query to the reddit API and a list of subreddits to search,
this function returns a DataFrame with comments relating to the query.

The query dictionary should take the form
query_dict = {
    'title' : keyword, # keyword must be contained in title
    'before' : '2019-03-02',
    'after' : '2019-02-27',
    'sort' : 'asc',
    'size' : '1000'
}
Here, the keywordn(title parameter) is required, but the remaining parameters are optional.
An exhaustive list of parameter options can be found at https://github.com/pushshift/api
The title parameter indicates which keyword must occur in submission titles.
subreddits is a list of subreddits to be searched.

"""

def getDataFrame(query_dict, subreddits):
    comment_data = [] # to store dictionaries with comment data

    for subreddit in subreddits:
        query_dict['subreddit'] = subreddit

        # Get first 1000 submissions with keyword from subreddit in date range
        query_string = '&'.join(k + '=' + v for k,v in query_dict.items())
        url = 'https://api.pushshift.io/reddit/search/submission/?' + query_string
        data = get_data_dict_list(url)
        if len(data)>1000: print('More than 1000 submissions in subreddit for date range')
        data_DF = pd.DataFrame(data)

        if data_DF.empty:
            print(subreddit + ' subreddit contains no posts with keyword in date range.')
        else:
            # Get comment ids
            comment_ids = []
            for post_id in data_DF[data_DF['num_comments'] > 0]['id']: # Posts with comments
                url = 'https://api.pushshift.io/reddit/submission/comment_ids/' + post_id
                new_ids = get_data_dict_list(url)
                comment_ids.extend(new_ids)

            # Collect comments
            # Need to group into 1000s otherwise it will break
            num_blocks = int(len(comment_ids)/1000)
            if len(comment_ids) % 1000 > 0: num_blocks = num_blocks + 1
            for i in range(num_blocks):
                comment_ids_string = ','.join(comment_ids[i*1000:min((i+1)*1000,len(comment_ids))])
                url = 'https://api.pushshift.io/reddit/comment/search?ids=' + comment_ids_string
                comment_data.extend(get_data_dict_list(url))

    # convert to dataframe and save
    return  pd.DataFrame(comment_data)
