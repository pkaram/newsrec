import numpy as np

def reco_trending_evaluate(timestamp_split, data, recos_method):
    '''
    function to calculate rank metric as to assess trending recommendation system performance
    :param data(dict): users with click behavior
    :param recos(dict): sorted recommendations for user
    :return: rank_metric(float): metric to assess recommender
    '''

    # for each user calculate the rank of the recommendations that have been clicked
    user_impressions_rank = []
    #calculation of all recos so as not to iterate through all users
    recos = recos_method(timestamp_split, data, userid=None)
    for u in data.keys():
        recos_u = recos
        impression_rank = []
        news_clicked = []
        user_news_seen = []
        for t in data[u].keys():
            # check only for the impressions that happened after training
            if timestamp_split < t:
                for i, v in data[u][t]['impressions'].items():
                    if v == 1:
                        news_clicked.append(i)
            else:
                user_news_seen = list(set(user_news_seen) | set(data[u][t]['past_clicks']))

        if len(news_clicked) > 0:
            # filter out user_news_seen
            if len(user_news_seen) > 0:
                for n in user_news_seen:
                    if n in recos_u.keys():
                        recos_u.pop(n)
            # list is already sorted
            recos_l = list(recos_u.keys())
            j = 0
            for n in news_clicked:
                if n in recos_l:
                    impression_rank.append(recos_l.index(n))
                    j += 1
            if j > 0:
                user_impressions_rank.append(np.sum(impression_rank) / j)

    # calculate rank metric
    rank_metric = np.sum(user_impressions_rank) / len(user_impressions_rank)

    return rank_metric

