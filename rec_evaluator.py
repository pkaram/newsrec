import numpy as np

def reco_evaluate(timestamp_split, data, recos_method):
    '''
    function to calculate rank metric as to assess recommendation system performance
    :param data(dict): users with click behavior
    :param recos(dict): sorted recommendations for user
    :return: rank_metric(float): metric to assess recommender
    '''

    # for each user calculate the rank of the recommendations that have been clicked
    user_impressions = {}
    user_impressions_rank = []
    for u in data.keys():
        impression_rank = []
        news_clicked = []
        # calculate the recos according to method selected
        recos = recos_method(timestamp_split, data, userid=u)
        for t in data[u].keys():
            # check only for the impressions that happened after training
            if timestamp_split < t:
                for i, v in user_b[u][t]['impressions'].items():
                    if v == 1:
                        news_clicked.append(i)

        # list is already sorted
        recos_l = list(recos.keys())
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

