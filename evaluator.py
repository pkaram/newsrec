import numpy as np

def evaluate_rank(data, model):
    '''
    function to calculate rank metric to assess recommendation system performance
    :param test(dict):
    :param model(dict): recos that have been provided
    :return:
    '''

    # for each user calculate the rank of the recommendations that have been clicked
    user_impressions_rank = []
    #check if recos are provided from the model
    if model.recos:
        recos_u = model.recos
        for u in data:
            impression_rank = []
            news_clicked = []
            for t in data[u].keys():
                for i, v in data[u][t]['impressions'].items():
                    if v == 1:
                        news_clicked.append(i)

            if len(news_clicked) > 0:
                # list is already sorted
                recos_l = list(recos_u.keys())
                for n in news_clicked:
                    if n in recos_l:
                        impression_rank.append(recos_l.index(n)/len(recos_l))
                user_impressions_rank.append(np.mean(impression_rank))

    return np.nanmean(user_impressions_rank)


def evaluate_coverage(data, model, k=100):
    '''
    function to calculate coverage metric to assess recommendation system performance
    :param test(dict):
    :param model(dict): recos that have been provided
    :param k(int): top k recommendations to be selected
    :return:
    '''

    # for each user calculate the rank of the recommendations that have been clicked
    #check if recos are provided from the model
    coverage = []
    if model.recos:
        # list is already sorted
        recos_u = list(model.recos)[0:k]
        for u in data:
            news_clicked = 0
            user_news_seen = 0
            for t in data[u].keys():
                for i, v in data[u][t]['impressions'].items():
                    if v == 1 and i in recos_u:
                        news_clicked += 1
                    user_news_seen += 1
            coverage.append(news_clicked/user_news_seen)

    return np.mean(coverage)