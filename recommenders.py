def trending_recos(timestamp_split,data,userid):
    '''
    Provide top trending news recommendation for a user on a certain timestamp
    :param timestamp (timestamp): the timestamp on which we ask recommendations
    :param data (dict): user data on past clicks
    :return: recos(dict): sorted dict with newsids and number of users which have consumed it
    '''

    # create a list to store the users each news element has
    news_clicks ={}
    user_news_seen ={}
    for k in data.keys():
        for t in data[k].keys():
            if t < timestamp_split:
                for n in data[k][t]['past_clicks']:
                    if n not in news_clicks.keys():
                        news_clicks[n ] =[]
                    news_clicks[n].append(k)
                    if k not in user_news_seen.keys():
                        user_news_seen[k ] =[]
                    user_news_seen[k].append(n)

    # calculate the unique users per news element and give back the sorted list of news by number of users
    # which will be the base for the final recos
    recos ={}
    for n in news_clicks.keys():
        news_clicks[n ] =list(set(news_clicks[n]))
        recos[n] =len(news_clicks[n])

    for u in user_news_seen.keys():
        user_news_seen[u] =list(set(user_news_seen[u]))

    # sort the dictionary which will provide the recommendations
    recos ={k: v for k, v in sorted(recos.items(), key=lambda item: item[1] ,reverse=True)}

    # filter out all news that a user has already consumed if he has already history
    if userid in user_news_seen.keys():
        for n in user_news_seen[userid]:
            recos.pop(n)

    return recos


def implicit_recos():
    '''
    Provide implicit feedback recommendations


    :return:
    '''


    return recos





