
class Recommender:
    def __init__(self, method, data):
        self.model = None
        self.recos = None
        self.method = method
        if self.method == 'trending':
            self.recos = self.trending_recos(data)

    @classmethod
    def trending_recos(Recommender, data):
        '''
        Provide top trending news recommendation for a user
        :params
            data (dict): user data on past clicks
        :return:
            recos(dict): sorted dict with newsids and number of users which have consumed it
        '''

        # create a list to store the users each news element has
        news_clicks ={}
        user_news_seen ={}
        for k in data.keys():
            for t in data[k].keys():
                for n in data[k][t]['past_clicks']:
                    if n not in news_clicks.keys():
                        news_clicks[n] =[]
                    news_clicks[n].append(k)
                    if k not in user_news_seen.keys():
                        user_news_seen[k] =[]
                    user_news_seen[k].append(n)

        # calculate the unique users per news element and give back the sorted list of news by number of users
        # which will be the base for the final recos
        recs ={}
        for n in news_clicks.keys():
            news_clicks[n] =list(set(news_clicks[n]))
            recs[n] =len(news_clicks[n])

        for u in user_news_seen.keys():
            user_news_seen[u] =list(set(user_news_seen[u]))

        # sort the dictionary which will provide the recommendations
        recs ={k: v for k, v in sorted(recs.items(), key=lambda item: item[1], reverse=True)}

        return recs
