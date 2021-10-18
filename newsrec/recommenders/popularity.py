from .recommender_base import RecommenderBase


class Popular(RecommenderBase):
    """
    Class for calculating most popular recommendations for each user. Set up in yml file should be following:
    Popular:
     parameters:
    """
    def __init__(self):
        self.description = 'Popular'

    def pass_parameters(self, model_parameters):
        pass

    def train_model(self, data, k):
        self.data = data
        self.top_k = k
        self.produce_recos()

    def produce_recos(self):
        data_test = self.data.data_test
        data_train = self.data.data_train

        max_items_reviewed = data_train[['userid','itemid']].drop_duplicates()
        max_items_reviewed = max_items_reviewed.userid.value_counts()[0]

        top_items = data_train.groupby('itemid')['userid'].count().reset_index()
        top_items.columns = ['itemid', 'cnt']
        top_items = top_items.sort_values(by='cnt', ascending=False)
        top_items = top_items.reset_index(drop=True)
        top_items['dummy_col'] = 1
        top_items = top_items.head(self.top_k + max_items_reviewed)

        items_consumed = data_train[['userid','itemid']].drop_duplicates()
        items_consumed['consumed'] = 1

        recos = data_test[['userid']].drop_duplicates()
        recos['dummy_col'] = 1
        recos = recos.merge(top_items, on=['dummy_col'], how='left')
        recos = recos.drop(columns=['dummy_col'])
        recos = recos.merge(items_consumed, on=['userid','itemid'], how='left')
        recos = recos[recos['consumed'] != 1]
        recos = recos.sort_values(['userid', 'cnt'], ascending=(False, False))
        recos = recos.drop(columns=['consumed'])
        recos['rank'] = recos.groupby('userid').cumcount()

        recos = recos.groupby('userid').head(10)
        recos = recos.drop(columns=['cnt'])
        recos = recos.reset_index(drop=True)

        self.recos = recos