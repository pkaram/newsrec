import numpy as np


class Evaluator:
    def __init__(self, model):
        self.recos = model.recos
        self.data_train = model.data.data_train
        self.data_test = model.data.data_test
        self.item_coverage = None
        self.user_coverage = None
        self.precision = None
        self.map = None
        self.ndcg = None

    def calculate_metrics(self):
        ##there should be some explanation on the values and the logic
        self.calc_item_coverage()
        self.calc_user_coverage()
        self.calc_precision()
        self.calc_map()
        self.calc_ndcg()

    def metrics(self):
        return {
            'item_coverage': self.item_coverage,
            'user_coverage': self.user_coverage,
            'precision': self.precision,
            'map': self.map,
            'ndcg': self.ndcg
        }

    def calc_item_coverage(self):
        """
        items that are covered in the recos based on the available
        items on train dataset
        """
        itemid_rec_n = self.recos['itemid']
        itemid_n = self.data_train['itemid']
        self.item_coverage = len(itemid_rec_n.unique()) / len(itemid_n.unique())

    def calc_user_coverage(self):
        """
        users that are covered from the calculated recommendations
        """
        users_in_recs = set(self.recos['userid'])
        users_test = set(self.data_test['userid'])
        users_common = list(users_in_recs & users_test)
        self.user_coverage = len(users_common) / len(users_test)

    def calc_precision(self):
        """
        Precision among recommendations provided
        """
        recs_available = self.recos
        recs_interacted = self.data_test[['userid','itemid']]
        recs_interacted['interaction'] = 1
        recs_available = recs_available.merge(recs_interacted, on=['userid','itemid'], how='left')
        recs_available = recs_available.groupby(['userid']).agg({'interaction':'sum',
                                                                 'itemid':'count'})
        recs_available = recs_available.reset_index()
        recs_available['Precision'] = recs_available['interaction'] / recs_available['itemid']
        self.precision = np.sum(recs_available['Precision']) / recs_available.shape[0]

    def calc_map(self):
        """
        Mean Average Precision
        """
        recs_available = self.recos
        recs_interacted = self.data_test[['userid','itemid']]
        recs_interacted['interaction'] = 1
        recs_available = recs_available.merge(recs_interacted, on=['userid', 'itemid'], how='left')
        recs_available['interaction'] = recs_available['interaction'].fillna(0)
        recs_available['interaction_cs'] = recs_available.groupby('userid')['interaction'].cumsum()

        def helper_f(a, b):
            if b != 0:
                return b / (a + 1)
            else:
                return 0

        recs_available['AP'] = recs_available.apply(lambda x: helper_f(x['rank'], x['interaction_cs']), axis=1)
        recs_available = recs_available.groupby(['userid']).agg({'AP':'sum',
                                                                 'itemid':'count'})
        recs_available = recs_available.reset_index()
        recs_available['AP'] = recs_available['AP'] / recs_available['itemid']
        self.map = np.sum(recs_available['AP']) / recs_available.shape[0]

    def calc_ndcg(self):
        """
        Normalized Discounted Cumulative Gain.
        A hit at rank r contributes 1 / log2(r + 2). Rank 0 is the first
        position, so its discount is 1. Each user is divided by the gain
        of placing that user's test items in the best possible order.
        """
        recs_available = self.recos.sort_values(['userid', 'rank'])
        if recs_available.empty:
            self.ndcg = 0.0
            return

        recs_interacted = self.data_test[['userid', 'itemid']].drop_duplicates()
        recs_interacted['interaction'] = 1
        recs_available = recs_available.merge(recs_interacted, on=['userid', 'itemid'], how='left')
        recs_available['interaction'] = recs_available['interaction'].fillna(0)
        recs_available['gain'] = recs_available['interaction'] / np.log2(recs_available['rank'] + 2)

        per_user = recs_available.groupby('userid').agg(dcg=('gain', 'sum'), k=('itemid', 'count'))
        per_user = per_user.reset_index()
        n_relevant = self.data_test[['userid', 'itemid']].drop_duplicates()
        n_relevant = n_relevant.groupby('userid').size().rename('n_relevant')
        per_user = per_user.merge(n_relevant, on='userid', how='left')
        per_user['n_relevant'] = per_user['n_relevant'].fillna(0)

        max_k = int(per_user['k'].max())
        # ideal_at_n[n - 1] is the gain of n hits in the first n positions
        ideal_at_n = np.cumsum(1.0 / np.log2(np.arange(1, max_k + 1) + 1))
        n_ideal = np.minimum(per_user['n_relevant'].to_numpy(), per_user['k'].to_numpy()).astype(int)
        idcg = np.zeros(len(n_ideal))
        has_relevant = n_ideal > 0
        idcg[has_relevant] = ideal_at_n[n_ideal[has_relevant] - 1]
        per_user['ndcg'] = np.where(idcg > 0, per_user['dcg'] / idcg, 0.0)
        self.ndcg = np.sum(per_user['ndcg']) / per_user.shape[0]
