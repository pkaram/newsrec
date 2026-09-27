from newsrec.recommenders.recommender_base import RecommenderBase
import scipy.sparse as sparse
import implicit
import pandas as pd
import numpy as np


class BPR(RecommenderBase):
    """
    Bayesian Personalized Ranking for implicit feedback.
    Only interactions with rating > 0 are treated as positives.
    Set up in yml file should be following:
    BPR:
     parameters:
      factors: 50
      learning_rate: 0.01
      reg: 0.01
      iter: 100
      verify_negative_samples: True
      random_state: 42
    """

    def __init__(self):
        self.description = 'Bayesian Personalized Ranking'
        self.factors = None
        self.learning_rate = None
        self.reg = None
        self.iter = None
        self.verify_negative_samples = None
        self.random_state = None

    def pass_parameters(self, model_parameters):
        if not model_parameters:
            model_parameters = {}
        self.factors = model_parameters.get('factors', 100)
        self.learning_rate = model_parameters.get('learning_rate', 0.01)
        self.reg = model_parameters.get('reg', 0.01)
        self.iter = model_parameters.get('iter', 100)
        self.verify_negative_samples = model_parameters.get('verify_negative_samples', True)
        self.random_state = model_parameters.get('random_state', None)

    def train_model(self, data, k):
        self.data = data
        self.top_k = k
        self.produce_recos()

    def produce_recos(self):
        data_train = self.data.data_train
        positives = data_train.loc[data_train['rating'] > 0, ['userid', 'itemid', 'rating']].copy()
        if positives.empty:
            raise ValueError('BPR needs at least one interaction with rating > 0')

        positives['userid_code'] = positives['userid'].astype('category').cat.codes
        positives['itemid_code'] = positives['itemid'].astype('category').cat.codes

        sparse_user_item = sparse.csr_matrix(
            (
                positives['rating'].astype(np.float32),
                (positives['userid_code'], positives['itemid_code']),
            )
        )

        model = implicit.bpr.BayesianPersonalizedRanking(
            factors=self.factors,
            learning_rate=self.learning_rate,
            regularization=self.reg,
            iterations=self.iter,
            verify_negative_samples=self.verify_negative_samples,
            random_state=self.random_state,
            use_gpu=False,
        )
        model.fit(sparse_user_item, show_progress=False)

        userid_codes = positives[['userid', 'userid_code']].drop_duplicates()
        itemid_codes = positives[['itemid', 'itemid_code']].drop_duplicates()
        test_users = self.data.data_test[['userid']].drop_duplicates()
        test_users = test_users.merge(userid_codes, on='userid', how='inner')

        if test_users.empty:
            self.recos = pd.DataFrame(columns=['userid', 'itemid', 'rank'])
            return

        userids = test_users['userid_code'].to_numpy()
        recos_raw, _ = model.recommend(
            userids,
            sparse_user_item[userids],
            N=self.top_k,
            filter_already_liked_items=True,
        )

        n_rec = recos_raw.shape[1]
        recos = pd.DataFrame({
            'userid_code': np.repeat(userids, n_rec),
            'itemid_code': recos_raw.reshape(-1),
            'rank': np.tile(np.arange(n_rec), len(userids)),
        })
        recos = recos[recos['itemid_code'] >= 0]
        recos = recos.merge(userid_codes, on='userid_code', how='left')
        recos = recos.merge(itemid_codes, on='itemid_code', how='left')
        recos = recos[['userid', 'itemid', 'rank']]
        recos = recos.sort_values(['userid', 'rank']).reset_index(drop=True)

        self.recos = recos
