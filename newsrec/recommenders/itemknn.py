from newsrec.recommenders.recommender_base import RecommenderBase
import warnings
import scipy.sparse as sparse
import implicit
from implicit.utils import ParameterWarning
import pandas as pd
import numpy as np


class ItemKNN(RecommenderBase):
    """
    Item-item nearest neighbours for implicit feedback.
    Only interactions with rating > 0 are treated as positives.
    Set up in yml file should be following:
    ItemKNN:
     parameters:
      neighbors: 20
      similarity: cosine  # cosine, tfidf, or bm25
      k1: 1.2             # bm25 only
      b: 0.75             # bm25 only
    """

    def __init__(self):
        self.description = 'Item KNN'
        self.neighbors = None
        self.similarity = None
        self.k1 = None
        self.b = None

    def pass_parameters(self, model_parameters):
        if not model_parameters:
            model_parameters = {}
        self.neighbors = model_parameters.get('neighbors', 20)
        self.similarity = model_parameters.get('similarity', 'cosine')
        self.k1 = model_parameters.get('k1', 1.2)
        self.b = model_parameters.get('b', 0.75)

    def train_model(self, data, k):
        self.data = data
        self.top_k = k
        self.produce_recos()

    def _model(self):
        similarity = str(self.similarity).lower()
        if similarity == 'cosine':
            return implicit.nearest_neighbours.CosineRecommender(K=self.neighbors)
        if similarity == 'tfidf':
            return implicit.nearest_neighbours.TFIDFRecommender(K=self.neighbors)
        if similarity == 'bm25':
            return implicit.nearest_neighbours.BM25Recommender(
                K=self.neighbors, K1=self.k1, B=self.b
            )
        raise ValueError("ItemKNN similarity must be 'cosine', 'tfidf', or 'bm25'")

    def produce_recos(self):
        data_train = self.data.data_train
        positives = data_train.loc[data_train['rating'] > 0, ['userid', 'itemid', 'rating']].copy()
        if positives.empty:
            raise ValueError('ItemKNN needs at least one interaction with rating > 0')

        positives = positives.groupby(['userid', 'itemid'], as_index=False)['rating'].max()
        positives['userid_code'] = positives['userid'].astype('category').cat.codes
        positives['itemid_code'] = positives['itemid'].astype('category').cat.codes

        sparse_user_item = sparse.csr_matrix(
            (
                positives['rating'].astype(np.float32),
                (positives['userid_code'], positives['itemid_code']),
            )
        )

        model = self._model()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ParameterWarning)
            model.fit(sparse_user_item, show_progress=False)

        userid_codes = positives[['userid', 'userid_code']].drop_duplicates()
        itemid_codes = positives[['itemid', 'itemid_code']].drop_duplicates()
        test_users = self.data.data_test[['userid']].drop_duplicates()
        test_users = test_users.merge(userid_codes, on='userid', how='inner')

        if test_users.empty:
            self.recos = pd.DataFrame(columns=['userid', 'itemid', 'rank'])
            return

        userids = test_users['userid_code'].to_numpy()
        max_liked = int(positives.groupby('userid_code').size().max())
        recos_raw, scores = model.recommend(
            userids,
            sparse_user_item[userids],
            N=self.top_k + max_liked,
            filter_already_liked_items=True,
        )

        n_rec = recos_raw.shape[1]
        recos = pd.DataFrame({
            'userid_code': np.repeat(userids, n_rec),
            'itemid_code': recos_raw.reshape(-1),
            'score': scores.reshape(-1),
        })
        recos = recos[(recos['itemid_code'] >= 0) & (recos['score'] > 0)]
        recos = recos.merge(userid_codes, on='userid_code', how='left')
        recos = recos.merge(itemid_codes, on='itemid_code', how='left')

        consumed = positives[['userid', 'itemid']].drop_duplicates()
        consumed['consumed'] = 1
        recos = recos.merge(consumed, on=['userid', 'itemid'], how='left')
        recos = recos[recos['consumed'] != 1]

        recos = recos.sort_values(['userid', 'score'], ascending=[True, False])
        recos['rank'] = recos.groupby('userid').cumcount()
        recos = recos.groupby('userid').head(self.top_k)
        recos = recos[['userid', 'itemid', 'rank']]
        recos = recos.reset_index(drop=True)

        self.recos = recos
