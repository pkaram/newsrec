from newsrec.recommenders.recommender_base import RecommenderBase
import scipy.sparse as sparse
from scipy import linalg
import pandas as pd
import numpy as np


class EASE(RecommenderBase):
    """
    Embarrassingly Shallow Autoencoder for implicit feedback.
    Only interactions with rating > 0 are treated as positives.
    Set up in yml file should be following:
    EASE:
     parameters:
      lamb: 500
    """

    def __init__(self):
        self.description = 'Embarrassingly Shallow Autoencoder'
        self.lamb = None

    def pass_parameters(self, model_parameters):
        if not model_parameters:
            model_parameters = {}
        self.lamb = model_parameters.get('lamb', 500)

    def train_model(self, data, k):
        self.data = data
        self.top_k = k
        self.produce_recos()

    def produce_recos(self):
        if self.lamb is None or self.lamb <= 0:
            raise ValueError('EASE lamb must be positive')

        data_train = self.data.data_train
        positives = data_train.loc[data_train['rating'] > 0, ['userid', 'itemid']].copy()
        if positives.empty:
            raise ValueError('EASE needs at least one interaction with rating > 0')

        positives = positives.drop_duplicates(['userid', 'itemid'])
        positives['userid_code'] = positives['userid'].astype('category').cat.codes
        positives['itemid_code'] = positives['itemid'].astype('category').cat.codes
        n_users = int(positives['userid_code'].max()) + 1
        n_items = int(positives['itemid_code'].max()) + 1

        user_item = sparse.csr_matrix(
            (
                np.ones(len(positives), dtype=np.float64),
                (positives['userid_code'], positives['itemid_code']),
            ),
            shape=(n_users, n_items),
        )

        gram = (user_item.T @ user_item).toarray()
        gram.flat[::n_items + 1] += self.lamb
        precision = linalg.inv(gram, overwrite_a=True, check_finite=False)
        weights = -precision / np.diag(precision)
        np.fill_diagonal(weights, 0.0)

        userid_codes = positives[['userid', 'userid_code']].drop_duplicates()
        itemid_codes = positives[['itemid', 'itemid_code']].drop_duplicates()
        test_users = self.data.data_test[['userid']].drop_duplicates()
        test_users = test_users.merge(userid_codes, on='userid', how='inner')

        if test_users.empty:
            self.recos = pd.DataFrame(columns=['userid', 'itemid', 'rank'])
            return

        userids = test_users['userid_code'].to_numpy()
        seen = user_item[userids]
        scores = seen @ weights
        seen_rows = np.repeat(np.arange(seen.shape[0]), np.diff(seen.indptr))
        scores[seen_rows, seen.indices] = -np.inf

        k = min(self.top_k, n_items)
        if k == n_items:
            top_items = np.argsort(-scores, axis=1)
        else:
            candidates = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
            row_idx = np.arange(scores.shape[0])[:, None]
            order = np.argsort(-scores[row_idx, candidates], axis=1)
            top_items = candidates[row_idx, order]

        n_rec = top_items.shape[1]
        recos = pd.DataFrame({
            'userid_code': np.repeat(userids, n_rec),
            'itemid_code': top_items.reshape(-1),
            'score': scores[np.arange(len(userids))[:, None], top_items].reshape(-1),
        })
        recos = recos[np.isfinite(recos['score'])]
        recos = recos.merge(userid_codes, on='userid_code', how='left')
        recos = recos.merge(itemid_codes, on='itemid_code', how='left')
        recos = recos.sort_values(['userid', 'score'], ascending=[True, False])
        recos['rank'] = recos.groupby('userid').cumcount()
        recos = recos.groupby('userid').head(self.top_k)
        recos = recos[['userid', 'itemid', 'rank']]
        recos = recos.reset_index(drop=True)

        self.recos = recos
