from .recommender_base import RecommenderBase
from scipy.sparse.linalg import svds
import numpy as np
import pandas as pd


class SVD(RecommenderBase):
    """
    Class for Singular Value Decomposition. Set up in yml file should be following:
    SVD:
     parameters:
      singular_values_n: 10
    """

    def __init__(self):
        self.description = 'matrix factorization'
        self.singular_values_n = None

    def pass_parameters(self, model_parameters):
        self.singular_values_n = model_parameters.get('singular_values_n', 20)

    def train_model(self, data, k):
        self.data = data
        self.top_k = k
        self.produce_recos()

    def produce_recos(self):
        data_train = self.data.data_train
        data_train['userid'] = data_train['userid'].astype("category")
        data_train['itemid'] = data_train['itemid'].astype("category")
        data_train['userid_code'] = data_train['userid'].cat.codes
        data_train['itemid_code'] = data_train['itemid'].cat.codes

        R_df = data_train.pivot_table(index='userid_code', columns='itemid_code', values='rating',
                                      aggfunc='mean').fillna(0)
        R = R_df.to_numpy()
        user_ratings_mean = np.mean(R, axis=1)
        R_minus_mean = R - user_ratings_mean.reshape(-1,1)

        U, sigma, Vt = svds(R_minus_mean, k=self.singular_values_n)
        sigma = np.diag(sigma)
        pred_matrix = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

        pred_df = pd.DataFrame(pred_matrix, columns=R_df.columns)
        pred_df = pred_df.stack()
        pred_df = pred_df.reset_index()
        pred_df.columns = ['userid_code','itemid_code','rating_pred']

        data_test = self.data.data_test
        data_test = data_test[['userid']].drop_duplicates()
        df_codes = data_train[['userid','userid_code']].drop_duplicates()
        data_test = pd.merge(data_test, df_codes, on=['userid'], how='inner')
        data_test = pd.merge(data_test, pred_df, on=['userid_code'], how='inner')
        df_codes = data_train[['itemid','itemid_code']].drop_duplicates()
        data_test = pd.merge(data_test, df_codes, on=['itemid_code'], how='left')

        max_items_reviewed = data_train[['userid','itemid']].drop_duplicates()
        max_items_reviewed = max_items_reviewed.userid.value_counts()[0]
        data_test = data_test.sort_values(['userid', 'rating_pred'], ascending=[False, False])
        data_test['rank_index'] = data_test.groupby(['userid']).cumcount()
        data_test = data_test.groupby(['userid_code']).head(self.top_k + max_items_reviewed)
        data_train = data_train[['userid', 'itemid']]
        data_train['item_consumed'] = 1
        data_test = pd.merge(data_test, data_train, on=['userid', 'itemid'], how='left')
        data_test = data_test[data_test['item_consumed'] != 1]

        data_test = data_test.sort_values(['userid', 'rank_index'], ascending=[False, True])
        data_test['rank'] = data_test.groupby(['userid']).cumcount()
        data_test = data_test.groupby('userid').head(self.top_k)
        data_test = data_test[['userid','itemid','rank']]
        data_test = data_test.reset_index(drop=True)

        self.recos = data_test
