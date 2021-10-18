from .recommender_base import RecommenderBase
import scipy.sparse as sparse
import implicit
import pandas as pd
import numpy as np


class iALS(RecommenderBase):
    """
    Class for implicit matrix factorization. Set up in yml file should be following:
    iALS:
     parameters:
      reg: 0.05
      factors: 50
      iter: 20
      epsilon: 1
      log: False #for log transformation
    """

    def __init__(self):
        self.description = 'implicit'
        self.factors = None
        self.reg = None
        self.iter = None
        self.log = False
        self.alpha = None
        self.epsilon = None

    def pass_parameters(self, model_parameters):
        self.factors = model_parameters.get('factors', 100)
        self.reg = model_parameters.get('reg', 0.01)
        self.iter = model_parameters.get('iter', 15)
        self.log = model_parameters.get('log', False)
        self.alpha = model_parameters.get('alpha', 1)
        self.epsilon = model_parameters.get('epsilon', 1)

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

        if self.log:
            data_train['rating'] = [1 + self.alpha * np.log(1+r/self.epsilon) for r in data_train['rating']]
        else:
            data_train['rating'] = data_train['rating'] * self.alpha + 1

        sparse_item_user = sparse.csr_matrix((data_train['rating'].astype(float),
                                              (data_train['itemid_code'], data_train['userid_code'])))
        sparse_user_item = sparse.csr_matrix((data_train['rating'].astype(float),
                                              (data_train['userid_code'], data_train['itemid_code'])))

        data_conf = sparse_item_user.astype('double')

        #define model with parameters provided
        model = implicit.als.AlternatingLeastSquares(factors=self.factors,
                                                     regularization=self.reg,
                                                     iterations=self.iter)
        model.fit(data_conf)

        recos_raw = model.recommend_all(sparse_user_item, filter_already_liked_items=True)
        recos_raw = pd.DataFrame(recos_raw)
        recos = pd.DataFrame()
        for j in range(0,recos_raw.shape[1]):
            reco_col = pd.DataFrame(recos_raw.iloc[:, j])
            reco_col.columns = ['code']
            reco_col['rank'] = j
            recos = pd.concat([recos,reco_col])
        recos = recos.reset_index()
        recos.columns = ['userid_code', 'itemid_code', 'rank']
        recos.itemid_code = recos.itemid_code.astype(int)
        userid_codes = data_train[['userid', 'userid_code']].drop_duplicates()
        itemid_codes = data_train[['itemid', 'itemid_code']].drop_duplicates()
        recos = recos.merge(userid_codes, on='userid_code', how='left')
        recos = recos.merge(itemid_codes, on='itemid_code', how='left')
        recos = recos[['userid','itemid','rank']]

        self.recos = recos
