import numpy as np


class Splitter:
    def __init__(self, data, user_features=None, item_features=None, temporal=True, split_per=0.2):
        self.data = data
        self.temporal = temporal
        self.split_per = split_per
        self.data_train = None
        self.data_test = None
        self.user_features = user_features
        self.item_features = item_features
        self.validate_datasets()
        self.split_data()

    def split_data(self):
        data = self.data
        data = data[['userid','itemid','rating','timestamp']]

        if self.temporal:
            data = data.sort_values(by=['timestamp'])
            self.data_train, self.data_test = np.split(data, [int((1-self.split_per)*len(data))])
        else:
            self.data_train, self.data_test = np.split(data, [int((1-self.split_per)*len(data))])

    def validate_datasets(self):
        col_intersection = list(set(['userid','itemid','rating','timestamp']) & set(list(self.data.columns)))
        if len(col_intersection) < 4:
            raise ValueError('''Following column names should exist in user_item_ratings
            :userid,itemid,rating,timestamp. Rename your df accordingly''')
        if self.user_features is not None:
            if 'userid' not in self.user_features.columns:
                raise ValueError('userid should exist as a column in user_features')
        if self.item_features is not None:
            if 'itemid' not in self.item_features.columns:
                raise ValueError('itemid should exist as a column in item_features')
