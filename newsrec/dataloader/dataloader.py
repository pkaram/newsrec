import yaml
import sys
import pandas as pd


class DataLoader:
    def __init__(self, config_path):
        self.path = get_config(config_path)
        self.user_item_ratings_path = None
        self.user_features_path = None
        self.item_features_path = None
        self.user_item_info = None
        self.user_features_info = None
        self.item_features_info = None
        self.top_k = None
        self.split_per = None
        self.temporal = None
        self.get_params()
        self.data_paths()
        self.loader()

    def get_params(self):
        self.top_k = self.path.get('top_k', 10)
        self.split_per = self.path.get('split_per', 0.2)
        self.temporal = self.path.get('temporal', True)

    def data_paths(self):
        self.user_item_ratings_path = self.path.get('datapaths').get('user_item_ratings', None)
        self.user_features_path = self.path.get('datapaths').get('user_features', None)
        self.item_features_path = self.path.get('datapaths').get('item_features', None)

        if not self.user_item_ratings_path:
            sys.exit('Path for user_item_ratings should be provided in all cases')

    def loader(self):
        self.user_item_info = pd.read_csv(self.user_item_ratings_path)

        if self.user_features_path:
            self.user_features_info = pd.read_csv(self.user_features_path)
        if self.item_features_path:
            self.item_features_info = pd.read_csv(self.item_features_path)


def get_config(path):
    try:
        with open(path) as f:
            config_dict = yaml.safe_load(f)
            return config_dict
    except Exception as e:
        print(f'Error:{e}')
