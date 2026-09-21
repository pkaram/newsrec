from datetime import datetime
import os
import urllib.request
import zipfile

import pandas as pd

MIND_DIR = 'datasets/MIND'
BEHAVIORS_PATH = os.path.join(MIND_DIR, 'behaviors.tsv')
MIND_SMALL_TRAIN_URL = 'https://huggingface.co/datasets/Recommenders/MIND/resolve/main/MINDsmall_train.zip'


def ensure_behaviors():
    if os.path.exists(BEHAVIORS_PATH):
        return BEHAVIORS_PATH

    if os.path.isdir(MIND_DIR):
        for root, _, files in os.walk(MIND_DIR):
            if 'behaviors.tsv' in files:
                return os.path.join(root, 'behaviors.tsv')

    os.makedirs(MIND_DIR, exist_ok=True)
    zip_path = os.path.join(MIND_DIR, 'MINDsmall_train.zip')
    print(f'Downloading MIND-small train to {zip_path}')
    urllib.request.urlretrieve(MIND_SMALL_TRAIN_URL, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        member = next(name for name in zf.namelist() if os.path.basename(name) == 'behaviors.tsv')
        zf.extract(member, MIND_DIR)
        extracted = os.path.join(MIND_DIR, member)
        if extracted != BEHAVIORS_PATH:
            os.replace(extracted, BEHAVIORS_PATH)

    print(f'Saved {BEHAVIORS_PATH}')
    return BEHAVIORS_PATH


def prepare_data():
    behaviors_path = ensure_behaviors()
    data = pd.read_csv(behaviors_path, sep='\t', header=None)
    data.columns = ['index','userid','timestamp','past_history','clicks_impressions']
    data = data[['userid','clicks_impressions','timestamp']]
    data.timestamp = [datetime.strptime(s, '%m/%d/%Y %H:%M:%S %p') for s in data.timestamp]
    data.timestamp = [int(s.timestamp()) for s in data.timestamp]
    data["clicks_impressions"] = data["clicks_impressions"].str.split(" ")
    data = data.explode("clicks_impressions").reset_index(drop=True)
    data[['itemid','rating']] = data['clicks_impressions'].str.split('-',expand=True)
    data = data.drop(columns=['clicks_impressions'])
    data.rating = data.rating.astype(float)
    if not os.path.exists('datasets/MIND/data'):
        os.makedirs('datasets/MIND/data')
    data.to_csv('datasets/MIND/data/ratings.csv',index=False)

if __name__ == '__main__':
    prepare_data()