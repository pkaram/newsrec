from datetime import datetime
import pandas as pd
import os


def prepare_data():
    data = pd.read_csv('datasets/MIND/behaviors.tsv',sep='\t',header=None)
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