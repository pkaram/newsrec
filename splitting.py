import numpy as np
import random

def split_data(data,temporal=True,split_ratio=0.2):
    '''
    Split train test data according to parameters
    :param data(dict): Data created based on process_data.behaviors module
    :param temporal(bool): True, if False split will be random
    :param split_ratio(float): values between 0 and 1
    :return: train(dict),test(dict): Splitted dataframes
    '''

    if temporal:
        ts_l = []
        for u in data:
            l = list(data[u].keys())
            for j in l:
                ts_l.append(j)

        ts_l_split = np.quantile(ts_l, 1 - split_ratio)

        train={}
        test={}
        for u in data:
            for t in data[u]:
                if t < ts_l_split:
                    if u not in train:
                        train[u] = {}
                    train[u][t] = data[u][t]
                else:
                    if u not in test:
                        test[u] = {}
                    test[u][t] = data[u][t]

    else:
        train = {}
        test = {}
        random.seed(10)
        for u in data:
            for t in data[u]:
                if split_ratio < random.uniform(0,1):
                    if u not in train:
                        train[u] = {}
                    train[u][t] = data[u][t]
                else:
                    if u not in test:
                        test[u] = {}
                    test[u][t] = data[u][t]

    return train,test
