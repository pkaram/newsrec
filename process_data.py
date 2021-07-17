from datetime import datetime
import ast

def behaviors(path=None):
    '''
    :param path(string): path to file
    :return: user_b(dict): information on user_behavior
    '''

    #define path of file
    if path:
        file = open(path, 'r')
    else:
        file = open('archive/behaviors.tsv','r')

    #define dictionary where information will be saved
    user_b = {}
    for line in file:
        el=line.replace('\n','')
        el=el.split('\t')[1:]
        date_dt=datetime.strptime(el[1],'%m/%d/%Y %H:%M:%S %p')
        date_dt=int(date_dt.timestamp())
        if el[0] not in user_b.keys():
            user_b[el[0]]={}
        user_b[el[0]][date_dt]={'past_clicks':el[2].split(' ')}
        user_b[el[0]][date_dt]['impressions'] = {}
        click_impression=el[3].split(' ')
        for impression in click_impression:
            user_b[el[0]][date_dt]['impressions'][impression.split('-')[0]] = int(impression.split('-')[1])
    file.close()

    return user_b


def news(path=None):
    '''

    :param path(string): path to file
    :return: news_info(dict): information on news
    '''

    #define path of file
    if path:
        file = open(path, 'r')
    else:
        file = open('archive/news.tsv','r')

    news_info={}
    for line in file:
        el = line.split('\t')
        news_info[el[0]]={}
        news_info[el[0]]['category']=el[1]
        news_info[el[0]]['subcategory']=el[2]
        news_info[el[0]]['title']=el[3]
        news_info[el[0]]['abstract']=el[4]
        news_info[el[0]]['url']=el[5]
        news_info[el[0]]['title_entities']=ast.literal_eval(el[6])
        news_info[el[0]]['abstract_entities']=ast.literal_eval(el[7])
    file.close()

    return news_info
