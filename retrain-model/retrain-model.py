
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import os
import io
import json

import pandas as pd
import numpy as np

import lightgbm as lgb

from sklearn.model_selection import cross_val_score, train_test_split, ParameterGrid
from sklearn.metrics import median_absolute_error

import itertools

from tqdm import tqdm
import pickle

import mlflow
import boto3
import psycopg2

from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt 
from IPython import get_ipython


# In[3]:


pd.set_option('display.max_columns', None)


# In[4]:


with open('/home/dmitry-ds/postgresql-credentials/creds.json', 'r') as f:
    creds = json.load(f)


# In[5]:


session = boto3.session.Session()
s3 = session.client(service_name = 's3', endpoint_url = 'https://storage.yandexcloud.net', region_name = 'ru-central1')


# In[6]:


conn = psycopg2.connect(f"""host=c-c9qm2f2d6d1lkiqst4qn.rw.mdb.yandexcloud.net
port=6432
sslmode=disable
dbname=anime-rec-sys-db
user={creds['user']}
password={creds['password']}
target_session_attrs=read-write
                        """)


# In[7]:


def test_user_ids(s3):
    
    get_object_response = s3.get_object(Bucket = 'anime-rec-sys-data', Key = 'user_id_test.npy')
    user_id_test = np.load(io.BytesIO(get_object_response['Body'].read()))
    
    return user_id_test


# In[8]:


def get_item_data(s3):
    
    get_object_response = s3.get_object(Bucket = 'anime-rec-sys-data', Key = 'anime_selected.csv')
    anime_selected = pd.read_csv(io.BytesIO(get_object_response['Body'].read()))
    
    get_object_response = s3.get_object(Bucket = 'anime-rec-sys-data', Key = 'anime_synopsis_emb_2.csv')
    anime_synopsis_emb = pd.read_csv(io.BytesIO(get_object_response['Body'].read()))
    
    get_object_response = s3.get_object(Bucket = 'anime-rec-sys-data', Key = 'lightfm_emb_df_50e.csv')
    lightfm_emb_df = pd.read_csv(io.BytesIO(get_object_response['Body'].read()))
    
    return anime_selected, anime_synopsis_emb, lightfm_emb_df


# In[9]:


def get_user_data(connection):
    
    sql_query = pd.read_sql_query("SELECT * FROM selectedratings;", connection)
    rating_selected = pd.DataFrame(sql_query, columns = ['user_id', 'anime_id', 'rating'])
    
    sql_query = pd.read_sql_query("SELECT * FROM userpreferences;", connection)
    users = pd.DataFrame(sql_query)
    users.columns = ['user_id', 'Action', 'Adventure', 'Cars', 'Comedy', 'Dementia',
                     'Demons', 'Drama', 'Ecchi', 'Fantasy', 'Game', 'Harem', 'Hentai',
                     'Historical', 'Horror', 'Josei', 'Kids', 'Magic', 'Martial Arts',
                     'Mecha', 'Military', 'Music', 'Mystery', 'Parody', 'Police',
                     'Psychological', 'Romance', 'Samurai', 'School', 'Sci-Fi', 'Seinen',
                     'Shoujo', 'Shoujo Ai', 'Shounen', 'Shounen Ai', 'Slice of Life',
                     'Space', 'Sports', 'Super Power', 'Supernatural', 'Thriller', 'Unknown',
                     'Vampire', 'Yaoi', 'Yuri', 'mean_rating']
    
    connection.close()
    
    return rating_selected, users


# In[10]:


def create_lightgbm_dataset(rating_selected, users, lightfm_emb_df, anime_synopsis_emb):
    
    lgb_dataset = rating_selected.copy()
    lgb_dataset = lgb_dataset.merge(users[users.columns], on = 'user_id', how = 'left')
    lgb_dataset = lgb_dataset.merge(lightfm_emb_df[lightfm_emb_df.columns], on = 'anime_id', how = 'left')
    lgb_dataset = lgb_dataset.merge(anime_synopsis_emb[anime_synopsis_emb.columns], on = 'anime_id', how = 'left')
    
    return lgb_dataset


# In[11]:


def create_train_test_val(lgb_dataset):
    
    user_id_test = test_user_ids(s3)
    user_id_train = lgb_dataset['user_id'].unique()[~np.isin(lgb_dataset['user_id'].unique(), user_id_test)]
    user_id_train, user_id_val = train_test_split(user_id_train, test_size = 0.15, random_state = 586) 
    
    print(f'Num train {len(user_id_train)}')
    print(f'Num val {len(user_id_val)}')
    print(f'Num test {len(user_id_test)}')
    
    train = lgb_dataset[lgb_dataset['user_id'].isin(user_id_train)]
    val = lgb_dataset[lgb_dataset['user_id'].isin(user_id_val)]
    test = lgb_dataset[lgb_dataset['user_id'].isin(user_id_test)]
    
    return train, val, test


# In[12]:


def train_new_model(train, val):
    
    lgb_train = lgb.Dataset(train.iloc[:, 3:], train['rating'])
    lgb_val = lgb.Dataset(val.iloc[:, 3:], val['rating'])
    
    params = {'objective': 'regression',
          'max_depth': 8,
          'n_estimators': 2000,
          'num_leaves': 2**8-1,
          'learning_rate': 0.01,
          'colsample_bytree': 0.8,
          'subsample': 0.8,
          'early_stopping_rounds': 20,
          'random_state': 42,
          'n_jobs': 8
         }
    
    new_model = lgb.train(params,
                      lgb_train,
                      valid_sets = lgb_val,
                      verbose_eval = 10
                     )
    
    return new_model


# In[13]:


def get_prod_model(model_name):
    prod_model = mlflow.lightgbm.load_model(f'models:/{model_name}/Production')
    return prod_model


# In[15]:


def validate_and_log(test, new_model, prod_model, bootstrap_iterations = 100):
    
    mlflow.start_run()
    
    mlflow.lightgbm.log_model(new_model, 'LightGBM-regressor')
    
    new_model_preds = new_model.predict(test.iloc[:, 3:])
    new_model_mae = median_absolute_error(test['rating'], new_model_preds)
    mlflow.log_metric('test MAE', new_model_mae)
    
    new_scores = pd.DataFrame(data = {'test MAE': 0.0}, index=range(bootstrap_iterations))
    test['predicted_rating'] = new_model_preds
    new_model_results_df = test[['rating', 'predicted_rating']]
    
    prod_model_preds = prod_model.predict(test.iloc[:, 3:])
    prod_scores = pd.DataFrame(data = {'test MAE': 0.0}, index=range(bootstrap_iterations))
    test.drop('predicted_rating', axis = 1, inplace = True)
    test['predicted_rating'] = prod_model_preds
    prod_model_results_df = test[['rating', 'predicted_rating']]
    
    for i in range(bootstrap_iterations):
        new_model_sample = new_model_results_df.sample(frac=1.0, replace=True)
        prod_model_sample = prod_model_results_df.sample(frac=1.0, replace=True)
        
        new_scores.loc[i, 'test MAE'] = median_absolute_error(new_model_sample['rating'], new_model_sample['predicted_rating'])
        prod_scores.loc[i, 'test MAE'] = median_absolute_error(prod_model_sample['rating'], prod_model_sample['predicted_rating'])
    
    ax = sns.histplot(x = new_scores['test MAE'], alpha = 0.3)
    sns_plot = sns.histplot(x = prod_scores['test MAE'], alpha = 0.3, color = 'orange', ax = ax)
    plt.close()
    
    mlflow.log_figure(sns_plot.figure, 'distribution_plot.png')
    
    new_score_mean = new_scores['test MAE'].mean()
    new_score_std = new_scores['test MAE'].std()
    mlflow.log_metric('boostrap MAE', new_score_mean)
    mlflow.log_metric('boostrap std', new_score_std)
    
    users.to_csv('user-preferences.csv', index = False)
    rating_selected.to_csv('rating_selected.csv', index = False)
    mlflow.log_artifact('user-preferences.csv')
    mlflow.log_artifact('rating_selected.csv')  
    os.remove('user-preferences.csv')
    os.remove('rating_selected.csv')
    
    prod_score_mean = prod_scores['test MAE'].mean()
    prod_score_std = prod_scores['test MAE'].std()
    
    #p-test
    alpha = 0.05    
    pvalue = ttest_ind(prod_scores['test MAE'], new_scores['test MAE']).pvalue
    mlflow.log_metric('p-value', pvalue)
    
    mlflow.end_run()
    
    mlflow_run = mlflow.search_runs().iloc[0]
    model_name = 'LightGBM-ratings_predictor'
    new_model_version = mlflow.register_model(f'runs:/{mlflow_run.run_id}/LightGBM-regressor', model_name)
    
    if pvalue < alpha:
        print('Reject null hypothesis')
        if prod_score_mean < new_score_mean:
            print('Keep current prod model')
        else:
            print('Serve new model')
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(name = model_name,
                                                  version = new_model_version.version,
                                                  stage = "Production"
                                                 )
            
#             users.to_csv('/home/dmitry-ds/rec-sys/Anime-recommender-engine/app/data/user-preferences.csv', index = False)
#             rating_selected.to_csv('/home/dmitry-ds/rec-sys/Anime-recommender-engine/app/data/rating_selected.csv', index = False)
#             with open('/home/dmitry-ds/rec-sys/Anime-recommender-engine/app/models/LightGBM-v5.pickle', 'wb') as file:
#                 pickle.dump(new_model, file)        
            
    else:
        print('Accept null hypothesis')
        print('Serve new model because it knows more users')
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(name = model_name,
                                              version = new_model_version.version,
                                              stage = "Production"
                                             )
        
#         users.to_csv('/home/dmitry-ds/rec-sys/Anime-recommender-engine/app/data/user-preferences.csv', index = False)
#         rating_selected.to_csv('/home/dmitry-ds/rec-sys/Anime-recommender-engine/app/data/rating_selected.csv', index = False)
#         with open('/home/dmitry-ds/rec-sys/Anime-recommender-engine/app/models/LightGBM-v5.pickle', 'wb') as file:
#             pickle.dump(new_model, file)    


# In[16]:


anime_selected, anime_synopsis_emb, lightfm_emb_df = get_item_data(s3)


# In[17]:


rating_selected, users = get_user_data(conn)


# In[21]:


lgb_dataset = create_lightgbm_dataset(rating_selected, users, lightfm_emb_df, anime_synopsis_emb)


# In[22]:


train, val, test = create_train_test_val(lgb_dataset)


# In[23]:


new_model = train_new_model(train, val)


# In[30]:


prod_model = get_prod_model('LightGBM-ratings_predictor')


# In[35]:


validate_and_log(test, new_model, prod_model)

