import json
import random
from tqdm import tqdm
import itertools
from datetime import datetime
import pandas as pd
import numpy as np
import psycopg2
import psycopg2.extras as extras

with open('/home/dmitry-ds/postgresql-credentials/creds.json', 'r') as f:
    creds = json.load(f)

conn = psycopg2.connect(f"""
    host=c-c9qm2f2d6d1lkiqst4qn.rw.mdb.yandexcloud.net
    port=6432
    sslmode=disable
    dbname=anime-rec-sys-db
    user={creds['user']}
    password={creds['password']}
    target_session_attrs=read-write
""")

q = conn.cursor()
    
#function to insert pd dataframe into posgresql db
def execute_values(conn, df, table):
    
    df = df.astype(float)
    
    tuples = [tuple(x) for x in df.to_numpy()]
  
    cols = ','.join(list(df.columns))
  
    # SQL query to execute
    query = "INSERT INTO %s(%s) VALUES %%s" % (table, cols)
    cursor = conn.cursor()
    try:
        extras.execute_values(cursor, query, tuples)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    print("execute_values() done")
    cursor.close()

#function to generate user features
def create_users(rating_selected, total_genres):

    user_data = []

    for user_id in tqdm(rating_selected['user_id'].unique()):
        filter_user = rating_selected[rating_selected['user_id'] == user_id]
        total_anime_watch = len(filter_user)
        genres_user = filter_user['Genres']

        if isinstance(genres_user, str):
            genres_user = [genres_user]
        else:
            genres_user = genres_user.values

        user_gen_dat = {}

        all_genres = [x.strip() for x in (','.join(map(lambda x: str(x), genres_user))).split(',')]

        for unique_gen in total_genres:
            user_gen_dat[unique_gen] = all_genres.count(unique_gen)/total_anime_watch
        
        user_gen_dat['mean_rating'] = filter_user['rating'].sum()/total_anime_watch
        
        user_data.append({user_id: user_gen_dat})
        
    users = pd.DataFrame(user_data).stack().apply(pd.Series).reset_index().rename(columns = {'level_1': 'user_id'}).drop('level_0', axis = 1)
    
    return users

#function to update selectedratings and userpreferences tables
def fill_tables():
    
    sql_query = pd.read_sql_query ("SELECT * FROM futureusers WHERE user_id >\
    (SELECT user_id FROM selectedratings ORDER BY user_id DESC LIMIT 1)", conn)
    
    sel = pd.DataFrame(sql_query, columns = ['user_id', 'anime_id', 'rating'])
    
    n_new = random.randint(10, 25)
    user_range = sel.drop_duplicates('user_id').iloc[:n_new, 0].values
    sel_slice = sel[sel['user_id'].isin(user_range)]
    execute_values(conn, sel_slice, 'selectedratings')
    
    todays_date = datetime.today().strftime("%Y-%m-%d")
    
    print(f'Added {n_new} new users on {todays_date}')
    
    anime_selected = pd.read_csv('/home/dmitry-ds/rec-sys/Anime-recommender-engine/app/data/anime_selected.csv')
    
    total_genres = sorted(set([i.strip() for i in itertools.chain.from_iterable([i.split(',') for i in anime_selected['Genres'].values])]))
    
    sel_slice_genres = sel_slice.merge(anime_selected[['anime_id', 'Genres']], on = 'anime_id')
    user_preferences = create_users(sel_slice_genres, total_genres)
    
    user_preferences.rename({'Martial Arts': 'Martial_Arts', 
                         'Sci-Fi': 'Sci_Fi', 
                         'Shoujo Ai': 'Shoujo_Ai',
                        'Shounen Ai': 'Shounen_Ai', 
                        'Slice of Life': 'Slice_of_Life',
                        'Super Power': 'Super_Power'
                        }, axis = 1, inplace = True)
    
    execute_values(conn, user_preferences, 'userpreferences')
    
    todays_date = datetime.today().strftime("%Y-%m-%d")
    
    print(f'Added {n_new} user preferences on {todays_date}')
    
fill_tables()