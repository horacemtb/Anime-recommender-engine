import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics.pairwise import cosine_similarity
import jaro
import pickle
from loguru import logger

def recommend(user_input, model, ratings_df, user_df, anime_emb, anime_sel, new_user_preferences = {}):
    
    if type(user_input) is int:
        
        logger.debug('Getting recommendations for user')
        
        u_id = user_input
        
        if u_id in user_df.user_id.values:
            
            logger.debug('Getting recommendations for existing user')
            user_emb = user_df[user_df['user_id'] == u_id]
            lgb_dataset = pd.DataFrame(np.repeat(user_emb.values, len(anime_emb), axis = 0), columns = user_emb.columns).\
            join(anime_emb)

            logger.debug('Predicting rating')
            preds_rating = model.predict(lgb_dataset.drop(['user_id', 'anime_id'], axis = 1))
            lgb_dataset['predicted_rating'] = preds_rating

            emb_cols = [col for col in lgb_dataset.columns if 'emb' in col]

            user_ratings = ratings_df[ratings_df['user_id'] == u_id]

            logger.debug('Selecting anime watched by user')
            watched = lgb_dataset[lgb_dataset['anime_id'].\
                                 isin(user_ratings.anime_id.values)][['anime_id', 'predicted_rating']+emb_cols].\
            merge(user_ratings[['anime_id', 'rating']], on = 'anime_id').\
            merge(anime_sel[['anime_id', 'Name']], on = 'anime_id').\
            sort_values(by = 'rating', ascending = False)
            
            if watched['rating'].max() >= 7:
            
                user_favorite_emb = np.mean(watched[watched['rating'] == watched['rating'].max()][emb_cols].values, axis = 0)

                logger.debug('Calculating top 1000')
                top_1000_user = lgb_dataset[~lgb_dataset['anime_id'].\
                                         isin(user_ratings.anime_id.values)][['anime_id', 'predicted_rating']+emb_cols].\
                merge(anime_sel[['anime_id', 'Name']], on = 'anime_id').\
                sort_values(by = 'predicted_rating', ascending = False).iloc[:1000, :]

                top_1000_user['score'] = cosine_similarity([user_favorite_emb], top_1000_user[emb_cols].values)[0]

                logger.debug('Selecting top 10')
                top_10_user = top_1000_user.sort_values(by = ['score', 'predicted_rating'], ascending = False).\
                iloc[:10, :][['anime_id', 'Name', 'predicted_rating', 'score']].\
                sort_values(by = 'predicted_rating', ascending = False)
            
            else:
                
                lgb_dataset = lgb_dataset.merge(anime_sel[['anime_id', 'Name']], on = 'anime_id')

                logger.debug('Selecting top 10')
                top_10_user = lgb_dataset.sort_values(by = 'predicted_rating', ascending = False).\
                iloc[:10, :][['anime_id', 'Name', 'predicted_rating']].\
                sort_values(by = 'predicted_rating', ascending = False)

            watched = watched[['anime_id', 'Name', 'rating', 'predicted_rating']]
            
            return watched, top_10_user
        
        elif u_id not in user_df.user_id.values:
            
            if len(new_user_preferences) > 0:
                logger.debug('Getting recommendations for new user with preferences')

                user_dict = {}
                for k in user_df.columns:
                    if k in new_user_preferences:
                        user_dict[k] = new_user_preferences[k]
                    else:
                        user_dict[k] = [0]
                
                relevant_anime = []
                for anime_id, genres in zip(anime_sel['anime_id'], anime_sel['Genres'].values):
                    if set(genres.split(',')).intersection(set(new_user_preferences.keys())):
                        relevant_anime.append(anime_id)                             
                          
            else:
                logger.debug('Getting recommendations for new user without preferences')
                user_dict = {}
                for k in user_df.columns:
                    user_dict[k] = [0]
                    
                relevant_anime = []

            user_emb = pd.DataFrame.from_dict(user_dict)
            lgb_dataset = pd.DataFrame(np.repeat(user_emb.values, len(anime_emb), axis = 0), columns = user_emb.columns).\
            join(anime_emb)
            
            if len(relevant_anime) > 1:
                lgb_dataset = lgb_dataset[lgb_dataset['anime_id'].isin(relevant_anime)]

            logger.debug('Predicting rating')
            preds_rating = model.predict(lgb_dataset.drop(['user_id', 'anime_id'], axis = 1))
            lgb_dataset['predicted_rating'] = preds_rating

            lgb_dataset = lgb_dataset.merge(anime_sel[['anime_id', 'Name']], on = 'anime_id')

            logger.debug('Selecting top 10')
            top_10_user = lgb_dataset.sort_values(by = 'predicted_rating', ascending = False).\
            iloc[:10, :][['anime_id', 'Name', 'predicted_rating']].\
            sort_values(by = 'predicted_rating', ascending = False)

            return None, top_10_user
        
    elif type(user_input) is str:
        
        logger.debug('Searching for similar anime')
        
        a_name = user_input
        anime_df = anime_sel[['anime_id', 'Name', 'Genres', 'Score']].merge(anime_emb, on = 'anime_id')
        
        emb_cols = [col for col in anime_df.columns if 'emb' in col]
        
        if a_name not in anime_df['Name'].values:
            
            similar_names = []
            
            for name in anime_sel['Name'].values:
                score = jaro.jaro_winkler_metric(a_name, name)
                if score > 0.9:
                    similar_names.append((name, score))
            if len(similar_names) == 0:
                logger.debug('Anime not found in our database! Please check back later')
                anime_search = pd.DataFrame.from_dict({'result': ['Anime not found in our database! Please check back later']})
                return None, anime_search
            else:
                similar_names.sort(key = lambda x: x[1], reverse = True)
                a_name = similar_names[0][0]
            
        logger.debug('Requested anime found in database')

        anime_request = anime_df[anime_df['Name'] == a_name]
        anime_search = anime_df[~anime_df['Name'].isin([a_name])]
        anime_search['score'] = cosine_similarity(anime_request[emb_cols].values, anime_search[emb_cols].values)[0]

        logger.debug('Found new anime for you')

        anime_search = anime_search.sort_values(by = 'score', ascending = False).iloc[:10, :][['anime_id', 'Name', 'Genres', 'Score']]

        return None, anime_search
