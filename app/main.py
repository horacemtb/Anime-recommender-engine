import json
from loguru import logger
from fastapi import FastAPI
import pickle

import pandas as pd

from app.class_Recommend import Recommend
from app.predict import recommend

logger.debug('Running app')

app = FastAPI()

logger.debug('Loading data')
anime_selected = pd.read_csv('./app/data/anime_selected.csv')
anime_synopsis_emb = pd.read_csv('./app/data/anime_synopsis_emb_2.csv')
rating_selected = pd.read_csv('./app/data/rating_selected.csv')
lightfm_emb_df = pd.read_csv('./app/data/lightfm_emb_df_50e.csv')
user_df = pd.read_csv('./app/data/user-preferences.csv')
anime_emb = lightfm_emb_df.merge(anime_synopsis_emb, on = 'anime_id')

logger.debug('Data loaded')

model_path = './app/models/LightGBM-v5.pickle'

logger.debug('Loading model')
model = pickle.load(open(model_path, 'rb'))

logger.debug('Model loaded')

@app.post('/recommend/')
async def get_recommendations(rec: Recommend):
    
    try:
        watched, recs = recommend(rec.user_input, model, rating_selected, user_df, anime_emb, anime_selected, rec.user_preferences)

        try:
            rec.watched = json.dumps(watched.to_dict())
        except:
            rec.watched = None

        rec.recommendations = json.dumps(res.to_dict())
        
    except Exception as ex:
        logger.error(f'Error getting recommendations in main.py: {ex}')

    return Recommend