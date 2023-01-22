from typing import Union
from pydantic import BaseModel

class Recommend(BaseModel):
    user_input: Union[int, str]
    user_preferences: dict
    watched: Union[dict, None] = None
    recommendations: Union[dict, None] = None