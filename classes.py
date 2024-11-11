from dataclasses import dataclass

@dataclass
class Games:
    gameid:int
    year_released:int
    rating:float
    category:str
    min_required_age:int
    price:float
    platform_game:list
    has_offer:bool
    min_system_req:str
    
    

@dataclass
class User:
    uid:str
    gender:str
    age:int
    country:str
    platform_user:str
    cluster:int



