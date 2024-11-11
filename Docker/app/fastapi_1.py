#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load your clustering model
model = joblib.load("log_reg_like_dislike.joblib")

app = FastAPI()

class UserFeatures(BaseModel):
    # Define all the features that your model expects as input
    gender: float
    age: float
    platform_user: float
    gameid: float

@app.post("/like_dislike/")
async def predict_cluster(user_features: UserFeatures):
    # Extract features from the request
    features = [
        user_features.gender,
        user_features.age,
        user_features.platform_user,
        user_features.gameid,
    ]

    # Make prediction
    prediction = model.predict([features])

    # Convert the predicted cluster to a standard Python integer
    predicted_cluster = int(np.round(prediction[0]))

    # Return the predicted cluster
    return {"cluster_generated": predicted_cluster}

