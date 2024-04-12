from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from schemas import HouseDataInput
import pandas as pd
import numpy as np
import pickle

app = FastAPI()

pipe = pickle.load(open("RidgeModel.pkl", "rb"))


@app.get("/")
async def test():
    return {
        "success": True,
        "message": "App is running successfully."
    }


@app.post("/predict")
async def predict_house_price(house_data: HouseDataInput):
    house_data = jsonable_encoder(house_data)
    location = house_data["location"]
    bhk = house_data["bhk"]
    no_bathrooms = house_data["no_bathrooms"]
    total_sqft = house_data["total_sqft"]
    try:
        model_input = pd.DataFrame([[location, total_sqft, no_bathrooms, bhk]], columns=[
            "location", "total_sqft", "bath", "bhk"])
        prediction = pipe.predict(model_input)[0]
    except Exception as e:
        print(e)
    return {
        "success": True,
        "data": house_data,
        "prediction": np.round(prediction * 100000)
    }
