import pickle
import pandas as pd


pipe = pickle.load(open("RidgeModel.pkl", "rb"))

model_input = pd.DataFrame([["Electronic City Phase II", 1056, 2, 2]], columns=[
    "location", "total_sqft", "bath", "bhk"])

prediction = pipe.predict(model_input)

print(prediction[0])
