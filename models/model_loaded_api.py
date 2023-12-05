# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("model_loaded_api")

# Create input/output pydantic models
input_model = create_model("model_loaded_api_input", **{'AppID': 922050, 'Name': 13759, 'Estimated owners': 1, 'Peak CCU': 24, 'Required age': 0, 'DLC count': 1, 'About the game': 36260, 'Supported languages': 6203, 'Full audio languages': 42, 'Reviews': 7944, 'Windows': True, 'Mac': False, 'Linux': False, 'Metacritic score': 0, 'Metacritic url': 0, 'User score': 0, 'Positive': 37, 'Negative': 4, 'Achievements': 52, 'Recommendations': 0, 'Notes': 3504, 'Average playtime forever': 0, 'Average playtime two weeks': 0, 'Median playtime forever': 0, 'Median playtime two weeks': 0, 'Developers': 24224, 'Publishers': 14622, 'Categories': 4697, 'Genres': 287, 'Tags': 25660})
output_model = create_model("model_loaded_api_output", prediction=14.99)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
