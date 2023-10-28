import pickle
from utils import make_period_time
from fastapi import FastAPI
import re
import seaborn as sns
from fastapi.responses import StreamingResponse
import matplotlib.pyplot as plt
import os

app = FastAPI(title='Sale Predict')

filename_scaler = "models/bamland_scaler.pickle"
filename_model = "models/bamland_predictor.pickle"

scaler = pickle.load(open(filename_scaler, "rb"))
modelRFR = pickle.load(open(filename_model, "rb"))

delimiter_pattern = r'[-.,_/]'


@app.get("/sale_predict")
async def get_strings(start_date: str, end_date: str):
    start_time = re.split(delimiter_pattern, start_date)
    start_time = [int(x) for x in start_time]

    end_time = re.split(delimiter_pattern, end_date)
    end_time = [int(x) for x in end_time]

    period_time = make_period_time(start_time, end_time)

    inputs = scaler.transform(period_time)

    predictions = modelRFR.predict(inputs)
    return {"string1": start_date, "string2": end_date, "total_sale": predictions.sum(), "predictions": list(predictions)}


@app.get("/figure")
async def figure(start_date: str, end_date: str):
    start_time = re.split(delimiter_pattern, start_date)
    start_time = [int(x) for x in start_time]

    end_time = re.split(delimiter_pattern, end_date)
    end_time = [int(x) for x in end_time]

    period_time = make_period_time(start_time, end_time)

    inputs = scaler.transform(period_time)

    predictions = modelRFR.predict(inputs)
    sns.lineplot(x=range(len(predictions)), y=predictions, label='Sales Prediction')
    plt.savefig('sale.png')
    file = open('sale.png', mode='rb')
    return StreamingResponse(file, media_type="image/png")
