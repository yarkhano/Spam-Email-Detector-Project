from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
from typing import List

app = FastAPI(
    title= "This is a Spam email detector APi",
    description="Created an api for spam detection project",
    version = "1.0.0"
)


#load the model and vectorizer
try:
    with open("spam_mail_model.pkl","rb") as file:
        model = pickle.load(file)

    with open("spam_vectorizer.pkl","rb") as file:
        vectorizer = pickle.load(file)
        print("Model and vectrizer loaded successfully")

except FileNotFoundError as e:
    print(f"error in loading model is :{e}")


