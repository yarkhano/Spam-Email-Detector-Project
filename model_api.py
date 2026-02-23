#import required libraries
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


#Building API for Model

#Creating class, to make sure the input is as expected(string).
class input_message(BaseModel):
    message: str

##Creating class, to make sure the output is as expected(string).
class output_message(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Spam Shield API</title>
        <style>
            :root {
                --primary: #4f46e5;
                --primary-hover: #4338ca;
                --bg: #f9fafb;
                --card-bg: #ffffff;
                --text-main: #111827;
                --text-muted: #6b7280;
            }

            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background-color: var(--bg);
                color: var(--text-main);
                margin: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }

            .container {
                text-align: center;
                padding: 2rem;
                background: var(--card-bg);
                border-radius: 16px;
                box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
                max-width: 500px;
                width: 90%;
            }

            .icon {
                font-size: 3rem;
                margin-bottom: 1rem;
            }

            h1 {
                margin: 0 0 0.5rem 0;
                font-size: 1.875rem;
                font-weight: 700;
                color: var(--text-main);
            }

            p {
                color: var(--text-muted);
                margin-bottom: 2rem;
                line-height: 1.5;
            }

            .status-badge {
                display: inline-block;
                padding: 0.25rem 0.75rem;
                background-color: #dcfce7;
                color: #166534;
                border-radius: 9999px;
                font-size: 0.875rem;
                font-weight: 500;
                margin-bottom: 2rem;
            }

            .button-group {
                display: flex;
                gap: 1rem;
                justify-content: center;
            }

            .btn {
                text-decoration: none;
                padding: 0.75rem 1.5rem;
                border-radius: 8px;
                font-weight: 600;
                transition: all 0.2s;
            }

            .btn-primary {
                background-color: var(--primary);
                color: white;
            }

            .btn-primary:hover {
                background-color: var(--primary-hover);
                transform: translateY(-1px);
            }

            .btn-secondary {
                background-color: #f3f4f6;
                color: var(--text-main);
            }

            .btn-secondary:hover {
                background-color: #e5e7eb;
            }

            footer {
                margin-top: 2rem;
                font-size: 0.75rem;
                color: var(--text-muted);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="icon">🛡️</div>
            <div class="status-badge">● API System Operational</div>
            <h1>Spam Email Detector</h1>
            <p>Welcome to the Spam Detection Engine. This API uses machine learning to classify messages and protect your inbox from noise.</p>

            <div class="button-group">
                <a href="/docs" class="btn btn-primary">Explore Documentation</a>
                <a href="/redoc" class="btn btn-secondary">Redoc UI</a>
            </div>

            <footer>
                Version 1.0.0 | Powered by FastAPI & Scikit-Learn
            </footer>
        </div>
    </body>
    </html>
    """



