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