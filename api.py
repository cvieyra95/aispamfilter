from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:8080",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load('spamfilter.sav')
vectorizer = joblib.load('vectorizer.sav')

class Email(BaseModel):
    text:str


@app.post("/predict")
def predict(email: Email):
    vec = vectorizer.transform([email.text])
    vector = vec.toarray()
    prediction = model.predict(vector)[0]
    return {"spam": bool(prediction)}
