from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

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

model = joblib.load('spam_filter_update.pkl')

class Email(BaseModel):
    text:str

class Feedback(BaseModel):
    text:str
    correct: bool
    prediction: int


@app.post("/predict")
def predict(email: Email):
    prediction = model.predict([email.text])[0]
    return {"spam": bool(prediction)}

@app.post("/feedback")
def feedback(feedback: Feedback):
    df = pd.read_csv("feedback_data.csv")
    new_row = {"Text": feedback.text, "Prediction": feedback.prediction, "Correct": feedback.correct}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv("feedback_data.csv", index=False)
    return {"status: success"}
