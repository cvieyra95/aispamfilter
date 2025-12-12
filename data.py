import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
import joblib #to save ai model


#reads the CSV file
def load_data():
    data = pd.read_csv('emails_cleaned.csv')
    return data

#splits the data
def split_data(data):
    X = data["text"]
    y = data["target"]
    return train_test_split(X, y, random_state=104, test_size=0.2, shuffle=True)  


#vectorizes text and runs NB
def build_model():
    pipe_line = make_pipeline(TfidfVectorizer(stop_words="english", min_df=2, max_df=.95, ngram_range=(1,2)), MultinomialNB()) 
    return pipe_line

#trains the model (supervised learning)
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

#how accuarete the ai model is
def predictions(model, X_test, y_test):
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    print("Accuracy: ", accuracy)
    return prediction


#saves the model
def save_model(model):
    joblib.dump(model, "emails.pkl")



if __name__ == "__main__":
    data = load_data()
    print(data["target"].value_counts()) 
    print(data["target"].value_counts(normalize=True))
    X_train, X_test, y_train, y_test = split_data(data)
    model = build_model()
    model = train_model(model, X_train, y_train)

    #classification report
    predict = predictions(model,X_test, y_test)
    print(classification_report(y_test, predict))
    save_model(model)

    