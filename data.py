import pandas as pd
import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import joblib #to save ai model


#function reads the CSV file
def load_data():
    data = pd.read_csv('emails.csv')
    return data

#function splits the data
def split_data(X, y):
    return train_test_split(X, y, random_state=104, test_size=0.25, shuffle=True)  

#function converts enlgish words into Vectors 
def convert_data(X_train, X_test):
    vectorizer = TfidfVectorizer()
    X_train_vector= vectorizer.fit_transform(X_train)
    X_test_vector = vectorizer.transform(X_test)
    X_trainVector = X_train_vector.toarray()
    X_testVector = X_test_vector.toarray()
    return X_trainVector, X_testVector, vectorizer
    
#function trains the model (supervised learning)
def train_model(X_train_vector, y_train):
    model = GaussianNB()
    model.fit(X_train_vector, y_train)
    return model

#function gives us an accuracy score
def predictions(model, X_test_vector, y_test):
    prediction = model.predict(X_test_vector)
    accuracy = accuracy_score(y_test, prediction)
    return accuracy

#function checks whether or not an email is spam or not
def check_email(text):
    vector = vectorizer.transform([text])
    text_vector = vector.toarray()
    prediction = model.predict(text_vector)
    return prediction[0]



emails = load_data()
X_train, X_test, y_train, y_test = split_data(emails['text'], emails['spam'])
X_train_vector, X_test_vector, vectorizer = convert_data(X_train, X_test)
model = train_model(X_train_vector, y_train)

#saves the ai model
filename = 'spamfilter.sav'
joblib.dump(model,filename)
joblib.dump(vectorizer, 'vectorizer.sav')
    