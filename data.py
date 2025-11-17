import pandas as pd
import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


#This function reads the CSV file
def load_data():
    data = pd.read_csv('emails.csv')
    return data

#This functions splits the data
def split_data(X, y):
    return train_test_split(X, y, random_state=104, test_size=0.25, shuffle=True)  

#This function converts enlgish words into Vectors 
def convert_data(X_train, X_test):
    vectorizer = TfidfVectorizer()
    X_train_vector= vectorizer.fit_transform(X_train)
    X_test_vector = vectorizer.transform(X_test)
    X_trainVector = X_train_vector.toarray()
    X_testVector = X_test_vector.toarray()
    return X_trainVector, X_testVector, vectorizer
    
#This function trains the model    
def train_model(X_train_vector, y_train):
    model = GaussianNB()
    model.fit(X_train_vector, y_train)
    return model

#This function gives us an accuracy score
def predictions(model, X_test_vector, y_test):
    prediction = model.predict(X_test_vector)
    accuracy = accuracy_score(y_test, prediction)
    return accuracy

emails = load_data()
X_train, X_test, y_train, y_test = split_data(emails['text'], emails['spam'])
X_train_vector, X_test_vector, vectorizer = convert_data(X_train, X_test)
model = train_model(X_train_vector, y_train)
accuracy = predictions(model, X_test_vector, y_test)
print(accuracy)
