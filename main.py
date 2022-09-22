#eda Packages
import time
import pandas as pd
import random
import numpy as np
from art import *
from termcolor import colored
#import TF-IDF vectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier

def predict(passwd):
    #reading dataset
    pswd_data = pd.read_csv("data.csv",error_bad_lines=False)
    print(colored('[+]Fetching datasets\n','green'))
    #Here 1 refers average strength, 2 refers strong strength and 0 refers weak strength
    pswd_data['strength'].unique()
    #remove all null values in dataset
    pswd_data.isna().sum()
    pswd_data.dropna(inplace=True)
    pswd_data.isna().sum()
    pswd = np.array(pswd_data)
    #randoming the data for robustness
    random.shuffle(pswd)
    ylabels  = [s[1] for s in pswd]
    allpasswords = [s[0] for s in pswd]
    print(colored('[+]Data Preprocessing finished\n','green'))
    #function to split the input into characters of list
    #apply TF-IDF vertorizer on data for ML
    vectorizer = TfidfVectorizer(tokenizer=createTokens)
    X = vectorizer.fit_transform(allpasswords)
    print(colored('[+]Applying TF-IDF vectorizer on the given dataset containing passwords\n','green'))
    t=time.time()
    #split the data into training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2, random_state=42)	
    #using logistic regression 
    clf=DecisionTreeClassifier()	
    clf.fit(X_train, y_train)
    
    print(colored('[+]Finished training the data\n','green'))
    #print(colored('[+]Time taken\n'+str(time.time()-t),'yellow'))
    #check the accuracy of our model
    #print("Accuracy on trained set:",clf.score(X_train, y_train))
    #print("Accuracy on test sets:",clf.score(X_test, y_test))
    #predicting with our model
    X_predict = [passwd]
    X_predict = vectorizer.transform(X_predict)
    y_Predict = clf.predict(X_predict)
    print(colored('[+]Applying logistic regression on the untrained data/from user password\n','green'))
    if y_Predict[0]==0:
        print(colored('The password strength is very weak','red'))
    elif y_Predict[0]==1:
        print(colored('The passwords strength is medium','yellow'))
    elif y_Predict[0]==2:
        print(colored('The passwords strength is strong','green'))

def createTokens(f):
    tokens = []
    for i in f:
        tokens.append(i)
    return tokens


def main():
    tprint("PASSWORD STRENGTH PREDICTOR","rnd-medium")
    print(colored('Enter the password: ','green'))
    pas=str(input())
    predict(pas)

if __name__=="__main__":
    main()
