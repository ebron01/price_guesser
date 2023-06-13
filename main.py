import pandas as pd
import pickle
from utils import *
import os
from model.RandomForest import RandomForest
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

projectDir = os.path.join(os.getcwd() , 'data') 

def main():
    if False:
        with open(os.path.join(projectDir, 'hotel_data.pkl'), 'wb') as output:
            encoded_data = processData(projectDir)
            pickle.dump(encoded_data, output, pickle.HIGHEST_PROTOCOL)
    else :
        with open(os.path.join(projectDir, 'hotel_data.pkl'), 'rb') as input:
            encoded_data = pickle.load(input)
    
    
    X_data = dataPreprocess(encoded_data)    

    #TODO: Decide which columns to use as training data. Datetimes are not required, consider years also.
    # X_valid = X_data.sample(frac = 0.3, random_state = 42)
    # Y_valid = X_valid['average_amount'].values
    # X_train = X_data.drop(X_valid.index)
    # Y_train = X_train['average_amount'].values
    
    # X_valid = X_valid.drop(['total_amount', 'average_amount'], axis=1).values
    # X_train = X_train.drop(['total_amount', 'average_amount'], axis=1).values
    
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_data.drop(['total_amount', 'average_amount'], axis=1).values, X_data['average_amount'].values, test_size=0.2)
    
    regr = RandomForest(X_train, Y_train)

    for i in range(15):
        regr.forward()
        loss = regr.predict(X_valid, Y_valid)
        print(loss)

if __name__ == '__main__':
    main()