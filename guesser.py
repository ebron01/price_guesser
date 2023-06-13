import pandas as pd
import pickle
from utils import *
import os
from model.RandomForest import RandomForest

projectDir = os.path.join(os.getcwd() , 'data') 

def main():
    if True:
        with open(os.path.join(projectDir, 'hotel_data.pkl'), 'wb') as output:
            encoded_data = processData(projectDir)
            pickle.dump(encoded_data, output, pickle.HIGHEST_PROTOCOL)
    else :
        with open(os.path.join(projectDir, 'hotel_data.pkl'), 'rb') as input:
            encoded_data = pickle.load(input)
    
    
    X_data = dataPreprocess(encoded_data)    

    #TODO: Decide which columns to use as training data. Datetimes are not required, consider years also.
    X_valid = X_data.sample(frac = 0.3, random_state = 42)
    Y_valid = X_valid['average_amount'].values
    X_train = X_data.drop(X_valid.index)
    Y_train = X_train['average_amount'].values
    
    X_valid = X_valid.drop(['total_amount', 'average_amount'], axis=1).values
    X_train = X_train.drop(['total_amount', 'average_amount'], axis=1).values
    
    
    rf = RandomForest(X_train, Y_train)
    rf = rf.forward()
    y_train_preds = rf.predict_proba(X_train)[:,1]
    y_valid_preds = rf.predict_proba(X_valid)[:,1]
    
    
if __name__ == '__main__':
    main()