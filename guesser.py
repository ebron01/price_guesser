import pandas as pd
import pickle
from utils import *
import os

projectDir = os.path.join(os.getcwd() , 'data') 

def main():
    if True:
        with open(os.path.join(projectDir, 'hotel_data.pkl'), 'wb') as output:
            encoded_data = processData(projectDir)
            pickle.dump(encoded_data, output, pickle.HIGHEST_PROTOCOL)
    else :
        with open(os.path.join(projectDir, 'hotel_data.pkl'), 'rb') as input:
            encoded_data = pickle.load(input)
            
if __name__ == '__main__':
    main()