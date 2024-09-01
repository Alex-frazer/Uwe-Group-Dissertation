import pandas as pd

import logging as log
from tqdm import tqdm 

import skfuzzy as fuzz

import skfuzzy.control as ctrl

import sys
import os
# sys.path.insert(0, './anfis_twmeggs/')
# import anfis_twmeggs as anfis
from anfis import ANFIS
import membershipfunction

log.basicConfig(
    level=log.DEBUG,  # Set the logging level to DEBUG to capture all log messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Include timestamp, logger name, and log level in messages
    handlers=[log.StreamHandler()]  # Ensure logs are sent to the console (stdout)
)

def doAnfis(processed_data):
    to_keep = ['exclamation_score', 'question_score', 'ellipsis_score', 'comma_score', 'period_score',
           'Subjectivity Score', 'polarity_score', 'afinn_score', 'Negation Score', 'Sarcasm Score']
    fuzzy_data = processed_data[to_keep]
    
    mapping_dict = {value: index for index, value in enumerate(processed_data['Emotion'].unique())}
    processed_data['Emotion_mapped'] = processed_data['Emotion'].map(mapping_dict)

    X = fuzzy_data.values    
    Y = processed_data['Emotion_mapped'].values
    print(Y)
    
    mf = [
        [['gaussmf',{'mean':-1,'sigma':1}],
        ['gaussmf',{'mean':1,'sigma':1}]], # exclamation_score
        [['gaussmf',{'mean':-1,'sigma':1}],
        ['gaussmf',{'mean':1,'sigma':1}]], # question_score
        [['gaussmf',{'mean':-1,'sigma':1}],
        ['gaussmf',{'mean':1,'sigma':1}]], # ellipsis_score
        [['gaussmf',{'mean':-1,'sigma':1}],
        ['gaussmf',{'mean':1,'sigma':1}]], # comma_score
        [['gaussmf',{'mean':-1,'sigma':1}],
        ['gaussmf',{'mean':1,'sigma':1}]], # period_score
        [['gaussmf',{'mean':-1,'sigma':1}],
        ['gaussmf',{'mean':0,'sigma':1}],
        ['gaussmf',{'mean':1,'sigma':1}]], # Subjectivity Score
        [['gaussmf',{'mean':-1,'sigma':1}],
        ['gaussmf',{'mean':0,'sigma':1}],
        ['gaussmf',{'mean':1,'sigma':1}]], # polarity_score
        [['gaussmf',{'mean':-2,'sigma':1}],
        ['gaussmf',{'mean':-1,'sigma':1}],
        ['gaussmf',{'mean':0,'sigma':1}],
        ['gaussmf',{'mean':1,'sigma':1}],
        ['gaussmf',{'mean':2,'sigma':1}]], # afinn_score
        [['gaussmf',{'mean':-1,'sigma':1}],
        ['gaussmf',{'mean':1,'sigma':1}]], # Negation Score
        [['gaussmf',{'mean':-1,'sigma':1}],
        ['gaussmf',{'mean':0,'sigma':1}],
        ['gaussmf',{'mean':1,'sigma':1}]]] #Sarcasm Score
    
    
    mfc = membershipfunction.MemFuncs(mf)
    log.debug("Memberships defined")
    anf = ANFIS(X, Y, mfc)
    log.debug("Anfis Created")
    anf.trainHybridJangOffLine(epochs=20)
    log.debug("Anfis Trained")
    anf.plotErrors()
    
    anf.memFuncs
    
    # print 'linguistic variables' in our membership functions
    # since we have defined 4 MF for each variable we will have 0,1,2,3 as values
    # for example, they could be low=0, medium=1, high=2, very high=3
    print('Linguistic variables:',anf.memFuncsByVariable)

    # print rules order
    # this is all the possible combinations of rules
    # e.g., [2,1] = a activates linguistic variable 2, b activates linguistic value 1 
    print('Combinations:', anf.rules)

    # print consequents
    print('Consequents:', anf.consequents)
    print('# of consenquents:',len(anf.consequents))
    anf.plotResults()
    return anf

if os.path.isfile('code/processed_data.csv'):
    processed_data = pd.read_csv('code/processed_data.csv', nrows=20)
    log.debug("Data loaded")
    anfis = doAnfis(processed_data=processed_data)
    log.debug("ANFIS completed")
else:
    log.error("File 'processed_data.csv' not found in the current directory.")