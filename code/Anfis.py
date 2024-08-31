import pandas as pd
import string
import re
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions
from transformers import BertTokenizer, AutoTokenizer
from tqdm import tqdm
from transformers import pipeline, AutoModelForSequenceClassification
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer, BertModel
import logging as log
import torch
import torch.nn as nn
from torchCRF import CRF
from afinn import Afinn


import matplotlib.pyplot as plt
import seaborn as sns

import skfuzzy as fuzz
import sys
sys.path.insert(0, './anfis_twmeggs/')
import anfis_twmeggs as anfis
from anfis import ANFIS


def getData():
    malta_loc_18 = '../data/Malta-Budget-2018-dataset-v1.csv'
    malta_loc_19 = '../data/Malta-Budget-2019-dataset-v1.csv'
    malta_loc_20 = '../data/Malta-Budget-2020-dataset-v1.csv'

    malta_data_18 = pd.read_csv(malta_loc_18)
    malta_data_19 = pd.read_csv(malta_loc_19)
    malta_data_20 = pd.read_csv(malta_loc_20)

    malta_data_19 = malta_data_19.rename(columns={'Off-topic ':'Off-topic'})
    combined_data = pd.concat([malta_data_18, malta_data_19, malta_data_20], ignore_index=True)

    clean_data = combined_data.dropna(subset=['Online Post Text'])
    clean_data = clean_data.drop(['Twitter ID', 'Related Online Post ID', 'Source ID','Off-topic'], axis=1)
    clean_data = clean_data[clean_data['Language'] == 0] # get all data that is in english 
    clean_data = clean_data.drop(['Language'], axis=1)
    clean_data = clean_data.rename(columns={'Online Post ID':'ID','Online Post Text':'Text'})

    return clean_data

def processData(clean_data):

    processed_data = clean_data.copy(deep=True)

    def remove_special_characters(text):
        pattern = re.compile(r'[^a-zA-Z\s]')
        return pattern.sub('', text)

    # Remove URLs and HTML tags
    processed_data['Raw_Text'] = processed_data['Text']

    processed_data['Text'] = processed_data['Text'].str.replace(r'http\S+|www\S+|https\S+', '', regex=True)
    processed_data['Text'] = processed_data['Text'].str.replace(r'<.*?>', '', regex=True)

    # Expand contractions
    processed_data['Text'] = processed_data['Text'].apply(lambda x: contractions.fix(x))

    # Convert to lowercase
    processed_data['Text'] = processed_data['Text'].str.lower()

    # Remove punctuation
    processed_data['Text'] = processed_data['Text'].str.replace(f"[{string.punctuation}]", " ", regex=True)

    # Remove numbers
    processed_data['Text'] = processed_data['Text'].str.replace(r'\d+', '', regex=True)

    # Remove special characters
    processed_data['Text'] = processed_data['Text'].apply(remove_special_characters)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    processed_data['Text'] = processed_data['Text'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))

    # Remove extra whitespace
    processed_data['Text'] = processed_data['Text'].str.strip()
    processed_data['Text'] = processed_data['Text'].str.replace(r'\s+', ' ', regex=True)

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    processed_data['Text'] = processed_data['Text'].apply(lambda x: ' '.join(lemmatizer.lemmatize(word) for word in x.split()))

    # Tokenize
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') #30522 
    tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    tokenizer.model_max_length = 100
    tokenizer_features = 30522
    processed_data['tokens'] = processed_data['Text'].apply(lambda x: tokenizer.tokenize(x)) 

    max_words = processed_data['Text'].apply(lambda x: len(x.split())).max()

    max_tokens = processed_data['tokens'].apply(lambda x: len(x)).max()

    def encode_texts(texts, tokenizer, max_len): 
        input_ids = []
        attention_masks = []

        for text in texts:
            encoded = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_len,
                padding='max_length',
                return_attention_mask=True,
                return_token_type_ids=False,
                truncation=True
            )
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])
        
        return input_ids, attention_masks

    processed_data['padded'], processed_data['masks'] = encode_texts(processed_data['Text'].tolist(), tokenizer, 100)

    return processed_data


def getPunctuation(processed_data):
    def count_punctuation(text, tokens):
        exclamation_count = len(re.findall(r'!', text))
        question_count = len(re.findall(r'\?', text))
        ellipsis_count = len(re.findall(r'\.\.\.', text))
        comma_count = len(re.findall(r',', text))
        period_count = len(re.findall(r'\. ', text))
        token_count = len(tokens)
        if token_count == 0:
            token_count = 1
        return pd.Series({
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'ellipsis_count': ellipsis_count,
            'comma_count': comma_count,
            'period_count': period_count,
            'exclamation_score': exclamation_count / token_count,
            'question_score': question_count / token_count,
            'ellipsis_score': ellipsis_count / token_count,
            'comma_score': comma_count / token_count,
            'period_score': period_count / token_count
        })

    processed_data = processed_data.join(
        processed_data.apply(lambda row: count_punctuation(row['Raw_Text'], row['tokens']), axis=1)
    )
    return processed_data
    

def getPolarity(processed_data):
    tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    sentiment_analyzer = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

    def get_sentiment(tokens):
        total_score = 0
        result = sentiment_analyzer(tokens)
        if len(result) == 0:
            return 0.0
        for r in result:
            total_score += r['score']
        return total_score / len(result)

    processed_data['polarity_score'] = processed_data['tokens'].apply(get_sentiment)

    return processed_data

def getSubjectivity(processed_data):
    loggers = [log.getLogger(name) for name in log.root.manager.loggerDict]
    for logger in loggers:
        if "transformers" in logger.name.lower():
            logger.setLevel(log.ERROR)

    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    # Tokenize and convert to BERT embeddings
    def get_bert_embeddings(texts):
        inputs = tokenizer(texts.tolist(), return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    X = get_bert_embeddings(processed_data['Text'])
    y = processed_data['Subjectivity']

    # Train SVM classifier
    clf = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
    clf.fit(X, y)

    # Get the decision function values or probabilities
    subjectivity_scores = clf.decision_function(X)
    processed_data['Subjectivity Score'] = subjectivity_scores
    return processed_data

def getSarcasm(processed_data):
    class SarcasmLSTM(nn.Module):
        def __init__(self, bert_model, hidden_dim, output_dim, num_layers):
            super(SarcasmLSTM, self).__init__()
            self.bert = bert_model
            self.lstm = nn.LSTM(bert_model.config.hidden_size, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)
            self.sigmoid = nn.Sigmoid()

        def forward(self, inputs):
            with torch.no_grad():
                bert_output = self.bert(**inputs).last_hidden_state
            lstm_output, _ = self.lstm(bert_output)
            logits = self.fc(lstm_output[:, -1, :])
            return self.sigmoid(logits)

    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    # Initialize the LSTM model
    sarcasm_model = SarcasmLSTM(bert_model, hidden_dim=256, output_dim=1, num_layers=2)

    # Tokenize and predict sarcasm for the entire dataset
    inputs = tokenizer(processed_data['Text'].tolist(), return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        sarcasm_prob = sarcasm_model(inputs).numpy()

    processed_data['Sarcasm Score'] = sarcasm_prob.flatten()  # Keep the raw probability
    return processed_data

def getNegation(processed_data):
    class LSTM_CRF(nn.Module):
        def __init__(self, bert_model, hidden_dim, tagset_size):
            super(LSTM_CRF, self).__init__()
            self.bert = bert_model
            self.lstm = nn.LSTM(bert_model.config.hidden_size, hidden_dim, batch_first=True, bidirectional=True)
            self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)
            self.crf = CRF(tagset_size, batch_first=True)

        def forward(self, inputs):
            with torch.no_grad():
                bert_output = self.bert(**inputs).last_hidden_state
            lstm_output, _ = self.lstm(bert_output)
            emissions = self.hidden2tag(lstm_output)
            return emissions

        def predict(self, inputs):
            emissions = self(inputs)
            return self.crf.decode(emissions), emissions

    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    # Initialize the LSTM-CRF model
    negation_model = LSTM_CRF(bert_model, hidden_dim=128, tagset_size=3)  # Assume 3 tags (e.g., "O", "NEG", "POS")

    # Tokenize and predict negation for the entire dataset
    inputs = tokenizer(processed_data['Text'].tolist(), return_tensors='pt', padding=True, truncation=True)

    negation_tags, negation_scores = negation_model.predict(inputs)

    # Extract the highest score for negation tag for each sentence
    processed_data['Negation Score'] = [torch.max(score[:, 1]).item() for score in negation_scores]  # Assuming index 1 is the "NEG" tag

    return processed_data

def getAfinn(processed_data):
    afinn = Afinn()
    processed_data['afinn_score'] = processed_data['Raw_Text'].apply(lambda x: afinn.score(x))

    return processed_data

def doAnfis(processed_data):
    to_keep = ['exclamation_score', 'question_score', 'ellipsis_score', 'comma_score', 'period_score',
           'Subjectivity Score', 'polarity_score', 'afinn_score', 'Negation Score', 'Sarcasm Score']
    fuzzy_data = processed_data[to_keep]
    X = fuzzy_data.values
    Y = processed_data['Emotion'].values
    
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
    
    mfc = anfis.membershipfunction.MemFuncs(mf)

    anf = ANFIS(X, y, mfc)

    anf.trainHybridJangOffLine(epochs=2)
    
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
    

clean_data = getData()
processed_data = processData(clean_data=clean_data)
log.debug("processed")
processed_data = getPunctuation(processed_data=processed_data)
log.debug("punctuaton counted")
processed_data = getPolarity(processed_data=processed_data)
log.debug("polarity calculated")
processed_data = getSubjectivity(processed_data=processed_data)
log.debug("subjectivity calculated")
processed_data = getSarcasm(processed_data=processed_data)
log.debug("sarcasm calculated")
processed_data = getAfinn(processed_data=processed_data)
log.debug("afinn calculated")
anfis = doAnfis(processed_data=processed_data)
log.debug("Anfis completed")





