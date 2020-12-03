import sys
import nltk
nltk.download(['punkt','stopwords', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

import re
import numpy as np
import pandas as pd

from sqlalchemy import create_engine
import joblib

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC



def load_data(database_filepath):
    '''
    Loads preprocessed data in process_data.py and returns 
    input data for model estimation
    '''
    #Create engine and load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('SELECT * FROM Clean_Messages', con = engine)
    
    #Separate dependent and independent variables
    X = df['message']
    non_target=['id','message','original','genre',]
    category_columns=[i for i in df.columns if i not in non_target]
    y=df[category_columns]
    
    return X,y,category_columns


def tokenize(text):
    '''
    Tokenize function for Countvectorizer. Applies standard text preprocessing
    steps like normalization, stop words removal, lemmatization and stemming.
    '''
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Normalize text
    text = re.sub(r"\W"," ",text.lower())
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    porter=PorterStemmer()
    
    # Remove stop words
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    #We apply lemmatization
    lem_tokens = []
    for tok in tokens:
        lem_tok = lemmatizer.lemmatize(tok).lower().strip()
        lem_tokens.append(lem_tok)
        
    #and also stemming
    stem_tokens = []
    for tok in lem_tokens:
        stem_tok = porter.stem(tok).lower().strip()
        stem_tokens.append(stem_tok)

    return stem_tokens


def build_model():
    '''
    Defines a machine learning pipeline that firstly applies CountVectorizer and Tfidf
    transformation to the corpus data, then discards variables with lower variance that
    may be unnecessary and finally trains/fit a Random Forest Classifier 
    for each category tag.
    '''
    
    rfc=RandomForestClassifier(class_weight='balanced')

    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize, ngram_range= (1, 1))),
        ('tfidf',TfidfTransformer()),
        ('varth',VarianceThreshold(threshold = 0.0005)),
        ('clf',MultiOutputClassifier(rfc,n_jobs=1))
    ])
    
    
    parameters = {'vect__ngram_range':[(1,1),(1,2)],
              'varth__threshold':[0,0.0001,0.0002,0.0005]
    }
    
    return GridSearchCV(pipeline,param_grid=parameters)

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Shows performance metrics for each model
    '''
    y_pred=model.predict(X_test)
    
    for index, col in enumerate(category_names):
        print(col+' summary:')
        print("Accuracy:",accuracy_score(Y_test.iloc[:,index], y_pred[:,index]))
        print(classification_report(Y_test.iloc[:,index], y_pred[:,index]))


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()