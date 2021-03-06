import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Loads message input data and categories tags associated
    '''
    messages = pd.read_csv(messages_filepath,dtype=str)
    categories = pd.read_csv(categories_filepath,dtype=str)
    
    return pd.merge(messages,categories,on='id')


def clean_data(df):
    '''
    Cleans raw data by defining dummy columns for each message tag
    category and deletes duplicated rows
    '''
    categories = df['categories'].str.split(pat=';',expand=True)
    
    #Rename categories colums
    row = categories.iloc[0,:]
    categories.columns = [i[0:-2] for i in row] 
    
    #Replace category columns in df with new category columns
    for column in categories.columns:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] =pd.to_numeric(categories[column]) 
    
    
    df=df.drop(['categories'],axis=1)   
    df = pd.concat([df,categories],axis=1)
    
    #Drop duplicated rows
    df=df.drop_duplicates()
    
    #Drop messages with no binary categorie's values
    for col in categories.columns:
        df=df[df[col]<2]

    return df
    
    return df
    
def save_data(df, database_filename):
    '''
    Creates an engine and loads data into a database
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Clean_Messages', engine, index=False,if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
