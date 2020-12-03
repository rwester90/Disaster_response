# Disaster Response Pipeline Project
### Summary
The repository contains a web application based on python and flask for automatically classification of text messages in a Disaster Response context. 
The web interface provides new message classification and also a main overview of the used dataset.

The main components of the app are the following:
1) ETL Pipeline (process_data.py) Loads the input data, cleans the data and stores it as a database for further processing.
2) ML Pipeline (train_classifier.py) Loads previous database file, builds a text preprocessing and machine learning pipeline, 
trains and tunes the best model using GridsearchCV and exports it.
3) Flask Web App (run.py) Interactive web app for message classification and main data visualization.

### Instructions: 

1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
      

2. Run the following command in the app's directory to run the web app
    `python run.py`

3. Go to http://0.0.0.0:3001/ to use the web app to classify you own message and see some visualizations about the original dataset.


### Requirements:
The following python must be installed: 
* re
* joblib
* flask
* plotly
* sqlalchemy
* pandas
* numpy
* sklearn
* nltk
* json
