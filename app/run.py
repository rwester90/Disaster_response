import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Clean_Messages', engine)

# load model
model = joblib.load('../models/classifier.pkl')


# indexwebpage displays cool visuals and receives user input text for model

@app. route('/')
@app.route('/index')
def index():
    
    # data for thirst visual: messages per genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # data for second visual: messages per category 
    non_target=['id','message','original','genre']
    category_names=[i for i in df.columns if i not in non_target]
    category_counts = (df.loc[:,category_names]).sum().values
    
    # data for third visual: tags per message distribution
    row_sum = df.loc[:,category_names].sum(axis=1)
    multilabel_counts = row_sum.value_counts().sort_index()
    multi_labels, multi_label_counts = multilabel_counts.index, multilabel_counts.values
    
    # create visuals
    graphs = [
        # Thirst graph
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # Second graph
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 45
                }
            }
        },
        # Third graph
        {
            'data': [
                Bar(
                    x=multi_labels,
                    y=multi_label_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Labels per message',
                'yaxis': {
                    'title': "Number of Messages"
                },
                'xaxis': {
                    'title': "Number of different labels per message"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)
   

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()