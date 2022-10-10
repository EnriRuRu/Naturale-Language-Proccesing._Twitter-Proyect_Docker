from flask import Flask, request, jsonify
import os
import pickle
from sklearn.model_selection import cross_val_score
import pandas as pd
import json
from os import environ
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


#os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route("/", methods=['GET'])
def hello():
    return "Bienvenido la API de predicción de sentimiento de texto"

# 1. Crea un endpoint que devuelva la predicción de los nuevos datos enviados mediante argumentos en la llamada
@app.route('/predict', methods=['GET'])
def predict():
    model = pickle.load(open('sentiment_model','rb'))

    text = request.args.get('texto', None)
    

    if text is None:
        return "<h3>Missing arg, the input 'texto' is needed to predict<h3>"
    else:
        
        
        # tratamiento de text antes de aplicar el modelo
    
    # lo meto a un dataFrame
    
        cosa = pd.DataFrame(columns=['texto'])
        cosa['texto']=[text]
        
        # quito signos puntuación
        signos = re.compile("(\.)|(\;)|(\:)|(\!)|(\?)|(\¿)|(\@)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")

        def signs_tweets(tweet):
            return signos.sub('', tweet.lower()) 
        
        cosa['texto'] = cosa['texto'].apply(signs_tweets)
        
        # quito links
        def remove_links(df):
            return " ".join(['{link}' if ('http') in word else word for word in df.split()])

        cosa['texto']=  cosa['texto'].apply(remove_links)
        
        # borro palabras 
        spanish_stopwords = stopwords.words('spanish')

        def remove_stopwords(df):
            return " ".join([word for word in df.split() if word not in spanish_stopwords])

        cosa['texto'] = cosa['texto'].apply(remove_stopwords)
        
        
        
        # recorto palabras    
        def spanish_stemmer(x):
            stemmer = SnowballStemmer('spanish')
            return " ".join([stemmer.stem(word) for word in x.split()])

        cosa['texto'] = cosa['texto'].apply(spanish_stemmer)
        
    
        # lo meto en una lista para poder aplicar el modelo
        listaa = list(cosa['texto'])
        
        # aplico el modelo
        
        prediction = model.predict(listaa)
        if prediction==0:
            return "<h3>The prediction of the feeling is:  GOOD<h3>"
        else:
             return "<h3>The prediction of the feeling is:  BAD<h3>"

if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0', port=environ.get("PORT", 5000))