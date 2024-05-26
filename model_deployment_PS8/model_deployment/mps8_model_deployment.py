import pandas as pd
import numpy as np
import re

import os
import sys

import joblib
import fasttext

from nltk.corpus import stopwords

ft_model = fasttext.load_model('cc.en.300.bin')
stop_words = set(stopwords.words('english'))

def preprocesamiento_texto(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def obtener_embedding(texto, modelo):
    tokens = texto.split()
    embeddings = [modelo.get_word_vector(word) for word in tokens]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(modelo.get_dimension())

def probabilidad_genero_pelicula(year, title, plot):

    scaler = joblib.load(os.path.dirname(__file__) + '/scaler_year.pkl')

    dicc = {'year': [year],
            'title': [title],
            'plot': [plot]}
    
    base = pd.DataFrame(dicc)

    base['plot'] = base['plot'].apply(preprocesamiento_texto)
    base['title'] = base['title'].apply(preprocesamiento_texto)

    X_plot = np.array([obtener_embedding(resumen, ft_model) for resumen in base['plot']])
    X_title = np.array([obtener_embedding(titulo, ft_model) for titulo in base['title']])

    X_year = scaler.transform(base[['year']])

    X = np.concatenate([X_year, X_title, X_plot], axis=1)
    
    cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
        'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
        'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']
    
    clf = joblib.load(os.path.dirname(__file__) + '/red_neuronal_peliculas.pkl')
    
    proba_genres = clf.predict(X)
    df = pd.DataFrame(proba_genres, columns=cols)
    diccionario = df.to_dict(orient='list')

    return diccionario

if __name__ == "__main__":
    
    print('Inicio Test')
    p1 = probabilidad_genero_pelicula('1999', 'Drugs', 'This is a comedy movie')
    print(p1)
        
