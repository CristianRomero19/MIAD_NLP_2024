import pandas as pd
import os
import joblib

def probabilidad_genero_pelicula(plot: str):
    # Sequential modelo pre entrenado 
    clf = joblib.load(os.path.dirname(__file__) + '/red_neuronal_peliculas_alterno.pkl')
    
    # Inicializar CountVectorizer con el máximo de características deseado
    vect = joblib.load(os.path.dirname(__file__) + '/vect.pkl')

    # Lista de documentos
    documents = [plot]

    # Transformar los documentos
    plotFit = vect.transform(documents)
    
    # Asegúrate de que la entrada tenga la forma correcta
    plotFit = plotFit.toarray()  # Convertir a matriz densa si es necesario
    
    # Realizar la predicción
    predict = clf.predict(plotFit)

    cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
        'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
        'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']

    df = pd.DataFrame(predict, columns=cols)
    result = df.to_dict(orient='list')

    # retornar prediccion
    return result

if __name__ == "__main__":
    
    print('Inicio Test')
    p1 = probabilidad_genero_pelicula('This is a porn movie')
    print(p1)
        
