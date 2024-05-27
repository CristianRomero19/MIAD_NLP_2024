import pandas as pd
import os
import joblib
import heapq

def probabilidad_genero_pelicula(plot: str):
    # Sequential modelo pre entrenado 
    clf = joblib.load(os.path.dirname(__file__) + '/red_neuronal_peliculas_alterno.pkl')
    
    # Inicializar CountVectorizer con el máximo de características deseado
    vect = joblib.load(os.path.dirname(__file__) + '/vect.pkl')

    # Lista de documentos
    text = [plot]

    # Transformar los documentos
    plotFit = vect.transform(text)
    
    # Asegúrate de que la entrada tenga la forma correcta
    plotFit = plotFit.toarray()  # Convertir a matriz densa si es necesario
    
    # Realizar la predicción
    predict = clf.predict(plotFit)

    # Valores columnas
    cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
        'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
        'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']

    # crear un data frame con las predicciones y sus nombres
    df = pd.DataFrame(predict, columns=cols)

    # Transformar el df a un diccioario
    result = df.to_dict(orient='list')

    # Obtener los 5 valores más altos
    valores_mas_altos = heapq.nlargest(5, result.values())

    # Ordenar los valores más altos de mayor a menor
    valores_mas_altos.sort(reverse=True)

    # Para obtener las claves correspondientes a estos valores ordenados
    claves_y_valores_mas_altos = {clave: result[clave] for valor in valores_mas_altos for clave in result if result[clave] == valor}

    # retornar prediccion
    return claves_y_valores_mas_altos

if __name__ == "__main__":
    
    print('Inicio Test')
    p1 = probabilidad_genero_pelicula('This is a porn movie')

    print('esta es la respuesta:')
    print(p1)
        
