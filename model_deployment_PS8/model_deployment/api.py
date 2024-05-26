#!/usr/bin/python

from flask import Flask
from flask_restx import Api, Resource, fields
from mps8_model_deployment import probabilidad_genero_pelicula
from flask_cors import CORS

# Definición aplicación Flask
app = Flask(__name__)

# Definición API Flask
api = Api(
    app, 
    version='1.0', 
    title='Movie genres prediction',
    description='API for predicting movie genres')

ns = api.namespace('predict', description='Movie genres prediction Endpoint')

# Definición argumentos o parámetros de la API
parser = api.parser()
parser.add_argument('year', type=int, required=True, help='movie year', location='args')
parser.add_argument('title', type=str, required=True, help='movie title', location='args')
parser.add_argument('plot', type=str, required=True, help='movie plot', location='args')


resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class PredictGenres(Resource):
    @api.expect(parser)
    @api.marshal_with(resource_fields)
    def get(self):
        # Obtener los parámetros de la solicitud 
        args = parser.parse_args()
        
        # Llamar a la función movie_genres con los parámetros obtenidos
        result = probabilidad_genero_pelicula(args['year'], args['title'], args['plot'])

        # Devolver el resultado en formato JSON
        return {'result': result}, 200
    
# http://192.168.0.45:5000/predict/?year=1990&title=Drugs&plot=This is a comedy movie

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
