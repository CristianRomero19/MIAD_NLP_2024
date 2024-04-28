#!/usr/bin/python

from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from mps4_model_deployment import precio_carro
from flask_cors import CORS

# Definición aplicación Flask
app = Flask(__name__)

# Definición API Flask
api = Api(
    app, 
    version='1.0', 
    title='Car Price Prediction API',
    description='API for predicting car prices')

ns = api.namespace('predict', description='Car Price Prediction Endpoint')

# Definición argumentos o parámetros de la API
parser = api.parser()
parser.add_argument('year', type=int, required=True, help='Car year', location='args')
parser.add_argument('mileage', type=int, required=True, help='Car mileage', location='args')
parser.add_argument('state', type=str, required=True, help='Car state', location='args')
parser.add_argument('make', type=str, required=True, help='Car make', location='args')
parser.add_argument('model', type=str, required=True, help='Car model', location='args')


resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class PredictPrice(Resource):
    @api.expect(parser)
    @api.marshal_with(resource_fields)
    def get(self):
        # Obtener los parámetros de la solicitud
        args = parser.parse_args()
        
        # Llamar a la función precio_carro con los parámetros obtenidos
        result = precio_carro(args['year'], args['mileage'], args['state'], args['make'], args['model'])

        # Devolver el resultado en formato JSON
        return {'result': result}, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
