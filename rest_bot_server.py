from flask import Flask, request
from application.inference_service import InferenceService

app = Flask(__name__)


@app.route('/')
def index():
    return 'Server Works!'


@app.route('/inference')
def inference():
    date = request.args.get('date')
    symbols = request.args.get('symbols').split('-')
    service = InferenceService()
    result = service.get_prediction(symbols, date)
    return result


@app.route('/config')
def get_config():
    pass