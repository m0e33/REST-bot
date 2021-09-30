from flask import Flask, request
from application.inference_service import InferenceService

class FlaskApp(Flask):

    def __init__(self, *args, **kwargs):
        super(FlaskApp, self).__init__(*args, **kwargs)
        self.inference_service = InferenceService()
        self.route("/")(self.works)
        self.route("/inference")(self.inference)

    def works(self):
        return "Server works"

    def inference(self):
        date = request.args.get('date')
        symbols = request.args.get('symbols').split('-')
        result = self.inference_service.get_prediction(symbols, date)
        return str(result)


app = FlaskApp(__name__)