
from application.inference_service import InferenceService

if __name__ == '__main__':
    service = InferenceService()

    result = service.get_prediction(["TSLA", "NFLX"], "2021-09-01")