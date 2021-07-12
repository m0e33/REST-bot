from pipeline import compile_run_pipeline
from xgboost_model import XGBoostModel

if __name__ == '__main__':
    model = XGBoostModel()

    # model.create_bucket()
    # model.upload_data('ames_dataset', 'ames_dataset')
    # model.download_data('ames_dataset', 'ames_dataset2')

    # # local training
    # model.train()

    # # local prediction
    # model.predict({'data': [[2.000e+00, 2.000e+01, 8.000e+01, 9.600e+03, 6.000e+00, 8.000e+00, 1.976e+03,
    #                          1.976e+03, 0.000e+00, 9.780e+02, 0.000e+00, 2.840e+02, 1.262e+03, 1.262e+03,
    #                          0.000e+00, 0.000e+00, 1.262e+03, 0.000e+00, 1.000e+00, 2.000e+00, 0.000e+00,
    #                          3.000e+00, 1.000e+00, 6.000e+00, 1.000e+00, 1.976e+03, 2.000e+00, 4.600e+02,
    #                          2.980e+02, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
    #                          5.000e+00, 2.007e+03]]}, None)

    # build the "pre-build" docker image which includes all needed dependencies. When the pre-build image is updated everytime the dependencies change the actual build time of the normal docker image is a lot faster.
    model.build_prebuild_docker_image()

    # build docker image locally - not needed when retraining model with new data - needed when changing code
    model.build_push_docker_image()

    # trains the model in kubeflow (cluster)
    # model.train_online()

    # serves the model for inference in kubeflow (cluster)
    # model.deploy()

    # compile_run_pipeline()
