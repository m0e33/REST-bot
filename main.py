from pipeline import compile_run_pipeline
from model.rest_kubeflow_adapter import KubeflowAdapter
import logging

format = '([%(name)s] %(levelname)s %(asctime)s) -- %(message)s'
logging.basicConfig(filename='log.log', level=logging.DEBUG, format=format, datefmt='%H:%M:%S', force=True)

# set up console logging
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter(format)
console.setFormatter(formatter)

logging.getLogger().addHandler(console)

logger = logging.getLogger("MAIN")

if __name__ == '__main__':

    logger.info(
        "\n*************************************************\n"
        "*                                               *\n"
        "*               Starting model run              *\n"
        "*                                               *\n"
        "*************************************************\n"
    )

    model = KubeflowAdapter()

    # model.create_bucket()
    # model.upload_data('storage', 'data_raw')

    # model.download_data('data_raw/storage', 'data')

    # # local training
    model.train()

    # # local prediction
    # model.predict({'data': [[2.000e+00, 2.000e+01, 8.000e+01, 9.600e+03, 6.000e+00, 8.000e+00, 1.976e+03,
    #                          1.976e+03, 0.000e+00, 9.780e+02, 0.000e+00, 2.840e+02, 1.262e+03, 1.262e+03,
    #                          0.000e+00, 0.000e+00, 1.262e+03, 0.000e+00, 1.000e+00, 2.000e+00, 0.000e+00,
    #                          3.000e+00, 1.000e+00, 6.000e+00, 1.000e+00, 1.976e+03, 2.000e+00, 4.600e+02,
    #                          2.980e+02, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
    #                          5.000e+00, 2.007e+03]]}, None)

    # build the "pre-build" docker image which includes all needed dependencies. When the pre-build image is updated everytime the dependencies change the actual build time of the normal docker image is a lot faster.
    # model.build_prebuild_docker_image()

    # build docker image locally - not needed when retraining model with new data - needed when changing code
    # model.build_push_docker_image()

    # trains the model in kubeflow (cluster)
    # model.train_online()

    # serves the model for inference in kubeflow (cluster)
    # model.deploy()

    # compile_run_pipeline()

    logger.info(
        "\n*************************************************\n"
        "*                                               *\n"
        "*               Finished model run              *\n"
        "*                                               *\n"
        "*************************************************\n"
    )
