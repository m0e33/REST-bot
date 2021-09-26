import logging
import argparse
import os

from model.rest_kubeflow_adapter import KubeflowAdapter

from utils.gpu import get_cuda_visible_devices
root_logger = logging.getLogger('')
root_logger.handlers = []

format = '([%(name)s] %(levelname)s %(asctime)s) -- %(message)s'
logging.basicConfig(filename="log.log", level=logging.DEBUG, format=format, datefmt='%H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
# set a format which is simpler for console use
formatter = logging.Formatter(format)
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)
logger = logging.getLogger('')
print(logger.handlers)

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpus', nargs='+', default='auto')
parser.add_argument('-n', '--n_gpus', type=int, default=-1)
args = parser.parse_args()

cuda_visible_devices = get_cuda_visible_devices(args.gpus, args.n_gpus)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
logger.info(f'SET CUDA_VISIBLE_DEVICES {cuda_visible_devices}')

if __name__ == '__main__':

    logger.info(
        "\n*************************************************\n"
        "*                                               *\n"
        "*               Starting model run              *\n"
        "*                                               *\n"
        "*************************************************\n"
    )
    num_gpus = len(args.gpus)
    model = KubeflowAdapter(num_gpus)

    # model.create_bucket()
    # model.upload_data('storage', 'data_raw')

    # model.download_data('data_raw/storage', 'data')

    # # local training
    model.train()

    # # local prediction
    #model.predict({'data': [[2.000e+00, 2.000e+01, 8.000e+01, 9.600e+03, 6.000e+00, 8.000e+00, 1.976e+03,
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
