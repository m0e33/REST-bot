import os

import fire
import kfp
from kfp.compiler import compiler

from model.rest_kubeflow_adapter import KubeflowAdapter

if __name__ == '__main__':
    fire.Fire(KubeflowAdapter)


def compile_run_pipeline():
    download_data_container_op = kfp.components.load_component_from_file(os.path.join(os.getcwd(), 'download_component.yaml'))
    train_container_op = kfp.components.load_component_from_file(os.path.join(os.getcwd(), 'train_component.yaml'))

    def pipeline(cloud_path="ames_dataset", learning_rate=0.1, n_estimators=50):
        download_task = download_data_container_op(
            cloud_path=cloud_path
        )

        train_task = train_container_op(
            data_path=download_task.output,
            learning_rate=learning_rate,
            n_estimators=n_estimators
        )

    pipeline_filename = "train_pipeline.zip"
    compiler.Compiler().compile(pipeline, pipeline_filename)

    # in Kubeflow UI zip hochladen und dann starten
