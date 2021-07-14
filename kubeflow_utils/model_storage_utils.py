import logging
import subprocess
import sys

import joblib


def gcs_copy(src_path: str, dst_path: str):
    if sys.platform == 'win32' or sys.platform == 'nt':
        logging.info(
            subprocess.run(['gsutil', 'cp', src_path, dst_path], stdout=subprocess.PIPE, shell=True).stdout[:-1].decode('utf-8'))
    else:
        logging.info(
            subprocess.run(['gsutil', 'cp', src_path, dst_path], stdout=subprocess.PIPE).stdout[:-1].decode(
                'utf-8'))
    logging.info(f'Copied {src_path} to {dst_path}')


def gcs_copy_dir(src_path: str, dst_path: str):
    if sys.platform == 'win32' or sys.platform == 'nt':
        logging.info(
            subprocess.run(['gsutil', '-m', 'cp', '-r', src_path, dst_path], stdout=subprocess.PIPE, shell=True).stdout[
            :-1].decode('utf-8'))
    else:
        logging.info(
            subprocess.run(['gsutil', '-m', 'cp', '-r', '-o', '"GSUtil:parallel_process_count=1"',  src_path, dst_path], stdout=subprocess.PIPE).stdout[:-1].decode(
                'utf-8'))
    logging.info(f'Copied {src_path} to {dst_path}')


def gcs_make_bucket(bucket_name: str, project_name: str):
    if sys.platform == 'win32' or sys.platform == 'nt':
        logging.info(
            subprocess.run(['gsutil', 'mb', '-p', project_name, bucket_name], stdout=subprocess.PIPE,
                           shell=True).stdout[:-1].decode('utf-8'))
    else:
        logging.info(subprocess.run(['gsutil', 'mb', '-p', project_name, bucket_name], stdout=subprocess.PIPE).stdout[
                     :-1].decode('utf-8'))
    logging.info(f'Created bucket {bucket_name}')


def save_model(model, model_file):
    """Save XGBoost model for serving."""
    joblib.dump(model, model_file)
    logging.info("Model export success: %s", model_file)
