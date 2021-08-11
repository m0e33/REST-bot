import os
import numpy as np


def find_free_gpus():
    if os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp') == 0:
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        gpus_most_idle = np.argsort(memory_available)[::-1]
        os.remove('tmp')
        return gpus_most_idle


def find_free_gpu():
    gpus = find_free_gpus()
    return str(gpus[0]) if gpus else ""
