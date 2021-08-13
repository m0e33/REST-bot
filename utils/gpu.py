import os
import numpy as np


def get_cuda_visible_devices(selected_gpus, n_gpus):
    gpus = []
    if selected_gpus == "auto":
        if os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp') == 0:
            memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
            gpus_most_idle = list(np.argsort(memory_available)[::-1])
            os.remove('tmp')
            if n_gpus != -1:
                gpus_most_idle = gpus_most_idle[:n_gpus]
            gpus = gpus_most_idle
    else:
        assert(n_gpus == "all", "Can only use one of --gpus and --n_gpus.")
        gpus = list(selected_gpus)
    return ','.join(str(g) for g in gpus) if gpus else ''

