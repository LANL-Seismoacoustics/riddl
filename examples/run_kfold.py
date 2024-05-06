# © 2024. Triad National Security, LLC. All rights reserved.
"""
Perform k-fold evaluation of infrasonic detection and 
categorization methods in TensorFlow 2.0
"""

import os
import json
import numpy as np

from riddl.models import fk
from riddl.utils import data_io


# This function should be rewritten
def default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError('Not serializable')

if __name__ == '__main__':
    ##############
    # Load Data  #
    ##############

    data_id = "soi"
    dir = "./models/train/soi"

    num_folds = 5
    epoch_num = 2000
    batch_size = 32

    X, Y, _ = data_io.load_tensor_data(dir + "/" + data_id)
    kfold_results = fk.run_kfold(X, Y, n_splits=num_folds, epoch_cnt=epoch_num, batch_size=batch_size, model_out="model_out_test")

    with open(data_id + ".kfold_results.json", 'w') as file_out:
        file_out.write(json.dumps(kfold_results, default=default))

    fk.summarize_kfold(kfold_results, drop_minmax=True)
