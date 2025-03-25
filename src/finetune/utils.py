import logging
import os
import random

import numpy as np
import torch
from sklearn import metrics


def set_seeds(seed):
    logging.info('Unified seeds !!')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    #
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mkdir_p(dirs):
    """
    make a directory (dir) if it doesn't exist
    """
    if not os.path.exists(dirs):
        try:
            #
            os.makedirs(dirs)
        except:
            pass

    return True, 'OK'


def get_prophet_result(df):
    df = df.sort_values(by='score', ascending=False, ignore_index=True)
    df['decoy'] = np.where(df['label'] == 1, 0, 1)

    target_num = (df.decoy == 0).cumsum()
    decoy_num = (df.decoy == 1).cumsum()

    target_num[target_num == 0] = 1
    decoy_num[decoy_num == 0] = 1
    df['q_value'] = decoy_num / target_num
    df['q_value'] = df['q_value'][::-1].cummin()

    # log
    id_10 = ((df['q_value'] <= 0.1) & (df['decoy'] == 0)).sum()
    id_01 = ((df['q_value'] <= 0.01) & (df['decoy'] == 0)).sum()

    #  conservative FDR estimate
    filtered_df = df[(df['q_value'] <= 0.01)][
        ['transition_group_id', 'score', 'label', 'decoy', 'q_value', 'file_name', 'iRT', 'RT']]
    return filtered_df, id_10, id_01

def eval_predict(predictions, targets):
    fpr, tpr, threshold = metrics.roc_curve(targets, predictions, pos_label=1)
    auc_results = metrics.auc(fpr, tpr)
    round_pred = np.round(predictions)
    correct_count = 0
    for i in range(len(round_pred)):
        if round_pred[i] == targets[i]:
            correct_count += 1
    correctness = float(correct_count) / len(round_pred)
    return auc_results, correctness
