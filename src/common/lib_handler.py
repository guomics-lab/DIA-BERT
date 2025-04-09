import os
import pickle
import time

import numpy as np

from src.common import decoy_generator
from src.common import drdia_utils
from src.common_logger import logger
from src.utils import msg_send_utils
from src.common.constant import ProgressStepEnum, ProgressStepStatusEnum

replacement_dict = {'b': 1, 'y': 2, 'p': 3}

'''
'''


def base_load_lib(lib_cols, library, diann_match_p_list, intersection=True):
    t1 = time.time()
    pr_id_count = library[lib_cols["PRECURSOR_ID_COL"]].nunique()
    if intersection:
        logger.info("before filter, lib pr_id count: {}, diann count: {}, common pr_id count: {}".
                    format(pr_id_count,
                           len(diann_match_p_list),
                           len(set(library[lib_cols["PRECURSOR_ID_COL"]]) & set(diann_match_p_list))))
        library = library[library[lib_cols["PRECURSOR_ID_COL"]].isin(diann_match_p_list)]
    else:
        logger.info("lib pr_id count: {}". format(pr_id_count))
    lib_data = library.sort_values(
        by=[lib_cols['PRECURSOR_MZ_COL'], lib_cols['PRECURSOR_ID_COL'], lib_cols['LIB_INTENSITY_COL']],
        ascending=[True, True, False]).reset_index(drop=True)

    lib_data[lib_cols['FRAGMENT_TYPE_COL']] = lib_data[lib_cols['FRAGMENT_TYPE_COL']].replace(replacement_dict)
    t2 = time.time()
    logger.info('replace: {}'.format(t2 - t1))
    return lib_cols, lib_data


def calc_pr_all(library, lib_cols):
    library[lib_cols["PRECURSOR_ID_COL"]] = np.where(library['decoy'] == 0,
                                                     library[lib_cols['FULL_SEQUENCE_COL']] + library[
                                                         lib_cols['PRECURSOR_CHARGE_COL']].astype(str),

                                                     'DECOY_' + library[lib_cols['FULL_SEQUENCE_COL']] + library[lib_cols['PRECURSOR_CHARGE_COL']].astype(str).sp
                                                     )
    return library


def calc_pr(transition_group_id, decoy):
    if decoy == 0:
        return ''.join(transition_group_id.split('_')[-2:])
    else:
        return 'DECOY_' + ''.join(transition_group_id.split('_')[-2:])

