import glob
import os
import os.path
import pickle
import shutil

import numpy as np
import pandas as pd
from src.utils import msg_send_utils
from src.common.constant import ProgressStepEnum, ProgressStepStatusEnum


def process_parse_rt(base_output_dir, mzml_file_list, logger):
    msg_send_utils.send_msg(step=ProgressStepEnum.QUANTIFICATION, msg='Process parse RT')
    logger.info('Process parse RT')
    feature_dir = os.path.join(base_output_dir, 'feature')
    if os.path.exists(feature_dir):
        shutil.rmtree(feature_dir)

    os.makedirs(feature_dir)

    feature_list = ['sequence_length', 'precursor_mz', 'charge', 'precursor_irt', 'nr_peaks',
                    'assay_rt_kept', 'delta_rt_kept']

    for dd, mzml_name in enumerate(mzml_file_list):
        logger.info('Process parse RT, {}/{}'.format(dd, len(mzml_file_list)))
        rawdata_prefix = mzml_name[:-5]
        construct_data_set(base_output_dir,
                           rawdata_prefix,
                           feature_dir,
                           feature_list, logger)
    logger.info('Finished parse RT')
    msg_send_utils.send_msg(step=ProgressStepEnum.QUANTIFICATION, msg='Finished parse RT')


def construct_data_set(base_output_dir,
                       raw_prefix,
                       feat_dir,
                       feature_list, logger):
    data_path = os.path.join(base_output_dir, raw_prefix, 'identify_data')
    pkl_list = glob.glob(f'{data_path}/*.pkl')
    logger.info('data_path: {}, total pkl: {}'.format(data_path, len(pkl_list)))

    file_precursor_id, file_precursor_feat = [], []

    for chrom_file in pkl_list:
        f = open(chrom_file, "rb")
        precursor_data = pickle.load(f)
        f.close()
        precursor, precursor_feat, _, _, _ = precursor_data

        precursor = np.array(precursor)
        precursor_id = precursor[:, 0].tolist()
        file_precursor_id.extend(precursor_id)  # precursor_id
        file_precursor_feat.append(precursor_feat)  # precursor_feat

    file_precursor_feat = np.concatenate(file_precursor_feat, axis=0)
    logger.info('file: %s, precursor num: %s, precursor_feat: %s' % (
        raw_prefix, len(file_precursor_id), file_precursor_feat.shape))

    # gen df
    df = pd.DataFrame(file_precursor_feat, index=file_precursor_id, columns=feature_list)
    df["filename"] = raw_prefix

    # save feature
    file_dir = os.path.join(feat_dir, "{}_feature.tsv".format(raw_prefix))
    df.to_csv(file_dir, sep="\t")
    logger.info('feature save to: %s' % file_dir)
