import os
import pickle
import time

from src.common import drdia_utils
from src.common.constant import ProgressStepEnum, ProgressStepStatusEnum
from src.utils import msg_send_utils

'''
'''


# 加载raw
def load_and_temp_raw(rawdata_file_dir_path, mzml_name, mz_min, mz_max, rt_unit='min', thread_num=10, skip_no_temp=False, logger=None):


    msg_send_utils.send_msg(step=ProgressStepEnum.PARSE_MZML, status=ProgressStepStatusEnum.RUNNING,
                            mzml_name=os.path.join(rawdata_file_dir_path, mzml_name), msg='Processing parse mzML, {}'.format(mzml_name))
    t1 = time.time()
    rawdata_prefix = mzml_name[:-5]
    temp_pkl_dir_path = os.path.join(rawdata_file_dir_path, 'temp_pkl')
    if not os.path.exists(temp_pkl_dir_path):
        os.mkdir(temp_pkl_dir_path)

    if rt_unit == 'min':
        temp_mz_dir_path = os.path.join(temp_pkl_dir_path, '{}_{}'.format(mz_min, mz_max))
    else:
        temp_mz_dir_path = os.path.join(temp_pkl_dir_path, '{}_{}_{}'.format(mz_min, mz_max, rt_unit))
    if not os.path.exists(temp_mz_dir_path):
        os.mkdir(temp_mz_dir_path)

    ms1_pkl_file = os.path.join(temp_mz_dir_path, rawdata_prefix + '_ms1.pickle')
    ms2_pkl_file = os.path.join(temp_mz_dir_path, rawdata_prefix + '_ms2.pickle')
    win_range_pkl_file = os.path.join(temp_mz_dir_path, rawdata_prefix + '_win_range.pickle')
    logger.info('temp path: {}, {}, {}'.format(ms1_pkl_file, ms2_pkl_file, win_range_pkl_file))
    if os.path.exists(ms1_pkl_file) and os.path.exists(ms2_pkl_file) and os.path.exists(win_range_pkl_file):
        logger.info('temp mzml pkl is exist, load pkl')
        msg_send_utils.send_msg(msg='Temp mzML info is exist, load temp file: {}'.format(ms1_pkl_file))
        with open(ms1_pkl_file, 'rb') as f:
            ms1 = pickle.load(f)
        with open(ms2_pkl_file, 'rb') as f:
            ms2 = pickle.load(f)
        with open(win_range_pkl_file, 'rb') as f:
            win_range = pickle.load(f)
        msg_send_utils.send_msg(step=ProgressStepEnum.PARSE_MZML, status=ProgressStepStatusEnum.SUCCESS)
        return ms1, ms2, win_range

    if skip_no_temp:
        logger.info('skip no temp raw, mzml_name = {}'.format(mzml_name))
        return None, None, None
    logger.info('temp mzml pkl is not exist, load_rawdata ')
    msg_send_utils.send_msg(msg='Temp mzML info is not exist, parse mzML')
    try:
        ms1, ms2, win_range = drdia_utils.load_rawdata(os.path.join(rawdata_file_dir_path, mzml_name), mz_min, mz_max, rt_unit, logger=logger)
        with open(ms1_pkl_file, 'wb') as f:
            pickle.dump(ms1, f)
        with open(ms2_pkl_file, 'wb') as f:
            pickle.dump(ms2, f)
        with open(win_range_pkl_file, 'wb') as f:
            pickle.dump(win_range, f)
        t2 = time.time()
        logger.info('load and temp raw time: {}'.format(t2 - t1))
        msg_send_utils.send_msg(step=ProgressStepEnum.PARSE_MZML, status=ProgressStepStatusEnum.SUCCESS)
        return ms1, ms2, win_range
    except Exception as e:
        msg_send_utils.send_msg(step=ProgressStepEnum.PARSE_MZML, status=ProgressStepStatusEnum.ERROR, msg='load_rawdata exception: {}'.format(e))
        logger.exception('load_rawdata exception: ')
    t2 = time.time()
    logger.info('load and temp raw time: {}'.format(t2 - t1))
    return None, None, None
