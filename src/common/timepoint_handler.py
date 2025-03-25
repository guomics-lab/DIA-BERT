'''
'''

import json
import math
import os
import pickle
import random
import time
from collections import Counter

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from src.common import lib_tensor_handler, lib_handler
from src.common import rt_utils_v5
from src.common.constant import ProgressStepEnum, ProgressStepStatusEnum
from src.common.obj import InputParam
from src.utils import frag_rt_matrix_check_utils
from src.utils import msg_send_utils
from src.utils import win_id_utils

IRT_COL_INDEx = 0
RT_COL_INDEx = 1
SUPPY_IRT_RT_NUM = 5


#
def load_diann_precursor_count():
    with open('./resource/precursor_count.pkl', 'rb') as f:
        precursor_count_list = pickle.load(f)
    #
    return precursor_count_list


def peak_precursor(lib_cols_org, lib_data_org, fitting_rt_num, logger):
    t1 = time.time()
    logger.debug('start peak {} rt precursor'.format(fitting_rt_num))
    #
    all_lib_precursor_set = set(lib_data_org[lib_cols_org["PRECURSOR_ID_COL"]].unique().tolist())
    precursor_count_list = load_diann_precursor_count()
    peak_precursor_id_list = []
    for precursor_info in precursor_count_list:
        p_name = precursor_info[0]
        if p_name in all_lib_precursor_set:
            peak_precursor_id_list.append(p_name)
            if len(peak_precursor_id_list) > fitting_rt_num:
                break
    t2 = time.time()
    logger.debug('peak {} rt precursor time: {}'.format(len(peak_precursor_id_list), t2 - t1))
    return peak_precursor_id_list


#
def load_deal_precursor(input_param, lib_cols_org, lib_data_org, logger):
    device = input_param.device

    frag_repeat_num = input_param.frag_repeat_num
    #
    peak_precursor_id_list = peak_precursor(lib_cols_org, lib_data_org, input_param.fitting_rt_num, logger)
    # peak_precursor_id_list = ['ALHGSWFDGK3']
    #
    #

    lib_cols, lib_data = lib_handler.base_load_lib(lib_cols_org, lib_data_org, peak_precursor_id_list,
                                                   intersection=True)

    precursor_list, ms1_data_list, ms2_data_list, precursor_info_list = lib_tensor_handler.build_lib_matrix(lib_data,
                                                                                                            lib_cols,
                                                                                                            input_param.run_env,
                                                                                                            None,
                                                                                                            input_param.iso_range,
                                                                                                            input_param.mz_max,
                                                                                                            input_param.max_fragment,
                                                                                                            input_param.n_thread)
    ms1_data_tensor, ms2_data_tensor = lib_tensor_handler.build_precursors_matrix_step1(ms1_data_list, ms2_data_list,
                                                                                        device=device)
    ms2_data_tensor = lib_tensor_handler.build_precursors_matrix_step2(ms2_data_tensor)
    ms1_data_tensor, ms2_data_tensor, ms1_extract_tensor, ms2_extract_tensor, ms2_mz_tol_half = lib_tensor_handler.build_precursors_matrix_step3_v2(
        ms1_data_tensor, ms2_data_tensor, frag_repeat_num, device=device)

    return precursor_list, precursor_info_list, ms1_data_tensor, ms2_data_tensor, ms1_extract_tensor, ms2_extract_tensor, ms2_mz_tol_half


class TimePointPeakGroupEntity(object):
    def __init__(self, mzml_name, rawdata_prefix, each_num, precursor_list, precursor_info_list,
                 ms1_data_tensor, ms2_data_tensor,
                 ms1_extract_tensor, ms2_extract_tensor,
                 ms1, ms2,
                 win_range):
        self.mzml_name = mzml_name
        self.rawdata_prefix = rawdata_prefix
        self.precursor_list = precursor_list
        self.each_num = each_num
        self.precursor_info_list = precursor_info_list
        self.ms1_data_tensor = ms1_data_tensor
        self.ms2_data_tensor = ms2_data_tensor
        self.ms1_extract_tensor = ms1_extract_tensor
        self.ms2_extract_tensor = ms2_extract_tensor

        self.ms1 = ms1
        self.ms2 = ms2
        self.win_range = win_range


def deal(input_param: InputParam, lib_cols_org, lib_data_org, mzml_path, ms1, ms2, win_range, logger):
    logger.info('Start pick rt precursor, param: {}'.format(json.dumps(input_param.__dict__)))

    fitting_rt_batch_size = input_param.fitting_rt_batch_size
    precursor_list, precursor_info_list, ms1_data_tensor, ms2_data_tensor, ms1_extract_tensor, ms2_extract_tensor, ms2_mz_tol = load_deal_precursor(
        input_param, lib_cols_org, lib_data_org, logger)

    a_t1 = time.time()

    pick_precursor_list = []
    mt1 = time.time()
    mzml_name = os.path.split(mzml_path)[-1]
    rawdata_prefix = mzml_name[:-5]
    logger.info('start deal file {}.'.format(mzml_name))
    success = True
    try:
        #
        for start_pos in range(0, len(precursor_list), fitting_rt_batch_size):
            logger.info('pick rt process: {}/{}'.format(start_pos, len(precursor_list)))
            # msg_send_utils.send_msg(msg='RT normalization progress: {}/{}'.format(start_pos, len(precursor_list)))
            end_pos = start_pos + fitting_rt_batch_size
            each_precursor_list = precursor_list[start_pos: end_pos]
            each_precursor_info_list = precursor_info_list[start_pos: end_pos]
            each_ms1_data_tensor = ms1_data_tensor[start_pos: end_pos]
            each_ms2_data_tensor = ms2_data_tensor[start_pos: end_pos]
            each_ms1_extract_tensor = ms1_extract_tensor[start_pos: end_pos]
            each_ms2_extract_tensor = ms2_extract_tensor[start_pos: end_pos]

            peak_group_info = TimePointPeakGroupEntity(mzml_name, rawdata_prefix, 0,
                                                       each_precursor_list,
                                                       each_precursor_info_list, each_ms1_data_tensor,
                                                       each_ms2_data_tensor,
                                                       each_ms1_extract_tensor, each_ms2_extract_tensor, ms1, ms2,
                                                       win_range)
            batch_result_list = deal_peak(input_param, peak_group_info, logger)
            pick_precursor_list.extend(batch_result_list)
    except Exception as e:
        logger.exception('mzml: {} deal exception.'.format(mzml_name))
        msg_send_utils.send_msg(msg='RT normalization exception: {}'.format(e))
        success = False
    mt2 = time.time()
    logger.info('each mzml deal time is: {}, precursor num: {}'.format(mt2 - mt1, len(precursor_list)))
    a_t2 = time.time()
    logger.info('Done， all time is: {}'.format(a_t2 - a_t1))
    return success, pick_precursor_list


def get_rt_model_params(input_param: InputParam, rawdata_prefix, lib_prefix, lib_cols_org, lib_data_org, mzml_path, ms1,
                        ms2, win_range, logger):
    prt1 = time.time()
    lib = input_param.lib
    lib_path = os.path.split(lib)[0]
    logger.info('start pick rt process...')
    msg_send_utils.send_msg(step=ProgressStepEnum.RT_NORMALIZATION, status=ProgressStepStatusEnum.RUNNING,
                            msg='Processing RT normalization, precursor count is {}'.format(input_param.fitting_rt_num))


    rt_out_file_dir = os.path.join(input_param.out_path, rawdata_prefix, 'peak_rt_v1', '{}'.format(lib_prefix))
    if not os.path.exists(rt_out_file_dir):
        os.makedirs(rt_out_file_dir)

    #
    # rt_param_exist_flag = rt_utils_v5.only_check_rt_model_params(rt_out_file_dir)
    # if rt_param_exist_flag:
    #     logger.info('rt param file exist')
    #     msg_send_utils.send_msg(step=ProgressStepEnum.RT_NORMALIZATION, status=ProgressStepStatusEnum.SUCCESS,
    #                             msg='RT normalization is exist, {}'.format(rt_out_file_dir))
    #     return rt_utils_v5.get_rt_model_params(rt_out_file_dir)

    success, pick_precursor_list = deal(input_param, lib_cols_org, lib_data_org, mzml_path, ms1, ms2, win_range, logger)
    prt2 = time.time()
    logger.info('end pick rt process..., time: {}'.format(prt2 - prt1))
    if not success:
        logger.info('error pick rt process')
        msg_send_utils.send_msg(step=ProgressStepEnum.RT_NORMALIZATION, status=ProgressStepStatusEnum.ERROR,
                                msg='RT normalization error, {}'.format(rt_out_file_dir))
        return None
    with open(os.path.join(rt_out_file_dir, 'all_pick_precursor_list.pkl'), mode='wb') as f:
        pickle.dump(pick_precursor_list, f)
    #
    pick_precursor_list_np = np.array(pick_precursor_list)
    if len(pick_precursor_list_np) == 0:
        return None
    irt_rt_matrix = pick_precursor_list_np[:, :2]
    #
    # save_data(irt_rt_matrix.tolist(), rt_out_file_dir)

    all_irt_rt_data_list = irt_rt_matrix.tolist()
    all_rt_list = irt_rt_matrix[:, RT_COL_INDEx].tolist()
    #
    rt_irt_list_dict = {}
    for each_irt, each_rt in all_irt_rt_data_list:
        rt_irt_list_dict.setdefault(each_rt, []).append(each_irt)

    rt_count_set = Counter(all_rt_list)
    count_num_arr = []
    delete_rt_list = []
    for each_rt, rt_count in rt_count_set.items():
        if rt_count > 500:
            delete_rt_list.append(each_rt)
        else:
            count_num_arr.append(rt_count)
    #
    count_num_arr.sort()
    median_num = int(np.median(count_num_arr))

    new_irt_rt_list = []
    for each_rt, each_irt_list in rt_irt_list_dict.items():
        if len(each_irt_list) > 500:
            continue
        elif len(each_irt_list) > median_num:
            each_choose_irt_list = random.choices(each_irt_list, k=median_num)
        else:
            each_choose_irt_list = each_irt_list
        new_irt_rt_list.extend([[each_irt, each_rt] for each_irt in each_choose_irt_list])
    irt_rt_matrix = np.array(new_irt_rt_list)

    min_assay_rt = min(irt_rt_matrix[:, 1])
    max_assay_rt = max(irt_rt_matrix[:, 1])

    group_num = 10
    n_neighbors = 9
    irt_rt_matrix_arr = []
    irt_rt_matrix_count_arr = []
    assay_rt_group = calc_assay_rt_group(min_assay_rt, max_assay_rt, group_num)
    for each_min_assay_rt, each_max_assay_rt in assay_rt_group:
        each_irt_rt_matrix = []
        for irt, assay_rt in irt_rt_matrix:
            if assay_rt > each_min_assay_rt and assay_rt <= each_max_assay_rt:
                each_irt_rt_matrix.append([irt, assay_rt])
        irt_rt_matrix_arr.append(each_irt_rt_matrix)
        irt_rt_matrix_count_arr.append(len(each_irt_rt_matrix))

    pick_rt_data = []
    for each_group_index, each_irt_rt_matrix in enumerate(irt_rt_matrix_arr):
        if len(each_irt_rt_matrix) < n_neighbors:
            pick_rt_data.extend(each_irt_rt_matrix)
            continue
        irt_rt_matrix_np = np.array(each_irt_rt_matrix)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(irt_rt_matrix_np)
        distances, indices = nbrs.kneighbors(irt_rt_matrix_np)
        #
        sum_distances = np.sum(distances, axis=1).reshape(-1, 1)
        distance_matrix = np.column_stack([indices[:, 0], sum_distances])
        #
        distance_matrix = distance_matrix[np.argsort(distance_matrix[:, 1])]
        irt_rt_array_index = distance_matrix[:, 0].astype(np.int16)
        #
        irt_rt_distance_matrix = np.column_stack([irt_rt_matrix_np[irt_rt_array_index], distance_matrix[:, 1]])
        #
        choose_top_m = int(len(irt_rt_distance_matrix) * 0.05)
        each_choose_irt_rt = irt_rt_distance_matrix[:choose_top_m, :2].tolist()
        pick_rt_data.extend(each_choose_irt_rt)

    pick_rt_data_np = np.array(pick_rt_data)
    limit_max_rt = np.max(pick_rt_data_np[:, RT_COL_INDEx]) * 0.9
    #
    pick_rt_data = filter_irt_rt(pick_rt_data, limit_max_rt=limit_max_rt, logger=logger)

    rt_model_params = rt_utils_v5.set_get_rt_model_params(pick_rt_data, rt_out_file_dir,
                                                          draw_pic=input_param.draw_rt_pic)
    msg_send_utils.send_msg(step=ProgressStepEnum.RT_NORMALIZATION, status=ProgressStepStatusEnum.SUCCESS,
                            msg='RT normalization is exist, {}'.format(rt_out_file_dir))
    return rt_model_params


def get_min_max_rt(input_param: InputParam, rawdata_prefix):
    lib = input_param.lib
    lib_path = os.path.split(lib)[0]
    lib_prefix = os.path.split(lib)[-1].split('.')[0]
    rt_out_file_dir = os.path.join(lib_path, 'peak_rt',
                                   '{}_{}'.format(rawdata_prefix, lib_prefix))
    return rt_utils_v5.get_minmax_rt(rt_out_file_dir)


def calc_assay_rt_group(min_assay_rt, max_assay_rt, group_num=10):
    min_assay_rt = min_assay_rt - 1
    max_assay_rt = max_assay_rt + 1
    each_width = (max_assay_rt - min_assay_rt) / group_num
    return [[round(min_assay_rt + index * each_width, 2), round(min_assay_rt + (index + 1) * each_width, 2)] for index
            in
            range(group_num)]


def filter_irt_rt(irt_rt_list, limit_max_irt=100, limit_max_rt=1000.0, logger=None):
    logger.info('filter_irt_rt, limit_max_irt: {}, limit_max_rt: {}'.format(limit_max_irt, limit_max_rt))
    use_irt_rt_list = []
    for irt, rt in irt_rt_list:
        if float(irt) < limit_max_irt and float(rt) > limit_max_rt:
            continue
        use_irt_rt_list.append((irt, rt))
    logger.info('filter_irt_rt result, org data num：{}，now data num：{}'.format(len(irt_rt_list), len(use_irt_rt_list)))
    return use_irt_rt_list


def calc_irt_group(min_irt, max_irt, group_num=10):
    min_irt = min_irt - 1
    max_irt = max_irt + 1
    each_width = (max_irt - min_irt) / 10
    return [[round(min_irt + index * each_width, 2), round(min_irt + (index + 1) * each_width, 2)] for index in
            range(group_num)]


def get_each_group_top_n(irt_rt_distance_matrix, irt_group_arr, all_choose_num=1000):
    choose_irt_list = []
    choose_rt_list = []
    all_irt_num = len(irt_rt_distance_matrix)
    for min_rt, max_rt in irt_group_arr:
        each_matrix = irt_rt_distance_matrix[
            (irt_rt_distance_matrix[:, 0] >= min_rt) & (irt_rt_distance_matrix[:, 0] < max_rt)]
        each_choose_num = math.ceil(len(each_matrix) * all_choose_num / all_irt_num)
        each_choose_matrix = each_matrix[: each_choose_num]
        choose_irt_list.extend(each_choose_matrix[:, 0].tolist())
        choose_rt_list.extend(each_choose_matrix[:, 1].tolist())

    return choose_irt_list, choose_rt_list


'''

'''


def clac_knn(irt_rt_matrix, min_irt, max_irt):
    nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(irt_rt_matrix)
    distances, indices = nbrs.kneighbors(irt_rt_matrix)
    #
    sum_distances = np.sum(distances, axis=1).reshape(-1, 1)
    distance_matrix = np.column_stack([indices[:, 0], sum_distances])
    #
    distance_matrix = distance_matrix[np.argsort(distance_matrix[:, 1])]
    #
    irt_rt_array_index = distance_matrix[:, 0].astype(np.int16)
    #
    irt_rt_distance_matrix = np.column_stack([irt_rt_matrix[irt_rt_array_index], distance_matrix[:, 1]])
    #
    irt_group_arr = calc_irt_group(min_irt, max_irt)
    choose_irt_list, choose_rt_list = get_each_group_top_n(irt_rt_distance_matrix, irt_group_arr)
    #
    pick_rt_data_np = np.zeros((len(choose_irt_list), 2))
    pick_rt_data_np[:, IRT_COL_INDEx] = choose_irt_list
    pick_rt_data_np[:, RT_COL_INDEx] = choose_rt_list
    return pick_rt_data_np


def save_data(irt_rt_list, out_file_dir):
    irt_list, rt_list, = [], []
    with open(os.path.join(out_file_dir, 'time_points_all.txt'), 'w+') as f:
        for irt, rt in irt_rt_list:
            irt_list.append(float(irt))
            rt_list.append(float(rt))
            f.write('%.5f\t%.2f\n' % (float(irt), float(rt)))


def deal_peak(input_param, peak_group_info: TimePointPeakGroupEntity, logger):
    mz_max = input_param.mz_max
    device = input_param.device
    frag_repeat_num = input_param.frag_repeat_num

    precursor_list = peak_group_info.precursor_list
    precursor_info_list = peak_group_info.precursor_info_list
    ms1_data_tensor = peak_group_info.ms1_data_tensor
    ms2_data_tensor = peak_group_info.ms2_data_tensor
    ms1_extract_tensor = peak_group_info.ms1_extract_tensor
    ms2_extract_tensor = peak_group_info.ms2_extract_tensor
    ms1 = peak_group_info.ms1
    ms2 = peak_group_info.ms2
    win_range = peak_group_info.win_range

    calc_win_t1 = time.time()
    pmt_win_id_list_org = lib_tensor_handler.calc_win_id(ms2_data_tensor, win_range)
    calc_win_t2 = time.time()
    logger.debug('calc win time: {}'.format(calc_win_t2 - calc_win_t1))

    sp_win_t1 = time.time()
    win_id_pos_arr_list = win_id_utils.split_win_id_list(pmt_win_id_list_org.tolist())
    sp_win_t2 = time.time()
    logger.debug('split win time: {}'.format(sp_win_t2 - sp_win_t1))

    all_win_t1 = time.time()

    #

    result_list = []
    for pos_index, w_p_arr in enumerate(win_id_pos_arr_list):
        #
        pmt_win_id_list = pmt_win_id_list_org[w_p_arr[0]: w_p_arr[1]]
        each_precursors_list = precursor_list[w_p_arr[0]: w_p_arr[1]]
        each_precursor_info_list = precursor_info_list[w_p_arr[0]: w_p_arr[1]]

        #
        build_m_t11 = time.time()
        ms1_moz_rt_matrix, ms2_moz_rt_matrix, ms1_frag_moz_matrix_coo_matrix, ms2_frag_moz_matrix_coo_matrix = \
            lib_tensor_handler.build_ms_rt_moz_matrix(ms1_extract_tensor[w_p_arr[0]: w_p_arr[1]],
                                                      ms2_extract_tensor[w_p_arr[0]: w_p_arr[1]], pmt_win_id_list,
                                                      mz_max, ms1,
                                                      ms2, device)

        build_m_t2 = time.time()
        logger.debug('build matrix time: {}'.format(build_m_t2 - build_m_t11))
        #
        rt_t1 = time.time()
        ms_rt_list = ms1.rt_list

        #
        rt_pos_arr = np.tile(np.arange(len(ms_rt_list)), (len(each_precursors_list), 1))

        rt_t2 = time.time()
        logger.debug('build rt arr time: {}'.format(rt_t2 - rt_t1))
        # *****************************************************************************
        rt_list_len = len(ms1.rt_list)
        rt_pos_list = rt_pos_arr.tolist()
        ms2_precursors_frag_rt_matrix = peak_one(ms2_frag_moz_matrix_coo_matrix, ms2_moz_rt_matrix, rt_list_len,
                                                 rt_pos_list, len(each_precursors_list), device)

        frag_info = build_frag_info(ms1_data_tensor, ms2_data_tensor, w_p_arr, frag_repeat_num, device)

        ms2_frag_info = frag_info[:, 6:26, :].cpu().numpy()
        #
        non_zero_count_matrix = frag_rt_matrix_check_utils.get_none_zero_more_indices_v3(
            ms2_precursors_frag_rt_matrix, ms2_frag_info, open_smooth=input_param.open_smooth)

        non_zero_count_matrix = non_zero_count_matrix.cpu().numpy()
        ddd_matrix = np.max(non_zero_count_matrix, axis=1)
        #
        for row_index, ddd in enumerate(ddd_matrix):
            max_pos_list = np.where(non_zero_count_matrix[row_index] == ddd)[0].tolist()
            max_pos = random.choice(max_pos_list)
            irt_rt = [each_precursor_info_list[row_index][3], ms_rt_list[max_pos]]
            result_list.append(irt_rt)
    all_win_t2 = time.time()
    logger.debug('[time peak group]all win deal time: {}'.format(all_win_t2 - all_win_t1))
    return result_list


def build_frag_info(ms1_data_tensor, ms2_data_tensor, w_p_arr, frag_repeat_num, device):
    # #
    ext_ms1_precursors_frag_rt_matrix = lib_tensor_handler.build_ext_ms1_matrix(
        ms1_data_tensor[w_p_arr[0]: w_p_arr[1]], device)
    ext_ms2_precursors_frag_rt_matrix = lib_tensor_handler.build_ext_ms2_matrix(
        ms2_data_tensor[w_p_arr[0]: w_p_arr[1]], device)

    ms1_ext_shape = ext_ms1_precursors_frag_rt_matrix.shape
    ms2_ext_shape = ext_ms2_precursors_frag_rt_matrix.shape

    ext_ms1_precursors_frag_rt_matrix = ext_ms1_precursors_frag_rt_matrix.reshape(ms1_ext_shape[0],
                                                                                  frag_repeat_num,
                                                                                  ms1_ext_shape[
                                                                                      1] // frag_repeat_num,
                                                                                  ms1_ext_shape[2]).cpu()
    ext_ms2_precursors_frag_rt_matrix = ext_ms2_precursors_frag_rt_matrix.reshape(ms2_ext_shape[0],
                                                                                  frag_repeat_num,
                                                                                  ms2_ext_shape[
                                                                                      1] // frag_repeat_num,
                                                                                  ms2_ext_shape[2]).cpu()
    frag_info = torch.cat([ext_ms1_precursors_frag_rt_matrix, ext_ms2_precursors_frag_rt_matrix], dim=2)
    frag_info = frag_info[:, 0, :, :]
    return frag_info


def peak_one(ms_frag_moz_matrix_coo_matrix, ms_moz_rt_matrix, rt_list_len, ms_rt_pos_list, each_precursors_list_length,
             device):
    ms_frag_rt_matrix_result = torch.matmul(ms_frag_moz_matrix_coo_matrix.to(device), ms_moz_rt_matrix.to(device))
    ms_frag_rt_matrix_result = lib_tensor_handler.adjust_diagonal_matrix(ms_frag_rt_matrix_result, rt_list_len)

    frag_rt_matrix_result = ms_frag_rt_matrix_result.reshape(each_precursors_list_length,
                                                             int(ms_frag_rt_matrix_result.shape[
                                                                     0] / each_precursors_list_length),
                                                             ms_frag_rt_matrix_result.shape[1])

    return frag_rt_matrix_result
