'''

'''

import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

from src.common import lib_tensor_handler, lib_handler
from src.common import rt_utils_v5
from src.common.constant import ProgressStepEnum, ProgressStepStatusEnum
from src.common.obj import InputParam
from src.utils import frag_rt_matrix_check_utils
from src.utils import instrument_info_utils
from src.utils import msg_send_utils
from src.utils import win_id_utils

IRT_COL_INDEx = 0
RT_COL_INDEx = 1
SUPPY_IRT_RT_NUM = 5


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


#
def random_choose_precursor(lib_data_org, choose_num=50000):
    target_transition_group_id_list = lib_data_org[lib_data_org['decoy'] == 0]['transition_group_id'].unique()
    decoy_transition_group_id_list = lib_data_org[lib_data_org['decoy'] == 1]['transition_group_id'].unique()
    random_choose_target_list = random.choices(target_transition_group_id_list, k=choose_num)
    random_choose_decoy_list = random.choices(decoy_transition_group_id_list, k=choose_num)
    return random_choose_target_list, random_choose_decoy_list


def build_lib_matrix(input_param, lib_cols_org, lib_data_org, random_choose_target_list, random_choose_decoy_list,
                     logger):
    device = input_param.device
    frag_repeat_num = input_param.frag_repeat_num
    #
    #
    peak_precursor_id_list = []
    peak_precursor_id_list.extend(random_choose_target_list)
    peak_precursor_id_list.extend(random_choose_decoy_list)

    lib_cols, lib_data = lib_handler.base_load_lib(lib_cols_org, lib_data_org, peak_precursor_id_list,
                                                   intersection=True)
    t1 = time.time()
    precursor_list, ms1_data_list, ms2_data_list, precursor_info_list = lib_tensor_handler.build_lib_matrix(lib_data,
                                                                                                            lib_cols,
                                                                                                            input_param.run_env,
                                                                                                            None,
                                                                                                            input_param.iso_range,
                                                                                                            input_param.mz_max,
                                                                                                            input_param.max_fragment,
                                                                                                            input_param.n_thread)
    t2 = time.time()
    logger.debug('[TIME COUNT]: timepoint, build_lib_matrix time {}'.format(t2 - t1))
    ms1_data_tensor, ms2_data_tensor = lib_tensor_handler.build_precursors_matrix_step1(ms1_data_list, ms2_data_list,
                                                                                        device=device)
    t3 = time.time()
    logger.debug('[TIME COUNT]: timepoint, build_precursors_matrix_step1 time {}'.format(t3 - t2))
    ms2_data_tensor = lib_tensor_handler.build_precursors_matrix_step2(ms2_data_tensor)
    t4 = time.time()
    logger.debug('[TIME COUNT]: timepoint, build_precursors_matrix_step2 time {}'.format(t4 - t3))
    ms1_data_tensor, ms2_data_tensor, ms1_extract_tensor, ms2_extract_tensor, ms2_mz_tol_half = lib_tensor_handler.build_precursors_matrix_step3_v2(
        ms1_data_tensor, ms2_data_tensor, frag_repeat_num, device=device)
    t5 = time.time()
    logger.debug('[TIME COUNT]: timepoint, build_precursors_matrix_step3_v2 time {}'.format(t5 - t4))
    return precursor_list, precursor_info_list, ms1_data_tensor, ms2_data_tensor, ms1_extract_tensor, ms2_extract_tensor, ms2_mz_tol_half


def peak_group_data_epoch(sc_deal_thread, input_param, mzml_path, ms1, ms2, win_range, lib_cols_org, lib_data_org,
                          logger, irt_rt_data_list, rt_out_file_dir, epoch_num):
    random_choose_target_list, random_choose_decoy_list = random_choose_precursor(lib_data_org,
                                                                                  choose_num=input_param.fitting_rt_num)
    precursor_list, precursor_info_list, ms1_data_tensor, ms2_data_tensor, ms1_extract_tensor, ms2_extract_tensor, ms2_mz_tol_half = build_lib_matrix(
        input_param, lib_cols_org, lib_data_org, random_choose_target_list, random_choose_decoy_list, logger)
    fitting_rt_batch_size = input_param.fitting_rt_batch_size

    all_rsm_list = []
    frag_info_list = []
    irt_rt_pos_list = []
    mzml_name = os.path.split(mzml_path)[-1]
    rawdata_prefix = mzml_name[:-5]

    if epoch_num > 0:
        time_points_file_name = 'time_points_epoch_{}.txt'.format(epoch_num - 1)
        rt_model_params = rt_utils_v5.set_get_rt_model_params(irt_rt_data_list, rt_out_file_dir, draw_pic=input_param.draw_rt_pic,
                                                              file_name=time_points_file_name)
        min_rt, max_rt = get_min_max_rt(input_param, rawdata_prefix, file_name=time_points_file_name)
    else:
        rt_model_params, min_rt, max_rt = None, None, None

    try:
        #
        for start_pos in range(0, len(precursor_list), fitting_rt_batch_size):
            ttt1 = time.time()
            logger.info('pick rt process: {}/{}'.format(start_pos, len(precursor_list)))
            msg_send_utils.send_msg(msg='RT normalization progress: {}/{}'.format(start_pos, len(precursor_list)))
            end_pos = start_pos + fitting_rt_batch_size
            each_precursor_list = precursor_list[start_pos: end_pos]
            each_precursor_info_list = precursor_info_list[start_pos: end_pos]
            each_ms1_data_tensor = ms1_data_tensor[start_pos: end_pos]
            each_ms2_data_tensor = ms2_data_tensor[start_pos: end_pos]
            each_ms1_extract_tensor = ms1_extract_tensor[start_pos: end_pos]
            each_ms2_extract_tensor = ms2_extract_tensor[start_pos: end_pos]

            peak_group_info = TimePointPeakGroupEntity(None, None, 0,
                                                       each_precursor_list,
                                                       each_precursor_info_list, each_ms1_data_tensor,
                                                       each_ms2_data_tensor,
                                                       each_ms1_extract_tensor, each_ms2_extract_tensor, ms1, ms2,
                                                       win_range)
            each_result_irt_rt_pos_list, each_rsm, each_frag_info = deal_peak_step_epoch(input_param, peak_group_info,
                                                                                         logger, rt_model_params,
                                                                                         min_rt, max_rt, epoch_num)
            all_rsm_list.append(each_rsm)
            frag_info_list.append(each_frag_info)
            irt_rt_pos_list.extend(each_result_irt_rt_pos_list)
            ttt2 = time.time()
            logger.debug(
                '[TIME COUNT]: timepoint, one batch {}, peak time: {}'.format(fitting_rt_batch_size, abs(ttt2 - ttt1)))

        #
        score_rsm = torch.concat(all_rsm_list)
        frag_info = torch.concat(frag_info_list)
        mzml_rt = math.ceil(math.ceil(ms1.rt_list[-1]) / 60)
        #
        mzml_rt = instrument_info_utils.get_mzml_nearest_rt(mzml_rt)
        logger.info('mzml rt is: {}'.format(mzml_rt))
        mzml_instrument = input_param.instrument
        mzml_name = os.path.split(mzml_path)[-1]

        precursor_info = torch.tensor(precursor_info_list, dtype=torch.float32)
        precursor_info = precursor_info[:, 0: 5]
        #
        precursor_info[:, 3] = -44.98297
        assay_rt_kept = torch.tensor([[1707.0387]] * len(precursor_list))
        delta_rt_kept = torch.tensor([[126.985392448737]] * len(precursor_list))
        precursor_info = torch.hstack([precursor_info, assay_rt_kept, delta_rt_kept])
        precursor_info = precursor_info.to(input_param.device)

        all_scores_list = sc_deal_thread.calc_score_v2(len(precursor_list), precursor_info, score_rsm, frag_info,
                                                       mzml_rt, mzml_instrument,
                                                       mzml_name)
        all_scores_list = all_scores_list.cpu().tolist()
        # print(all_scores_list)
        #
        score_df = pd.DataFrame([{'transition_group_id': transition_group_id, 'score': all_scores_list[nn],
                                  'label': label, 'irt': irt_rt_pos_list[nn][0], 'rt': irt_rt_pos_list[nn][1]} for
                                 nn, (transition_group_id, label) in enumerate(precursor_list)])
        irt_rt_list, score_df = get_fdr_irt_rt(score_df)
        logger.info('fdr data count {}'.format(len(irt_rt_list)))
        return irt_rt_list, score_df
    except Exception as e:
        logger.exception('Deal peak data exception.')
        msg_send_utils.send_msg(msg='RT normalization exception: {}'.format(e))
        return None, None


def get_fdr_irt_rt(score_df, fdr=0.05):
    score_df = score_df.sort_values(by='score', ascending=False, ignore_index=True)

    score_df['decoy'] = score_df['label']

    target_num = (score_df.decoy == 0).cumsum()
    decoy_num = (score_df.decoy == 1).cumsum()

    target_num[target_num == 0] = 1
    decoy_num[decoy_num == 0] = 1
    score_df['q_value'] = decoy_num / target_num
    score_df['q_value'] = score_df['q_value'][::-1].cummin()

    fdr_05 = score_df[score_df['decoy'] == 0][score_df['q_value'] <= fdr]
    #
    irt_rt_list = fdr_05[['irt', 'rt']].values.tolist()
    if len(irt_rt_list) < 100:
        #
        fdr_10 = score_df[score_df['decoy'] == 0][score_df['q_value'] <= 0.1]
        irt_rt_list = fdr_10[['irt', 'rt']].values.tolist()

    return irt_rt_list, score_df


def deal(sc_deal_thread, input_param: InputParam, lib_cols_org, lib_data_org, mzml_path, ms1, ms2, win_range,
         rt_out_file_dir, logger, epoch_count=2):
    logger.info('Start pick rt precursor, param: {}'.format(json.dumps(input_param.__dict__)))

    #
    irt_rt_data_list = None
    for epoch_num in range(epoch_count):
        tt1 = time.time()
        irt_rt_data_list, score_df = peak_group_data_epoch(sc_deal_thread, input_param, mzml_path, ms1, ms2, win_range,
                                                           lib_cols_org, lib_data_org, logger, irt_rt_data_list,
                                                           rt_out_file_dir,
                                                           epoch_num)
        tt2 = time.time()
        logger.info(f'[timepoint v3] epoch {epoch_num} speed time: {(tt2 - tt1)}')
        #
        time_points_file_name = 'time_points_epoch_{}.txt'.format(epoch_num)
        rt_utils_v5.set_get_rt_model_params(irt_rt_data_list, rt_out_file_dir, draw_pic=True,
                                            file_name=time_points_file_name)
        #
        torch.cuda.empty_cache()
        if len(irt_rt_data_list) == 0:
            return False, []

    return True, irt_rt_data_list


def get_rt_model_params(input_param: InputParam, rawdata_prefix, lib_prefix, lib_cols_org, lib_data_org,
                        mzml_path, ms1,
                        ms2, win_range, logger, sc_deal_thread):
    prt1 = time.time()
    lib = input_param.lib
    lib_path = os.path.split(lib)[0]
    logger.info('start pick rt process...')
    msg_send_utils.send_msg(step=ProgressStepEnum.RT_NORMALIZATION, status=ProgressStepStatusEnum.RUNNING,
                            msg='Processing RT normalization, precursor count is {}'.format(input_param.fitting_rt_num))
    #
    rt_out_file_dir = os.path.join(input_param.out_path, rawdata_prefix, 'peak_rt_v3', '{}'.format(lib_prefix))
    if not os.path.exists(rt_out_file_dir):
        os.makedirs(rt_out_file_dir)

    #
    rt_param_exist_flag = rt_utils_v5.only_check_rt_model_params(rt_out_file_dir)
    if rt_param_exist_flag:
        logger.info('rt param file exist')
        msg_send_utils.send_msg(step=ProgressStepEnum.RT_NORMALIZATION, status=ProgressStepStatusEnum.SUCCESS,
                                msg='RT normalization is exist, {}'.format(rt_out_file_dir))
        return rt_utils_v5.get_rt_model_params(rt_out_file_dir)

    success, irt_rt_data_list = deal(sc_deal_thread, input_param, lib_cols_org, lib_data_org, mzml_path, ms1, ms2,
                                     win_range, rt_out_file_dir, logger, epoch_count=input_param.fitting_rt_epochs)
    prt2 = time.time()
    logger.info('end pick rt process..., time: {}'.format(prt2 - prt1))
    if not success:
        logger.info('error pick rt process')
        msg_send_utils.send_msg(step=ProgressStepEnum.RT_NORMALIZATION, status=ProgressStepStatusEnum.ERROR,
                                msg='RT normalization error, {}'.format(rt_out_file_dir))
        return None

    rt_model_params = rt_utils_v5.set_get_rt_model_params(irt_rt_data_list, rt_out_file_dir,
                                                          draw_pic=input_param.draw_rt_pic)
    msg_send_utils.send_msg(step=ProgressStepEnum.RT_NORMALIZATION, status=ProgressStepStatusEnum.SUCCESS,
                            msg='RT normalization is exist, {}'.format(rt_out_file_dir))
    logger.info('success deal rt peak')
    return rt_model_params


def get_min_max_rt(input_param: InputParam, rawdata_prefix, file_name='time_points.txt'):
    lib = input_param.lib
    lib_path = os.path.split(lib)[0]
    lib_prefix = os.path.split(lib)[-1].split('.')[0]
    rt_out_file_dir = os.path.join(lib_path, 'peak_rt',
                                   '{}_{}'.format(rawdata_prefix, lib_prefix))
    return rt_utils_v5.get_minmax_rt(rt_out_file_dir, file_name=file_name)


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


'''

'''


def deal_peak_step_epoch(input_param, peak_group_info: TimePointPeakGroupEntity, logger, rt_model_params, min_rt,
                         max_rt, epoch_num):
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
    logger.debug('[TIME COUNT]: timepoint, calc win time: {}'.format(calc_win_t2 - calc_win_t1))

    sp_win_t1 = time.time()
    win_id_pos_arr_list = win_id_utils.split_win_id_list(pmt_win_id_list_org.tolist())
    sp_win_t2 = time.time()
    logger.debug('[TIME COUNT]: timepoint, split win time: {}'.format(sp_win_t2 - sp_win_t1))

    all_win_t1 = time.time()

    #
    result_irt_rt_pos_list = []

    rsm_list = []
    frag_info_list = []
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
        logger.debug('[TIME COUNT]: timepoint, build_ms_rt_moz_matrix: {}'.format(build_m_t2 - build_m_t11))
        #
        rt_t1 = time.time()
        ms_rt_list = ms1.rt_list

        #
        rt_pos_arr = np.tile(np.arange(len(ms_rt_list)), (len(each_precursors_list), 1))

        rt_t2 = time.time()
        logger.debug('[TIME COUNT]: timepoint, build rt arr time: {}'.format(rt_t2 - rt_t1))
        # *****************************************************************************
        rt_list_len = len(ms1.rt_list)
        rt_pos_list = rt_pos_arr.tolist()
        ms1_precursors_frag_rt_matrix = peak_one(ms1_frag_moz_matrix_coo_matrix, ms1_moz_rt_matrix, rt_list_len,
                                                 rt_pos_list, len(each_precursors_list), device)

        ms2_precursors_frag_rt_matrix = peak_one(ms2_frag_moz_matrix_coo_matrix, ms2_moz_rt_matrix, rt_list_len,
                                                 rt_pos_list, len(each_precursors_list), device)
        rt_t3 = time.time()
        logger.debug('[TIME COUNT]: timepoint, peak_one: {}'.format(abs(rt_t2 - rt_t3)))

        frag_info = build_frag_info(ms1_data_tensor, ms2_data_tensor, w_p_arr, frag_repeat_num, device)
        rt_t4 = time.time()
        logger.debug('[TIME COUNT]: timepoint, build_frag_info: {}'.format(abs(rt_t4 - rt_t3)))

        ms2_frag_info = frag_info[:, 6:26, :].cpu().numpy()
        #
        non_zero_count_matrix = frag_rt_matrix_check_utils.get_none_zero_more_indices_v3(
            ms2_precursors_frag_rt_matrix, ms2_frag_info, open_smooth=input_param.open_smooth)
        rt_t5 = time.time()
        logger.debug('[TIME COUNT]: timepoint, get_none_zero_more_indices_v3: {}'.format(abs(rt_t4 - rt_t5)))

        #

        #
        if epoch_num > 0:
            precursor_info_np_org = np.array(each_precursor_info_list)
            fitting_rt_data_list = rt_utils_v5.get_rt_limit(precursor_info_np_org[:, 3], rt_model_params, min_rt,
                                                            max_rt)

            thiz_n_cycles = 200

            max_rt = max(ms_rt_list)
            new_ms_rt_list = [-1 * max_rt for _ in range(thiz_n_cycles)]
            new_ms_rt_list.extend(ms_rt_list)
            new_ms_rt_list.extend([3 * max_rt for _ in range(thiz_n_cycles)])
            n_cycle_rt_pos_arr = rt_utils_v5.find_rt_pos(fitting_rt_data_list, new_ms_rt_list,
                                                         thiz_n_cycles)
            n_cycle_rt_pos_arr = n_cycle_rt_pos_arr - thiz_n_cycles
            #
            n_cycle_rt_pos_arr[n_cycle_rt_pos_arr < 0] = 0
            n_cycle_rt_pos_arr[n_cycle_rt_pos_arr > len(ms_rt_list) - 1] = len(ms_rt_list) - 1
            # print(n_cycle_rt_pos_arr)
            #
            non_zero_count_matrix = non_zero_count_matrix[
                np.arange(non_zero_count_matrix.shape[0])[:, None], n_cycle_rt_pos_arr]
        else:
            pass

        non_zero_count_matrix = non_zero_count_matrix.cpu().numpy()
        ddd_matrix = np.max(non_zero_count_matrix, axis=1)
        #
        model_cycles_rt_pos_arr = []
        for row_index, ddd in enumerate(ddd_matrix):
            if epoch_num > 0:
                max_pos_list = np.where(non_zero_count_matrix[row_index] == ddd)[0].tolist()
                max_pos = random.choice(max_pos_list)
                max_pos = n_cycle_rt_pos_arr[row_index][max_pos]
            else:
                max_pos_list = np.where(non_zero_count_matrix[row_index] == ddd)[0].tolist()
                max_pos = random.choice(max_pos_list)

            model_cycles_rt_pos_arr.append(max_pos)
            irt_rt_pos = [each_precursor_info_list[row_index][3], ms_rt_list[max_pos], max_pos]
            result_irt_rt_pos_list.append(irt_rt_pos)
        rt_t6 = time.time()
        logger.debug('[TIME COUNT]: timepoint, non_zero_count_matrix: {}'.format(abs(rt_t6 - rt_t5)))

        top_n_rt_real_pos_np = np.array(model_cycles_rt_pos_arr)
        model_cycles_rt_pos_arr_np = rt_utils_v5.find_rt_pos_by_middle_pos_list(top_n_rt_real_pos_np, rt_list_len,
                                                                                input_param.model_cycles)

        score_precursor_index_np = np.arange(len(ddd_matrix))
        each_parse_frag_rt_matrix_num = input_param.each_parse_frag_rt_matrix_num
        ms1_precursors_frag_rt_matrix = parse_frag_rt_matrix_v4(
            ms1_precursors_frag_rt_matrix
            , model_cycles_rt_pos_arr_np, score_precursor_index_np, each_parse_frag_rt_matrix_num)
        if ms1_precursors_frag_rt_matrix is None:
            continue
        ms2_precursors_frag_rt_matrix = parse_frag_rt_matrix_v4(
            ms2_precursors_frag_rt_matrix
            , model_cycles_rt_pos_arr_np, score_precursor_index_np, each_parse_frag_rt_matrix_num)
        if ms2_precursors_frag_rt_matrix is None:
            continue

        #
        ms1_matrix_shape = ms1_precursors_frag_rt_matrix.shape
        ms2_matrix_shape = ms2_precursors_frag_rt_matrix.shape
        ms1_precursors_frag_rt_matrix = ms1_precursors_frag_rt_matrix.reshape(ms1_matrix_shape[0],
                                                                              input_param.frag_repeat_num,
                                                                              ms1_matrix_shape[
                                                                                  1] // input_param.frag_repeat_num,
                                                                              ms1_matrix_shape[2])
        ms2_precursors_frag_rt_matrix = ms2_precursors_frag_rt_matrix.reshape(ms2_matrix_shape[0],
                                                                              input_param.frag_repeat_num,
                                                                              ms2_matrix_shape[
                                                                                  1] // input_param.frag_repeat_num,
                                                                              ms2_matrix_shape[2])
        #
        ms1_precursors_frag_rt_matrix = ms1_precursors_frag_rt_matrix.transpose(1, 2)
        #
        ms2_precursors_frag_rt_matrix = ms2_precursors_frag_rt_matrix.transpose(1, 2)
        #
        ms1_matrix_shape = ms1_precursors_frag_rt_matrix.shape
        ms2_matrix_shape = ms2_precursors_frag_rt_matrix.shape
        #
        ms1_precursors_frag_rt_matrix = ms1_precursors_frag_rt_matrix.reshape(ms1_matrix_shape[0],
                                                                              ms1_matrix_shape[1] * ms1_matrix_shape[2],
                                                                              ms1_matrix_shape[3])
        #
        ms2_precursors_frag_rt_matrix = ms2_precursors_frag_rt_matrix.reshape(ms2_matrix_shape[0],
                                                                              ms2_matrix_shape[1] * ms2_matrix_shape[2],
                                                                              ms2_matrix_shape[3])
        #
        rsm = torch.cat([ms1_precursors_frag_rt_matrix, ms2_precursors_frag_rt_matrix], dim=1)
        rsm_list.append(rsm)
        frag_info_list.append(frag_info)

    all_win_t2 = time.time()
    logger.debug('[TIME COUNT]: timepoint, all win deal time: {}'.format(all_win_t2 - all_win_t1))
    return result_irt_rt_pos_list, torch.concat(rsm_list), torch.concat(frag_info_list)


def parse_frag_rt_matrix_v4(ms_matrix, model_cycles_rt_pos_arr_np, score_precursor_index_np, each_choose_num):
    all_num = len(score_precursor_index_np)
    choose_matrix = [ms_matrix[score_precursor_index_np[start_pos:start_pos + each_choose_num][:, None], :,
                     model_cycles_rt_pos_arr_np[start_pos:start_pos + each_choose_num]] for start_pos in
                     range(0, all_num, each_choose_num)]
    choose_matrix = torch.concat(choose_matrix, dim=0)
    choose_matrix = choose_matrix.transpose(1, 2)
    return choose_matrix


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
