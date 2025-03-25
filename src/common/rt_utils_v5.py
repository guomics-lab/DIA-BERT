import numpy as np

from src.common import rt_normalization
from src.common_logger import logger

'''

'''


def get_rt(irt_list, rt_model_params):
    return rt_model_params.predict(irt_list)


def get_rt_limit(irt_list, rt_model_params, min_rt, max_rt):
    fitting_rt_data_list = rt_model_params.predict(irt_list)
    fitting_rt_data_list = fitting_rt_data_list.reshape(-1, 1)
    if min_rt is not None:
        fitting_rt_data_list[fitting_rt_data_list < min_rt] = min_rt
    if max_rt is not None:
        fitting_rt_data_list[fitting_rt_data_list > max_rt] = max_rt
    return fitting_rt_data_list


def get_rt_model_params(rt_norm_dir):
    try:
        return rt_normalization.fit_irt_model_by_data_v5(None, rt_norm_dir, True)
    except Exception:
        logger.exception('get_rt_model_params exception, {}'.format(rt_norm_dir))
    return None


def only_check_rt_model_params(out_file_dir):
    data_count = rt_normalization.read_rt_model_params(out_file_dir)
    if data_count >= 10:
        return True
    return False


def get_minmax_rt(out_file_dir, file_name='time_points.txt'):
    return rt_normalization.get_minmax_rt(out_file_dir, file_name)


'''
'''


def set_get_rt_model_params(pick_rt_data, out_file_dir, draw_pic, file_name='time_points.txt'):
    try:
        return rt_normalization.fit_irt_model_by_data_v5(pick_rt_data, out_file_dir, draw_pic, file_name)
    except Exception:
        logger.exception('get_rt_model_params_by_rt_data exception')
    return None


def build_irt_arr_v2(irt_list, ms_rt_list, n_cycles, model_cycles, rt_norm_model, rt_model_params, min_rt, max_rt):

    fitting_rt_data_list = get_rt(np.array(irt_list), rt_norm_model, rt_model_params)
    if min_rt is not None:
        fitting_rt_data_list[fitting_rt_data_list < min_rt] = min_rt
    if max_rt is not None:
        fitting_rt_data_list[fitting_rt_data_list > max_rt] = max_rt
    rt_pos_arr = find_rt_pos(fitting_rt_data_list, ms_rt_list, n_cycles)
    middle_post = model_cycles // 2
    middle_post_arr = rt_pos_arr[:, middle_post]
    assay_rt_kept = np.array(ms_rt_list)[middle_post_arr]
    delta_rt_kept = fitting_rt_data_list - assay_rt_kept
    return assay_rt_kept, delta_rt_kept


'''
'''


def build_irt_arr_all(precursor_info_list, ms_rt_list, model_cycles):
    n_precursor = len(precursor_info_list)
    n_cycles = len(ms_rt_list)
    # p * len(ms_rt_list)
    rt_pos_arr = np.tile(np.arange(len(ms_rt_list)), (n_precursor, 1))
    # 滑动获取
    middle_post_arr = np.lib.stride_tricks.sliding_window_view(rt_pos_arr, model_cycles, axis=1)
    middle_post = model_cycles // 2
    middle_post_arr = middle_post_arr[:, :, middle_post]
    assay_rt_kept = np.array(ms_rt_list)[middle_post_arr]
    # p * (n_cycles - model_cycles + 1)
    delta_rt_kept = np.full((n_precursor, (n_cycles - model_cycles + 1)), 20)
    return rt_pos_arr, assay_rt_kept, delta_rt_kept


'''
'''


def build_rt_arr_peak_group(precursor_info_list, ms_rt_list, model_cycles, rt_norm_model, rt_model_params):
    #
    rt_data_list = get_rt(np.array(precursor_info_list)[:, 3], rt_norm_model, rt_model_params)

    #
    ms1_rt_pos_arr = find_rt_pos(rt_data_list, ms_rt_list, model_cycles)
    ms2_rt_pos_arr = find_rt_pos(rt_data_list, ms_rt_list, model_cycles)

    middle_post = model_cycles // 2
    middle_post_arr = ms1_rt_pos_arr[:, middle_post]
    assay_rt_kept = np.array(ms_rt_list)[middle_post_arr]
    delta_rt_kept = rt_data_list - assay_rt_kept
    return ms1_rt_pos_arr, ms2_rt_pos_arr, assay_rt_kept, delta_rt_kept


'''
'''


def find_rt_pos(RT_list, rt_list, n_cycles, shifting_pos=False, shifting_pos_type=1):
    len_rt_list = len(rt_list)
    middle_pos_list = np.argmin(np.abs(np.tile(rt_list, (len(RT_list), 1)) - np.array(RT_list).reshape(-1, 1)), axis=1)
    if shifting_pos:
        if shifting_pos_type == 1:
            logger.info('deal shifting_pos, +- [-1, 1]')
            shifting_pos_data = np.random.randint(-1, 1, len(middle_pos_list))
            shifting_pos_data[shifting_pos_data == 0] = 1
            #
            middle_pos_list = np.add(middle_pos_list, shifting_pos_data)
        elif shifting_pos_type == 2:
            logger.info('deal shifting_pos, +- [2, 20]')
            shifting_pos_data = np.random.randint(-20, 18, len(middle_pos_list))
            shifting_pos_data[shifting_pos_data == -1] = 20
            shifting_pos_data[shifting_pos_data == 0] = 19
            shifting_pos_data[shifting_pos_data == 1] = 18
            #
            middle_pos_list = np.add(middle_pos_list, shifting_pos_data)
        elif shifting_pos_type == 3:
            # rt + 1
            logger.info('deal shifting_pos, +1 ')
            middle_pos_list = middle_pos_list + 1
        elif shifting_pos_type == 4:
            # rt - 1
            logger.info('deal shifting_pos, -1 ')
            middle_pos_list = middle_pos_list - 1
        middle_pos_list[np.where(middle_pos_list < 0)] = 0
        max_val = len_rt_list - 1
        middle_pos_list[np.where(middle_pos_list > max_val)] = max_val

    expand_range = n_cycles // 2
    start_pos_list = middle_pos_list - expand_range
    if n_cycles % 2 == 0:
        end_pos_list = middle_pos_list + expand_range
    else:
        end_pos_list = middle_pos_list + expand_range + 1

    result_pp = np.zeros((len(start_pos_list), n_cycles), dtype=int)

    # start_pos < 0
    less_zero_indices = np.where(start_pos_list < 0)
    #
    result_pp[less_zero_indices] = [i for i in range(n_cycles)]

    # end_pos > len(rt_list)
    more_end_indices = np.where(end_pos_list > len_rt_list)
    result_pp[more_end_indices] = [i for i in range(len_rt_list - n_cycles, len_rt_list)]

    # else
    other_indices = list(set(np.arange(len(middle_pos_list))) - set(less_zero_indices[0]) - set(more_end_indices[0]))
    for index in range(n_cycles):
        result_pp[other_indices, index] = start_pos_list[other_indices] + index
    return result_pp


def find_rt_pos_by_middle_pos_list(middle_pos_list, len_rt_list, n_cycles):
    expand_range = n_cycles // 2
    start_pos_list = middle_pos_list - expand_range
    if n_cycles % 2 == 0:
        end_pos_list = middle_pos_list + expand_range
    else:
        end_pos_list = middle_pos_list + expand_range + 1

    result_pp = np.zeros((len(start_pos_list), n_cycles), dtype=int)

    # start_pos < 0
    less_zero_indices = np.where(start_pos_list < 0)
    #
    result_pp[less_zero_indices] = [i for i in range(n_cycles)]

    # end_pos > len(rt_list)
    more_end_indices = np.where(end_pos_list > len_rt_list)
    result_pp[more_end_indices] = [i for i in range(len_rt_list - n_cycles, len_rt_list)]

    # else
    other_indices = list(set(np.arange(len(middle_pos_list))) - set(less_zero_indices[0]) - set(more_end_indices[0]))
    for index in range(n_cycles):
        result_pp[other_indices, index] = start_pos_list[other_indices] + index
    return result_pp
