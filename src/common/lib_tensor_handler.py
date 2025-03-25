import time

import numpy as np
import torch
from scipy.sparse import coo_matrix

from src.common import drdia_utils
from src.common_logger import logger

qt3_length = 6

type_column_qt3 = 5
type_column_lib = 0
type_column_light = -1
type_column_iso = 1

frag_type_qt3 = 1
frag_type_lib = 2
frag_type_light = 3
frag_type_iso = 4

frag_type_num = 3

frag_type_dict = {'qt3': 1, 'lib': 2, 'light': 3, 'iso': 4}

group_data_col = ['ProductMz', 'PrecursorCharge', 'FragmentCharge', 'LibraryIntensity', 'FragmentType', 'PrecursorMz',
                  'Tr_recalibrated', 'PeptideSequence', 'decoy', 'transition_group_id']

'''


'''


def build_lib_matrix(lib_data, lib_cols, run_env, diann_raw_rt_map, iso_range,
                     mz_max, max_fragment, thread_num=None):
    logger.info('start build lib matrix')
    times = time.time()
    logger.info('start calc tear library')
    precursor_indice, chunk_indice = drdia_utils.tear_library(lib_data, lib_cols, 1)
    logger.info('end calc tear library, time: {}, chunk_indice len: {}'.format(time.time() - times, len(chunk_indice)))

    precursors_list = []
    ms1_data_list = []
    ms2_data_list = []
    prec_data_list = []

    for i, chunk_index in enumerate(chunk_indice):
        precursor_index = [precursor_indice[idx] for idx in chunk_index]
        each_process_result = precursors_data_group_thread(lib_cols, lib_data, diann_raw_rt_map, precursor_index,
                                                           iso_range, mz_max, max_fragment, chunk_index, None)
        precursors_list.extend(each_process_result[0])
        ms1_data_list.extend(each_process_result[1])
        ms2_data_list.extend(each_process_result[2])
        prec_data_list.extend(each_process_result[3])
    t4 = time.time()
    logger.debug('build lib matrix time: {}'.format(t4 - times))
    return precursors_list, ms1_data_list, ms2_data_list, prec_data_list


def precursors_data_group_thread(lib_cols, library, diann_raw_rt_map, precursor_index_arr, iso_range, mz_max,
                                 max_fragment, chunk_index, process_result_arr=None):
    t1 = time.time()
    first_index_list = [idx[0] for idx in precursor_index_arr]
    precursors_list = library.iloc[first_index_list, :][[lib_cols['PRECURSOR_ID_COL'], 'decoy']].values.tolist()
    t2 = time.time()

    all_index_list = []
    for idx in precursor_index_arr:
        all_index_list.extend(idx)
    group_data_col_values = library.iloc[all_index_list, :][group_data_col].values

    ms_moz_list = [
        format_ms_data(group_data_col_values[idx], iso_range, mz_max, max_fragment, diann_raw_rt_map)
        for idx in
        precursor_index_arr]
    t3 = time.time()

    ms1_data_list = np.array([d[0] for d in ms_moz_list])
    ms2_data_list = np.array([d[1] for d in ms_moz_list])
    precursor_info_list = np.array([d[2] for d in ms_moz_list])

    if process_result_arr is not None:
        process_result_arr.append((precursors_list, ms1_data_list, ms2_data_list, precursor_info_list))
    t4 = time.time()
    return precursors_list, ms1_data_list, ms2_data_list, precursor_info_list


'''
分隔frag，超过最大值截取，小于暂时不处理
'''


def intercept_frags_sort(frag_list, length):
    frag_list.sort(reverse=True)
    if len(frag_list) > length:
        frag_list = frag_list[0: length]
    # if len(frag_list) < length:
    #     return frag_list + [0 for _ in range(length - len(frag_list))]
    return frag_list


'''
build precious frag moz matrix

'''


def build_ms1_data(frag_list, iso_range, mz_max):
    eg_frag = frag_list[0]
    charge = eg_frag[1]
    precursor_mz = eg_frag[5]
    iso_shift_max = int(min(iso_range, (mz_max - precursor_mz) * charge)) + 1
    qt3_frags = [precursor_mz + iso_shift / charge for iso_shift in range(iso_shift_max)]
    qt3_frags = intercept_frags_sort(qt3_frags, qt3_length)
    qt3_data = [
        [qt3_frag, eg_frag[1], eg_frag[2], eg_frag[3], 3, eg_frag[5], type_column_qt3, 0, frag_type_qt3] for
        qt3_frag in qt3_frags]
    if len(qt3_data) < qt3_length:
        qt3_data.extend([[0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(qt3_length - len(qt3_data))])
    return np.array(qt3_data)


'''
'''


def build_ms2_data(frag_list, max_fragment_num):
    frag_count = max_fragment_num * frag_type_num
    frag_num = frag_list.shape[0]
    frag_list = np.vstack([frag_list] * frag_type_num)
    win_id_column = np.array([0] * frag_num * frag_type_num)
    type_column = np.array([0] * frag_num * frag_type_num)
    type_column[frag_num: frag_num * (frag_type_num - 1)] = -1
    type_column[(frag_type_num - 1) * frag_num: frag_type_num * frag_num] = 1

    frag_type_column = np.array([0] * frag_num * frag_type_num)
    frag_type_column[:frag_num] = 2
    frag_type_column[frag_num: frag_num * (frag_type_num - 1)] = 3
    frag_type_column[(frag_type_num - 1) * frag_num: frag_type_num * frag_num] = 4

    frag_list = np.hstack(
        (frag_list, type_column[:, np.newaxis], win_id_column[:, np.newaxis], frag_type_column[:, np.newaxis]))
    if len(frag_list) >= frag_count:
        return frag_list[:frag_count]
    frag_moz = np.zeros((frag_count, frag_list.shape[1]))
    frag_moz[:len(frag_list)] = frag_list
    return frag_moz


'''

sequence_length, precursor_mz, charge, precursor_irt, nr_peaks, assay_rt_kept, delta_rt_kept
'''


def build_precursor_info(frag_list, diann_raw_rt_map):
    precursor_data = frag_list[0]
    if diann_raw_rt_map is not None:
        return [len(precursor_data[7]), precursor_data[5], precursor_data[1], precursor_data[6], len(frag_list),
                diann_raw_rt_map[precursor_data[9]]]
    else:
        return [len(precursor_data[7]), precursor_data[5], precursor_data[1], precursor_data[6], len(frag_list),
                0]


'''
'''


def format_ms_data(frag_list, iso_range, mz_max, max_fragment, diann_raw_rt_map):
    qt3_moz = build_ms1_data(frag_list, iso_range, mz_max)
    frag_moz = build_ms2_data(frag_list[:, 0:6], max_fragment)
    ms1_moz = np.copy(qt3_moz)
    ms1_moz[:, 8] = 5
    frag_moz = np.concatenate([frag_moz, ms1_moz], axis=0)
    precursor_info = build_precursor_info(frag_list, diann_raw_rt_map)
    return qt3_moz, frag_moz, precursor_info


'''

'''


def build_precursors_matrix_step1(ms1_data_list, ms2_data_list, device='cpu'):
    times = time.time()
    ms1_data_tensor = torch.tensor(ms1_data_list, dtype=torch.float32, device=device)
    ms2_data_tensor = torch.tensor(ms2_data_list, dtype=torch.float32, device=device)
    timee = time.time()
    logger.debug('step 1 time: {}'.format(timee - times))
    return ms1_data_tensor, ms2_data_tensor


'''

pmt: precursors_matrix_tensor


'''


def build_precursors_matrix_step2(ms2_data_tensor):
    times = time.time()
    ms2_data_tensor[:, :, 0] = ms2_data_tensor[:, :, 0] + ms2_data_tensor[:, :, 6] / ms2_data_tensor[:, :, 2]
    ms2_data_tensor[torch.isinf(ms2_data_tensor)] = 0
    ms2_data_tensor[torch.isnan(ms2_data_tensor)] = 0
    timee = time.time()
    logger.debug('step 2 time: {}'.format(timee - times))
    return ms2_data_tensor


'''



'''


def build_precursors_matrix_step3(ms1_data_tensor, ms2_data_tensor, frag_repeat_num=5, mz_unit='ppm', mz_tol_ms1=20,
                                  mz_tol_ms2=50, device='cpu'):
    times = time.time()
    ms1_data_tensor = ms1_data_tensor.repeat(1, frag_repeat_num, 1)

    ms2_data_tensor = ms2_data_tensor.repeat(1, frag_repeat_num, 1)
    ms1_extract_tensor, ms1_mz_tol_half_org = extract_width(ms1_data_tensor[:, :, 0], mz_unit, mz_tol_ms1,
                                                            device=device)
    ms2_extract_tensor, ms2_mz_tol_half_org = extract_width(ms2_data_tensor[:, :, 0], mz_unit, mz_tol_ms2,
                                                            device=device)

    timee = time.time()
    logger.debug('step 3 time: {}'.format(timee - times))
    return ms1_data_tensor, ms2_data_tensor, ms1_extract_tensor, ms2_extract_tensor


def build_precursors_matrix_step3_v2(ms1_data_tensor, ms2_data_tensor, frag_repeat_num=5, mz_unit='ppm', mz_tol_ms1=20,
                                     mz_tol_ms2=50, device='cpu'):
    times = time.time()
    ms1_data_tensor = ms1_data_tensor.repeat(1, frag_repeat_num, 1)

    ms2_data_tensor = ms2_data_tensor.repeat(1, frag_repeat_num, 1)
    ms1_extract_tensor, ms1_mz_tol_half = extract_width(ms1_data_tensor[:, :, 0], mz_unit, mz_tol_ms1, device=device)
    ms2_extract_tensor, ms2_mz_tol_half = extract_width(ms2_data_tensor[:, :, 0], mz_unit, mz_tol_ms2, device=device)

    timee = time.time()
    logger.debug('step 3 time: {}'.format(timee - times))
    return ms1_data_tensor, ms2_data_tensor, ms1_extract_tensor, ms2_extract_tensor, ms2_mz_tol_half


def extract_width(mz_to_extract, mz_unit, mz_tol, max_extract_len=20, frag_repeat_num=5, max_moz_num=50, device='cpu'):
    if mz_to_extract.eq(0).all():
        return torch.zeros(mz_to_extract.size() + (max_extract_len,))

    if mz_unit == "Da":
        mz_tol_half = (mz_to_extract / mz_to_extract) * mz_tol / 2
    elif mz_unit == "ppm":
        mz_tol_half = mz_to_extract * mz_tol * 0.000001 / 2
    else:
        raise Exception("Invalid mz_unit format: %s. Only Da and ppm are supported." % mz_unit)

    mz_tol_half[torch.isnan(mz_tol_half)] = 0

    mz_tol_half_num = (max_moz_num / 1000) / 2
    condition = mz_tol_half[:, :] > mz_tol_half_num
    mz_tol_half[condition] = mz_tol_half_num

    mz_tol_half = torch.ceil(mz_tol_half * 1000 / frag_repeat_num) * frag_repeat_num

    extract_width_list = torch.stack((mz_to_extract * 1000 - mz_tol_half, mz_to_extract * 1000 + mz_tol_half),
                                     dim=-1).floor()

    t1 = time.time()
    batch_num = int(mz_to_extract.shape[1] / frag_repeat_num)
    cha_tensor = (extract_width_list[:, 0:batch_num, 1] - extract_width_list[:, 0:batch_num, 0]) / frag_repeat_num

    extract_width_list[:, 0:batch_num, 0] = extract_width_list[:, 0:batch_num, 0]
    extract_width_list[:, 0:batch_num, 1] = extract_width_list[:, 0:batch_num, 0] + cha_tensor - 1

    extract_width_list[:, batch_num:batch_num * 2, 0] = extract_width_list[:, 0:batch_num, 0] + cha_tensor
    extract_width_list[:, batch_num:batch_num * 2, 1] = extract_width_list[:, 0:batch_num, 0] + 2 * cha_tensor - 1

    extract_width_list[:, batch_num * 2:batch_num * 3, 0] = extract_width_list[:, 0:batch_num, 0] + 2 * cha_tensor
    extract_width_list[:, batch_num * 2:batch_num * 3, 1] = extract_width_list[:, 0:batch_num, 0] + 3 * cha_tensor - 1

    extract_width_list[:, batch_num * 3:batch_num * 4, 0] = extract_width_list[:, 0:batch_num, 0] + 3 * cha_tensor
    extract_width_list[:, batch_num * 3:batch_num * 4, 1] = extract_width_list[:, 0:batch_num, 0] + 4 * cha_tensor - 1

    extract_width_list[:, batch_num * 4:batch_num * 5, 0] = extract_width_list[:, 0:batch_num, 0] + 4 * cha_tensor
    extract_width_list[:, batch_num * 4:batch_num * 5, 1] = extract_width_list[:, 0:batch_num, 0] + 5 * cha_tensor - 1

    t2 = time.time()
    logger.debug('extract_width step1 time: {}'.format(t2 - t1))

    new_tensor = torch.zeros(mz_to_extract.shape[0], mz_to_extract.shape[1], max_moz_num, dtype=torch.float32,
                             device=device)
    for i in range(new_tensor.shape[2]):
        new_tensor[:, :, i] = extract_width_list[:, :, 0] + i * 1
        condition = new_tensor[:, :, i] > extract_width_list[:, :, 1]
        new_tensor[:, :, i][condition] = 0
    return new_tensor, mz_tol_half


'''


'''


def calc_win_id(pmt, win_range):
    win_id = np.searchsorted(win_range[:, 0], pmt[:, 0, 5].cpu().numpy()) - 1
    win_id = np.maximum(win_id, 0)
    return win_id


'''
'''


def build_ms_rt_moz_matrix(ms1_extract_tensor, ms2_extract_tensor, pmt_win_id_list, mz_max, ms1, ms2, device='cpu'):
    ms1_frag_moz_matrix_coo_matrix, ms2_frag_moz_matrix_coo_matrix = construct_diagonal_matrix_v3(ms1_extract_tensor,
                                                                                                  ms2_extract_tensor,
                                                                                                  pmt_win_id_list,
                                                                                                  mz_max, device)

    mst1 = time.time()
    # 计算rt_moz
    ms1_moz_rt_list = [ms1.moz_rt_matrix for _ in sorted(set(pmt_win_id_list))]
    ms1_moz_rt_win_id_list = [win_id for win_id in sorted(set(pmt_win_id_list))]

    ms2_moz_rt_list = [ms2[win_id].moz_rt_matrix for win_id in sorted(set(pmt_win_id_list))]
    ms2_moz_rt_win_id_list = [win_id for win_id in sorted(set(pmt_win_id_list))]
    mst2 = time.time()
    # logger.info('build ms moz rt list time: {}'.format(mst2 - mst1))
    ms1_frag_moz_matrix_coo_matrix = construct_sparse_tensor(ms1_frag_moz_matrix_coo_matrix, device)
    ms2_frag_moz_matrix_coo_matrix = construct_sparse_tensor(ms2_frag_moz_matrix_coo_matrix, device)
    mst3 = time.time()
    # logger.info('build ms moz rt list time: {}'.format(mst3 - mst2))

    t1 = time.time()
    ms1_moz_rt_matrix = construct_diagonal_matrix(ms1_moz_rt_list, ms1_moz_rt_win_id_list, mz_max)
    t11 = time.time()
    # logger.info('build ms1 moz rt diagonal time: {}'.format(t11 - t1))
    ms1_moz_rt_matrix = construct_sparse_tensor(ms1_moz_rt_matrix, device)
    t12 = time.time()
    # logger.info('build ms1 moz rt time: {}'.format(t12 - t11))

    t2 = time.time()
    ms2_moz_rt_matrix = construct_diagonal_matrix(ms2_moz_rt_list, ms2_moz_rt_win_id_list, mz_max)
    t21 = time.time()
    # logger.info('build ms2 moz rt diagonal time: {}'.format(t21 - t2))
    ms2_moz_rt_matrix = construct_sparse_tensor(ms2_moz_rt_matrix, device)
    t22 = time.time()
    # logger.info('build ms2 moz rt time: {}'.format(t22 - t21))
    return ms1_moz_rt_matrix, ms2_moz_rt_matrix, ms1_frag_moz_matrix_coo_matrix, ms2_frag_moz_matrix_coo_matrix


'''
'''


def construct_diagonal_matrix_v3(ms1_extract_tensor_three, ms2_extract_tensor_three, pmt_win_id_list, mz_max, device):
    t1 = time.time()
    conn_zero_ms1 = (ms1_extract_tensor_three == 0)
    conn_zero_ms2 = (ms2_extract_tensor_three == 0)
    each_moz_max = (mz_max + 100) * 1000
    max_win_id = pmt_win_id_list[-1]
    min_win_id = pmt_win_id_list[0]
    pmt_win_id_t = pmt_win_id_list.reshape(pmt_win_id_list.shape[0], 1, 1)
    ms1_extract_tensor_three = ms1_extract_tensor_three + torch.tensor(pmt_win_id_t, device=device,
                                                                       dtype=torch.float32) * each_moz_max
    ms2_extract_tensor_three = ms2_extract_tensor_three + torch.tensor(pmt_win_id_t, device=device,
                                                                       dtype=torch.float32) * each_moz_max

    ms1_extract_tensor_three[conn_zero_ms1] = 0
    ms2_extract_tensor_three[conn_zero_ms2] = 0

    ms1_coo_matrix = convert_to_coo(ms1_extract_tensor_three, each_moz_max, max_win_id, min_win_id)
    ms2_coo_matrix = convert_to_coo(ms2_extract_tensor_three, each_moz_max, max_win_id, min_win_id)
    t2 = time.time()
    logger.debug('construct_diagonal_matrix time: {}'.format(t2 - t1))
    return ms1_coo_matrix, ms2_coo_matrix


'''
'''


def convert_to_coo(extract_tensor_three, each_moz_max, max_win_id, min_win_id):
    t1 = time.time()
    extract_tensor = extract_tensor_three.reshape(-1, extract_tensor_three.shape[2])
    non_zero_indices = torch.argwhere(extract_tensor != 0)
    non_zero_elements = extract_tensor[non_zero_indices[:, 0], non_zero_indices[:, 1]]
    pmt_data = np.array([1] * len(non_zero_elements))
    pmt_col = non_zero_elements.cpu().numpy()
    pmt_col = pmt_col - min_win_id * each_moz_max

    pmt_row = (non_zero_indices[:, 0]).cpu().numpy()
    pmt_shape = [extract_tensor.shape[0], each_moz_max * (max_win_id - min_win_id + 1)]
    pmt_matrix = coo_matrix((pmt_data, (pmt_row, pmt_col)), shape=pmt_shape)
    t3 = time.time()
    logger.debug('convert_to_coo time: {}'.format(t3 - t1))
    return pmt_matrix


'''
'''


def construct_sparse_tensor(sparse_mx, device):
    indices = torch.vstack((torch.tensor(sparse_mx.row, device=device), torch.tensor(sparse_mx.col, device=device)))
    values = torch.tensor(sparse_mx.data, dtype=torch.float32, device=device)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def adjust_diagonal_matrix(frag_rt_matrix, rt_list_len):
    col = frag_rt_matrix._indices()[1, :] % rt_list_len
    indices = torch.vstack((frag_rt_matrix._indices()[0, :], col))
    values = frag_rt_matrix._values()
    shape = torch.Size((frag_rt_matrix.shape[0], rt_list_len))
    del frag_rt_matrix
    logger.debug('adjust_diagonal_matrix: val count: {}, shape: {}'.format(len(values), shape))
    return torch.sparse_coo_tensor(indices, values, shape).to_dense()


'''

'''


def construct_diagonal_matrix(matrix_list, ms2_moz_rt_win_id_list, mz_max):
    t1 = time.time()
    each_moz_max = (mz_max + 100) * 1000
    max_win_id = ms2_moz_rt_win_id_list[-1]
    min_win_id = ms2_moz_rt_win_id_list[0]
    t2 = time.time()
    matrix_size = [(matrix.shape[0], matrix.shape[1]) for matrix in matrix_list]
    total_rows = each_moz_max * (max_win_id - min_win_id + 1)
    total_cols = sum([size[1] for size in matrix_size])
    t3 = time.time()
    # logger.info('diagonal_matrix step2: {}'.format(t3 - t2))

    matrix_size = [(0, 0)] + [(matrix.shape[0], matrix.shape[1]) for matrix in matrix_list]
    matrix_size.pop()
    t4 = time.time()
    # logger.info('diagonal_matrix step3: {}'.format(t4 - t3))

    matrix_row_offset = np.array([(d - min_win_id) * each_moz_max for d in ms2_moz_rt_win_id_list])
    t5 = time.time()
    # logger.info('diagonal_matrix step4: {}'.format(t5 - t4))
    matrix_col_offset = np.cumsum([size[1] for size in matrix_size])
    t6 = time.time()
    # logger.info('diagonal_matrix step5: {}'.format(t6 - t5))
    # adjust coo
    total_row = np.concatenate(([matrix.row + matrix_row_offset[index] for index, matrix in enumerate(matrix_list)]))
    t7 = time.time()
    # logger.info('diagonal_matrix step6: {}'.format(t7 - t6))
    total_col = np.concatenate(([matrix.col + matrix_col_offset[index] for index, matrix in enumerate(matrix_list)]))
    t8 = time.time()
    # logger.info('diagonal_matrix step7: {}'.format(t8 - t7))
    total_data = np.concatenate(([matrix.data for matrix in matrix_list]))
    t9 = time.time()
    # logger.info('diagonal_matrix step8: {}'.format(t9 - t8))
    logger.debug(
        'diagonal_matrix: total_rows: {}, total_cols: {}, data_len: {}'.format(total_rows, total_cols, len(total_data)))

    new_matrix = coo_matrix((total_data, (total_row, total_col)), shape=(total_rows, total_cols))
    t10 = time.time()
    logger.debug('diagonal_matrix step9: {}'.format(t10 - t9))
    return new_matrix


'''


'''


def peak2(frag_rt_matrix_result, rt_list, precursors_list_len, device=None):
    frag_rt_matrix_result = frag_rt_matrix_result.reshape(precursors_list_len,
                                                          int(frag_rt_matrix_result.shape[0] / precursors_list_len),
                                                          frag_rt_matrix_result.shape[1])
    ms2_rt_tensor = torch.LongTensor(rt_list).to(frag_rt_matrix_result.device)
    ms2_rt_diagonal_indices = torch.arange(ms2_rt_tensor.size(0))
    return frag_rt_matrix_result[ms2_rt_diagonal_indices[:, None], :, ms2_rt_tensor].transpose(1, 2)


'''

'''


def build_ext_ms1_matrix(ms1_data_tensor, device):
    ext_matrix = ms1_data_tensor[:, :, [0, 3, 8, 4]].to(device)
    return ext_matrix


'''
'''


def build_ext_ms2_matrix(ms2_data_tensor, device):
    ext_matrix = ms2_data_tensor[:, :, [0, 3, 8, 4]].to(device)
    return ext_matrix
