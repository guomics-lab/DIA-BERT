'''

'''

import torch


def get_none_zero_more_indices(frag_rt_matrix, ms2_frag_info, peak_rt_more_zero_limit=3, repeat_num=5):
    #
    frag_rt_matrix_shape = frag_rt_matrix.shape
    frag_rt_matrix = frag_rt_matrix.reshape(frag_rt_matrix_shape[0], repeat_num, frag_rt_matrix_shape[1] // repeat_num,
                                            frag_rt_matrix_shape[2])
    #
    frag_rt_matrix = frag_rt_matrix.transpose(1, 2)
    frag_rt_matrix = frag_rt_matrix.sum(dim=2)
    #

    frag_rt_matrix = frag_rt_matrix[:, 0:20, :]
    frag_rt_matrix[ms2_frag_info[:, :, 2] != 2, :] = 0
    non_zero_count_matrix = (frag_rt_matrix > 1e-6).sum(dim=1)
    #
    indices = (non_zero_count_matrix > peak_rt_more_zero_limit).nonzero()
    return indices, non_zero_count_matrix


def get_none_zero_more_indices_v2(frag_rt_matrix, ms2_frag_info, repeat_num=5):
    #
    frag_rt_matrix_shape = frag_rt_matrix.shape
    frag_rt_matrix = frag_rt_matrix.reshape(frag_rt_matrix_shape[0], repeat_num, frag_rt_matrix_shape[1] // repeat_num,
                                            frag_rt_matrix_shape[2])
    #
    frag_rt_matrix = frag_rt_matrix.transpose(1, 2)
    frag_rt_matrix = frag_rt_matrix.sum(dim=2)

    frag_rt_matrix = frag_rt_matrix[:, 0:20, :]
    frag_rt_matrix[ms2_frag_info[:, :, 2] != 2, :] = 0
    non_zero_count_matrix = (frag_rt_matrix > 1e-6).sum(dim=1)
    return non_zero_count_matrix


def get_none_zero_more_indices_v3(frag_rt_matrix, ms2_frag_info, peak_rt_more_zero_limit=3, repeat_num=5,
                                  open_smooth=False):
    #
    frag_rt_matrix_shape = frag_rt_matrix.shape
    frag_rt_matrix = frag_rt_matrix.reshape(frag_rt_matrix_shape[0], repeat_num, frag_rt_matrix_shape[1] // repeat_num,
                                            frag_rt_matrix_shape[2])
    #
    frag_rt_matrix = frag_rt_matrix.transpose(1, 2)
    frag_rt_matrix = frag_rt_matrix.sum(dim=2)

    frag_rt_matrix = frag_rt_matrix[:, 0:20, :]
    frag_rt_matrix[ms2_frag_info[:, :, 2] != 2, :] = 0

    if open_smooth:
        #
        smooth_result = torch.zeros_like(frag_rt_matrix)
        #
        smooth_view = smooth_result[:, :, 1:-1]
        #
        smooth_mask = torch.logical_or(frag_rt_matrix[:, :, :-2] != 0, frag_rt_matrix[:, :, 2:] != 0)
        #
        smooth_view[smooth_mask] = 1
        #
        smooth_result[:, :, 0][frag_rt_matrix[:, :, 1] > 1e-6] = 1
        #
        smooth_result[:, :, -1][frag_rt_matrix[:, :, -2] > 1e-6] = 1
        #
        smooth_result[frag_rt_matrix != 0] = 1
        non_zero_count_matrix = (smooth_result > 1e-6).sum(dim=1)
    else:
        non_zero_count_matrix = (frag_rt_matrix > 1e-6).sum(dim=1)

    return non_zero_count_matrix


def get_none_zero_more_indices_v4(frag_rt_matrix, ms2_frag_info, each_ms2_mz_tol, peak_rt_more_zero_limit=3,
                                  repeat_num=5):
    #
    frag_rt_matrix_shape = frag_rt_matrix.shape
    frag_rt_matrix = frag_rt_matrix.reshape(frag_rt_matrix_shape[0], repeat_num, frag_rt_matrix_shape[1] // repeat_num,
                                            frag_rt_matrix_shape[2])
    #
    frag_rt_matrix = frag_rt_matrix.transpose(1, 2)
    frag_rt_matrix = frag_rt_matrix.sum(dim=2)

    frag_rt_matrix = frag_rt_matrix[:, 0:20, :]
    frag_rt_matrix[ms2_frag_info[:, :, 2] != 2, :] = 0

    each_ms2_mz_tol = each_ms2_mz_tol[:, 0:20]
    each_ms2_mz_tol = each_ms2_mz_tol.unsqueeze(2)
    avg_frag_rt_matrix = frag_rt_matrix / each_ms2_mz_tol
    avg_frag_rt_matrix = torch.nan_to_num(avg_frag_rt_matrix)
    #
    sum_frag_rt_matrix = avg_frag_rt_matrix.sum(dim=1)
    return sum_frag_rt_matrix
