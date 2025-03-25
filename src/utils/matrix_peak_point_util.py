#

import torch


'''

'''
def get_peak_point(precursor_rt_matrix, win_size=7, step=1):
    #
    precursor_rt_matrix = precursor_rt_matrix.to(torch.float32)
    unfold_result = precursor_rt_matrix.unfold(1, win_size, step)
    #
    if win_size == 7:
        base_data = [1, 2, 3, 4, 3, 2, 1]
        base_mean = 16
    elif win_size == 5:
        base_data = [1, 2, 3, 2, 1]
        base_mean = 9
    elif win_size == 3:
        base_data = [1, 2, 1]
        base_mean = 4

    unfold_result = unfold_result * torch.tensor(base_data, device='cuda')
    unfold_mean = unfold_result.sum(dim=2)

    unfold_mean = unfold_mean / base_mean

    #
    unfold_mean = torch.hstack(
        [torch.zeros(size=(unfold_mean.shape[0], win_size // 2), device=precursor_rt_matrix.device), unfold_mean,
         torch.zeros(size=(unfold_mean.shape[0], win_size // 2), device=precursor_rt_matrix.device)])
    matrix1 = unfold_mean[:, :-1]
    zero_colum_matrix = torch.zeros(size=(unfold_mean.shape[0], 1), device=unfold_mean.device)
    #
    matrix1 = torch.hstack([zero_colum_matrix, matrix1])
    matrix2 = unfold_mean[:, 1:]
    zero_colum_matrix = torch.zeros(size=(unfold_mean.shape[0], 1), device=unfold_mean.device)
    #
    matrix2 = torch.hstack([matrix2, zero_colum_matrix])
    # less
    less_matrix = torch.zeros_like(unfold_mean)
    more_matrix = torch.zeros_like(unfold_mean)

    #
    more_matrix[unfold_mean >= matrix1] = 1
    less_matrix[unfold_mean >= matrix2] = 1
    more_matrix[:, 0] = 0
    less_matrix[:, -1] = 0
    peak_energy_pos_matrix = more_matrix * less_matrix
    return unfold_mean, peak_energy_pos_matrix


#
def deal_non_zero_count_matrix(choose_non_zero_count_matrix, win_size=7, step=1):
    choose_non_zero_count_matrix = choose_non_zero_count_matrix.to(torch.float32)
    unfold_result = choose_non_zero_count_matrix.unfold(1, win_size, step)
    #
    if win_size == 7:
        base_data = [1, 2, 3, 4, 3, 2, 1]
        base_mean = 16
    elif win_size == 5:
        base_data = [1, 2, 3, 2, 1]
        base_mean = 9
    elif win_size == 3:
        base_data = [1, 2, 1]
        base_mean = 4
    else:
        return choose_non_zero_count_matrix
    unfold_result = unfold_result * torch.tensor(base_data, device='cuda')
    unfold_mean = unfold_result.sum(dim=2)
    unfold_mean = unfold_mean / base_mean
    unfold_mean = torch.hstack(
        [torch.zeros(size=(unfold_mean.shape[0], win_size // 2), device=choose_non_zero_count_matrix.device), unfold_mean,
         torch.zeros(size=(unfold_mean.shape[0], win_size // 2), device=choose_non_zero_count_matrix.device)])

    return unfold_mean

