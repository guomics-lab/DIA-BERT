import os
import pickle
import numpy as np
import pandas as pd


def read_dump_file_list(pkl_dir):
    precursor_info_dump_file_list = []
    file_list = os.listdir(pkl_dir)
    for file_name in file_list:
        if file_name.endswith('.pkl'):
            file_path = os.path.join(pkl_dir, file_name)
            if os.path.exists(file_path):
                precursor_info_dump_file_list.append(file_path)
    if len(precursor_info_dump_file_list) == 0:
        return []
    return precursor_info_dump_file_list

'''
'''
def get_fdr_precursor(pkl_dir_path, fdr):
    all_p_info = []
    file_list = read_dump_file_list(pkl_dir_path)

    score_np_list = []
    for index, file_path in enumerate(file_list):
        with open(file_path, 'rb') as f:
            precursors_list, precursors_info, rsm, frag_info, score_np = pickle.load(f)
            precursors_list_np = np.array(precursors_list)
            score_np = score_np.reshape(-1, 1)
            p_info = np.column_stack([precursors_list_np, score_np, precursors_info.numpy()[:, 5]])
            all_p_info.append(p_info)

            score_np_list.append(frag_info[:, :, 0])

    all_score_np = np.vstack(score_np_list)
    all_p_info_np = np.vstack(all_p_info)
    df = pd.DataFrame(all_p_info_np, columns=['transition_group_id', 'decoy', 'score', 'assay_rt_kept'])
    fdr_num, filtered_df = get_prophet_precursor(df, fdr)
    precursor_list = filtered_df['transition_group_id'].tolist()
    precursor_df_index = df[df['transition_group_id'].isin(precursor_list)].index
    fdr_precursor_score_np = all_score_np[precursor_df_index]
    return fdr_num, filtered_df, fdr_precursor_score_np


def get_prophet_precursor(df, fdr):
    sort_df = df.sort_values(by='score', ascending=False, ignore_index=True)
    target_num = (sort_df.decoy == 0).cumsum()
    decoy_num = (sort_df.decoy == 1).cumsum()
    target_num[target_num == 0] = 1
    decoy_num[decoy_num == 0] = 1
    sort_df['q_value'] = decoy_num / target_num
    sort_df['q_value'] = sort_df['q_value'][::-1].cummin()

    fdr = round(fdr / 100, 2)
    fdr_num = ((sort_df['q_value'] <= fdr) & (sort_df['decoy'] == 0)).sum()

    #  conservative FDR estimate
    filtered_df = sort_df[(sort_df['q_value'] <= 0.1)][
        ['transition_group_id', 'score', 'decoy', 'assay_rt_kept', 'q_value']]
    return fdr_num, filtered_df


def calc_fdr(pkl_dir_path):
    all_p_info = []
    file_list = read_dump_file_list(pkl_dir_path)
    for index, file_path in enumerate(file_list):
        with open(file_path, 'rb') as f:
            precursors_list, precursors_info, rsm, frag_info, score_np = pickle.load(f)
            precursors_list_np = np.array(precursors_list)
            score_np = score_np.reshape(-1, 1)
            p_info = np.column_stack([precursors_list_np, score_np])
            all_p_info.append(p_info)
    all_p_info_np = np.vstack(all_p_info)
    df = pd.DataFrame(all_p_info_np, columns=['transition_group_id', 'decoy', 'score'])
    return get_prophet_result(df)


def get_prophet_result(df):
    df = df.sort_values(by='score', ascending=False, ignore_index=True)
    # df = df.rename(columns={'precursor_id': 'transition_group_id'})
    target_num = (df.decoy == 0).cumsum()
    decoy_num = (df.decoy == 1).cumsum()

    target_num[target_num == 0] = 1
    decoy_num[decoy_num == 0] = 1
    df['q_value'] = decoy_num / target_num
    df['q_value'] = df['q_value'][::-1].cummin()

    # log
    fdr10_num = ((df['q_value'] <= 0.1) & (df['decoy'] == 0)).sum()
    fdr1_num = ((df['q_value'] <= 0.01) & (df['decoy'] == 0)).sum()

    #  conservative FDR estimate
    # filtered_df = df[(df['q_value'] <= 0.1)][
    #     ['transition_group_id', 'score', 'decoy', 'q_value']]
    print('fdr10: {}, dr1: {}'.format(fdr10_num, fdr1_num))
    return fdr10_num, fdr1_num


