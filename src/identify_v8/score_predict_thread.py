#
import os
import pickle
import threading
import time
from queue import Queue

import pandas as pd
import torch
import torch.nn as nn

from src.common.model.score_model import DIArtModel
from src.common.model.score_model import FeatureEngineer


class ScorePrecursorInfoNew(object):

    def __init__(self, base_raw_name, index_num, pos_index,
                 precursor_list, precursor_info,
                 ms1_precursors_frag_rt_matrix, ms2_precursors_frag_rt_matrix,
                 frag_info, mzml_rt, mzml_instrument, score_precursor_index_list):
        self.base_raw_name = base_raw_name
        self.index_num = index_num
        self.pos_index = pos_index
        # p * 2
        self.precursor_list = precursor_list
        # p * 135 * 7
        self.precursor_info = precursor_info
        # 135 * p * 30 * 16
        self.ms1_precursors_frag_rt_matrix = ms1_precursors_frag_rt_matrix
        # 135 * p * 300 * 16
        self.ms2_precursors_frag_rt_matrix = ms2_precursors_frag_rt_matrix
        # p * 66 * 4
        self.frag_info = frag_info

        self.mzml_rt = mzml_rt
        self.mzml_instrument = mzml_instrument
        self.score_precursor_index_list = score_precursor_index_list


class ScorePredictThread(threading.Thread):

    def __init__(self, thread_name, xrm_model: DIArtModel, device, base_out_file, n_cycles, model_cycles,
                 frag_repeat_num, step_size, lib_max_intensity, wait_deal_queue: Queue, logger,
                 ext_frag_quant_fragment_num=None, ext_frag_quant_zero_type=None, ext_quant_data_open_smooth=None):
        super(ScorePredictThread, self).__init__(name=thread_name)
        self.wait_deal_queue = wait_deal_queue
        self.xrm_model = xrm_model
        self.device = device
        self.n_cycles = n_cycles
        self.model_cycles = model_cycles
        self.frag_repeat_num = frag_repeat_num
        self.step_size = step_size
        self.lib_max_intensity = lib_max_intensity
        self.base_out_file = base_out_file
        self.logger = logger

        self.ext_frag_quant_fragment_num = ext_frag_quant_fragment_num
        self.ext_frag_quant_zero_type = ext_frag_quant_zero_type
        self.ext_quant_data_open_smooth = ext_quant_data_open_smooth

    def run(self):
        #
        while True:
            extract_precursors_info = self.wait_deal_queue.get()
            if type(extract_precursors_info) == str:
                break
            self.logger.info('deal score, base_raw_name: {}, index_num:{}'.format(extract_precursors_info.base_raw_name,
                                                                                  extract_precursors_info.index_num))
            try:
                score_time_s = time.time()
                self.case_v2(extract_precursors_info, dump_disk=True)
                score_time_e = time.time()
                self.logger.debug('[score time]: {}'.format(score_time_e - score_time_s))
            except Exception:
                self.logger.exception('score group exception')

    def case_v2(self, extract_precursors_info, dump_disk=False):
        logger = self.logger
        score_start_time = time.time()
        mzml_rt = extract_precursors_info.mzml_rt
        mzml_instrument = extract_precursors_info.mzml_instrument
        score_precursor_index_list = extract_precursors_info.score_precursor_index_list

        # p * 2
        precursor_list = extract_precursors_info.precursor_list
        # batch
        # p * 7
        precursor_info = torch.tensor(extract_precursors_info.precursor_info, dtype=torch.float32, device=self.device)
        batch_size = len(precursor_info)
        # p * (5 * 6) * 16
        ms1_precursors_frag_rt_matrix = extract_precursors_info.ms1_precursors_frag_rt_matrix
        # p * (5 * 66) * 16
        ms2_precursors_frag_rt_matrix = extract_precursors_info.ms2_precursors_frag_rt_matrix


        # p * 72 * 4
        frag_info = extract_precursors_info.frag_info


        ms1_matrix_shape = ms1_precursors_frag_rt_matrix.shape
        ms2_matrix_shape = ms2_precursors_frag_rt_matrix.shape
        ms1_precursors_frag_rt_matrix = ms1_precursors_frag_rt_matrix.reshape(ms1_matrix_shape[0],
                                                                              self.frag_repeat_num,
                                                                              ms1_matrix_shape[
                                                                                  1] // self.frag_repeat_num,
                                                                              ms1_matrix_shape[2])
        ms2_precursors_frag_rt_matrix = ms2_precursors_frag_rt_matrix.reshape(ms2_matrix_shape[0],
                                                                              self.frag_repeat_num,
                                                                              ms2_matrix_shape[
                                                                                  1] // self.frag_repeat_num,
                                                                              ms2_matrix_shape[2])

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
        #

        mzml_name = extract_precursors_info.base_raw_name

        #
        all_scores_list = self.calc_score_v2(batch_size, precursor_info, rsm, frag_info, mzml_rt, mzml_instrument,
                                             mzml_name)
        scores_list = all_scores_list.cpu().tolist()
        score_index_list = [i for i in range(len(score_precursor_index_list))]
        pd_data = {'p_index': score_precursor_index_list, 'score': scores_list,
                   'score_index': score_index_list}
        df = pd.DataFrame(pd_data)

        group_result = df.groupby('p_index').apply(get_max_score_indices).reset_index()
        group_result.columns = ['p_index', 'score_index_list']
        delta_rt_list = precursor_info[:, -1].tolist()
        group_result['delta_rt_min_score_index'] = group_result['score_index_list'].apply(find_min_delta_rt_index,
                                                                                          args=(delta_rt_list,))
        p_index_list = group_result['p_index'].tolist()
        save_index_list = group_result['delta_rt_min_score_index'].tolist()

        save_precursor_list = [precursor_list[i] for i in p_index_list]
        save_precursors_info = precursor_info[save_index_list].cpu()

        save_rsm = rsm[save_index_list]
        save_frag_info = frag_info[save_index_list].cpu().numpy()

        raw_out_dir11 = os.path.join(self.base_out_file, extract_precursors_info.base_raw_name)
        if not os.path.exists(raw_out_dir11):
            os.makedirs(raw_out_dir11)

        if dump_disk:

            each_rsm = save_rsm.transpose(1, 2).reshape(-1, 16, 72, 5)
            rsm_pad = nn.ZeroPad2d((0, 3, 0, 0))
            save_rsm = rsm_pad(each_rsm).transpose(1, 3)
            save_rsm = save_rsm.cpu().numpy()

            save_score = all_scores_list[save_index_list].cpu()
            precursor_info_dump = (
                save_precursor_list, save_precursors_info, save_rsm, save_frag_info, save_score)
            raw_out_dir = os.path.join(self.base_out_file, extract_precursors_info.base_raw_name, 'identify_data')
            if not os.path.exists(raw_out_dir):
                os.makedirs(raw_out_dir)
            score_info_out_dir = os.path.join(raw_out_dir, 'score')
            if not os.path.exists(score_info_out_dir):
                os.makedirs(score_info_out_dir)

            save_path = os.path.join(raw_out_dir,
                                     'precursor_score_info_dump_{}_{}.pkl'.format(extract_precursors_info.index_num,
                                                                                  extract_precursors_info.pos_index))
            with open(save_path, 'wb') as f:
                pickle.dump(precursor_info_dump, f)

            save_path = os.path.join(score_info_out_dir,
                                     'score_info_{}_{}.pkl'.format(extract_precursors_info.index_num,
                                                                   extract_precursors_info.pos_index))
            with open(save_path, 'wb') as f:
                pickle.dump((save_precursor_list, save_score), f)

    def calc_score_v2(self, batch_size, precursor_info, rsm, frag_info, mzml_rt, mzml_instrument, base_raw_name=None):
        lib_max_intensity = self.lib_max_intensity
        logger = self.logger

        all_scores_list = []
        time_s = time.time()
        each_precursor_info = precursor_info
        each_rsm = rsm
        each_frag_info = frag_info

        logger.debug('each_rsm pre: {}'.format(each_rsm.shape))
        each_rsm = each_rsm.transpose(1, 2).reshape(-1, 16, 72, 5)
        rsm_pad = nn.ZeroPad2d((0, 3, 0, 0))
        each_rsm = rsm_pad(each_rsm).transpose(1, 3)

        t2 = time.time()
        logger.debug('score step1 time: {}'.format(t2 - time_s))

        each_precursor_info = each_precursor_info.to(self.device)
        each_rsm = each_rsm.to(self.device)
        each_frag_info = each_frag_info.to(self.device)
        t21 = time.time()
        logger.debug('score step11 time: {}'.format(t21 - t2))

        each_precursor_info = torch.nan_to_num(each_precursor_info)
        each_rsm = torch.nan_to_num(each_rsm)
        each_frag_info = torch.nan_to_num(each_frag_info)

        t3 = time.time()
        logger.debug('score step2 time: {}'.format(t3 - t2))

        for start_pos in range(0, batch_size, self.step_size):
            thiz_each_rsm = each_rsm[start_pos: start_pos + self.step_size]
            thiz_each_frag_info = each_frag_info[start_pos: start_pos + self.step_size]
            thiz_each_precursor_info = each_precursor_info[start_pos: start_pos + self.step_size]

            thiz_each_rsm, thiz_each_frag_info, thiz_each_precursor_info = FeatureEngineer.feature_engineer(
                thiz_each_rsm, thiz_each_frag_info,
                thiz_each_precursor_info,
                lib_max_intensity, mzml_rt,
                mzml_instrument)

            thiz_each_rsm = torch.nan_to_num(thiz_each_rsm)
            thiz_each_frag_info = torch.nan_to_num(thiz_each_frag_info)
            thiz_each_precursor_info = torch.nan_to_num(thiz_each_precursor_info)

            logger.debug('pred_f16 each_rsm shape: {}, each_frag_info shape: {}, each_precursor_info shape: {}'.format(
                thiz_each_rsm.shape, thiz_each_frag_info.shape, thiz_each_precursor_info.shape))

            each_batch_scores = DIArtModel.pred_f16(self.xrm_model, thiz_each_rsm.to(self.device),
                                                    thiz_each_frag_info.to(self.device),
                                                    thiz_each_precursor_info.to(self.device))

            if not len(each_batch_scores) == len(thiz_each_rsm):
                each_batch_scores = torch.tensor([0 for _ in range(len(thiz_each_rsm))], device=self.device, dtype=thiz_each_rsm.dtype)
                logger.error('******error data, save to: {}'.format(len(each_batch_scores)))

                error_data_out_dir = os.path.join(self.base_out_file, base_raw_name, 'error_data')
                if not os.path.exists(error_data_out_dir):
                    os.makedirs(error_data_out_dir)

                with open(os.path.join(error_data_out_dir, '{}.pkl'.format(time.time())), mode='wb') as f:
                    pickle.dump((thiz_each_rsm, thiz_each_frag_info, thiz_each_precursor_info), f)

            all_scores_list.append(each_batch_scores)

        time_e = time.time()
        logger.debug('[score time] all score time: {}, batch_size: {}'.format(time_e - time_s, batch_size))

        all_scores_list = torch.concatenate(all_scores_list, dim=0)
        return all_scores_list


def get_max_score_indices(group):
    max_score = group['score'].max()
    return group[group['score'] == max_score]['score_index'].tolist()


def find_min_delta_rt_index(score_index_list, delta_rt_list):
    #
    values = [delta_rt_list[i] for i in score_index_list]
    #
    min_value = min(values)
    #
    min_index = values.index(min_value)
    #
    result_index = score_index_list[min_index]
    return result_index


def find_middle_of_longest_subarray(subarray):
    if not subarray:
        return None

    subarray.sort()

    max_length = 0
    current_length = 0
    start_index = 0
    best_start_index = 0

    for i in range(len(subarray)):
        if i == 0 or subarray[i] == subarray[i - 1] + 1:
            current_length += 1
        else:
            if current_length > max_length:
                max_length = current_length
                best_start_index = start_index
            current_length = 1
            start_index = i

    if current_length > max_length:
        max_length = current_length
        best_start_index = start_index

    longest_subarray = subarray[best_start_index:best_start_index + max_length]
    middle_index = (len(longest_subarray) - 1) // 2
    return longest_subarray[middle_index]
