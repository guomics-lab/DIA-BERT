import time
from queue import Queue

import numpy as np
import torch

from src.common import lib_tensor_handler
from src.common import model_handler
from src.common import rt_utils_v5
from src.identify_v8.score_predict_thread import ScorePrecursorInfoNew
from src.identify_v8.score_predict_thread import ScorePredictThread
from src.utils import frag_rt_matrix_check_utils
from src.utils import win_id_utils


class BaseIdentifyProcess():

    def __init__(self, input_param, rawdata_prefix, lib_cols, lib_data, ms1, ms2, win_range, rt_model_params, min_rt,
                 max_rt,
                 mzml_rt, mzml_instrument, lib_max_intensity, process_gpu_device_num, base_each_num, logger):
        self.input_param = input_param
        self.rawdata_prefix = rawdata_prefix
        self.lib_cols = lib_cols
        self.lib_data = lib_data

        self.ms1 = ms1
        self.ms2 = ms2
        self.win_range = win_range
        self.rt_model_params = rt_model_params
        self.min_rt = min_rt
        self.max_rt = max_rt

        self.mzml_rt = mzml_rt
        self.mzml_instrument = mzml_instrument

        self.lib_max_intensity = lib_max_intensity

        self.process_gpu_device_num = process_gpu_device_num
        self.base_each_num = base_each_num
        self.logger = logger

    def deal_process(self, precursor_id_list_arr):
        logger = self.logger
        try:
            logger.info('Process identify, mzml: {}, gpu: {}, precursor batch num: {}'.format(self.rawdata_prefix,
                                                                                              self.process_gpu_device_num,
                                                                                              len(precursor_id_list_arr)))
            wait_deal_score_queue = Queue(maxsize=2, )
            score_device = torch.device('cuda:' + str(self.process_gpu_device_num))
            diart_model = model_handler.load_model(self.input_param.xrm_model_file, score_device)
            diart_model = diart_model.to(score_device)
            sc_deal_thread = ScorePredictThread('score deal', diart_model, score_device,
                                                self.input_param.out_path, self.input_param.n_cycles,
                                                self.input_param.model_cycles,
                                                self.input_param.frag_repeat_num,
                                                self.input_param.step_size,
                                                self.lib_max_intensity, wait_deal_score_queue, self.logger,
                                                self.input_param.ext_frag_quant_fragment_num,
                                                self.input_param.ext_frag_quant_zero_type,
                                                self.input_param.ext_quant_data_open_smooth)
            #
            # bar = tqdm(total=len(precursor_id_list_arr))
            for each_num, precursor_id_list in enumerate(precursor_id_list_arr):
                logger.info(
                    'Each process deal, mzml is: {}, precursor process: {}/{}'.format(
                        self.rawdata_prefix, each_num, len(precursor_id_list_arr)))
                each_lib_data = self.lib_data[self.lib_data[self.lib_cols['PRECURSOR_ID_COL']].isin(precursor_id_list)]
                self.deal_batch(each_lib_data, self.ms1, self.ms2, self.win_range, self.mzml_rt, self.mzml_instrument,
                                self.rt_model_params, self.min_rt, self.max_rt, self.base_each_num + each_num,
                                sc_deal_thread,
                                wait_deal_score_queue)
                # bar.update(1)
        except Exception:
            logger.exception('identify process exception')

    def deal_batch(self, each_lib_data, ms1, ms2, win_range, mzml_rt, mzml_instrument,
                   rt_model_params, min_rt, max_rt, each_num, score_deal_thread, wait_deal_score_queue):
        logger = self.logger

        ms_rt_list = ms1.rt_list
        mz_max = self.input_param.mz_max
        tt1 = time.time()
        frag_repeat_num = self.input_param.frag_repeat_num
        device = self.input_param.device

        precursors_list, ms1_data_list, ms2_data_list, precursor_info_list = lib_tensor_handler.build_lib_matrix(
            each_lib_data,
            self.lib_cols,
            None,
            None,
            self.input_param.iso_range,
            self.input_param.mz_max,
            self.input_param.max_fragment,
            None)
        ttt1 = time.time()
        ms1_data_tensor, ms2_data_tensor = lib_tensor_handler.build_precursors_matrix_step1(ms1_data_list,
                                                                                            ms2_data_list,
                                                                                            self.input_param.device)
        ttt2 = time.time()
        logger.debug('[TIME COUNT]: build_precursors_matrix_step1: {}'.format(abs(ttt2 - ttt1)))

        ms2_data_tensor = lib_tensor_handler.build_precursors_matrix_step2(ms2_data_tensor)
        ttt3 = time.time()
        logger.debug('[TIME COUNT]: build_precursors_matrix_step2: {}'.format(abs(ttt2 - ttt3)))

        ms1_data_tensor, ms2_data_tensor, ms1_extract_tensor, ms2_extract_tensor = lib_tensor_handler.build_precursors_matrix_step3(
            ms1_data_tensor, ms2_data_tensor, frag_repeat_num,
            device=device)

        ttt4 = time.time()
        logger.debug('[TIME COUNT]: build_precursors_matrix_step3: {}'.format(abs(ttt4 - ttt3)))

        calc_win_t1 = time.time()
        pmt_win_id_list_org = lib_tensor_handler.calc_win_id(ms2_data_tensor, win_range)
        calc_win_t2 = time.time()
        logger.debug('[TIME COUNT]: calc win time: {}'.format(calc_win_t2 - calc_win_t1))

        sp_win_t1 = time.time()
        win_id_pos_arr_list = win_id_utils.split_win_id_list(pmt_win_id_list_org.tolist())
        sp_win_t2 = time.time()
        logger.debug('[TIME COUNT]: split win time: {}'.format(sp_win_t2 - sp_win_t1))
        ttt2 = time.time()
        logger.debug('step1 time: {}'.format(ttt2 - ttt1))

        for pos_index, w_p_arr in enumerate(win_id_pos_arr_list):
            build_m_t11 = time.time()
            #
            pmt_win_id_list = pmt_win_id_list_org[w_p_arr[0]: w_p_arr[1]]
            each_precursors_list = precursors_list[w_p_arr[0]: w_p_arr[1]]
            each_precursor_info_list = precursor_info_list[w_p_arr[0]: w_p_arr[1]]

            #
            ms1_moz_rt_matrix, ms2_moz_rt_matrix, ms1_frag_moz_matrix_coo_matrix, ms2_frag_moz_matrix_coo_matrix = \
                lib_tensor_handler.build_ms_rt_moz_matrix(ms1_extract_tensor[w_p_arr[0]: w_p_arr[1]],
                                                          ms2_extract_tensor[w_p_arr[0]: w_p_arr[1]], pmt_win_id_list,
                                                          mz_max, ms1,
                                                          ms2, device)
            build_m_t2 = time.time()
            logger.debug('[TIME COUNT]: build matrix time: {}'.format(build_m_t2 - build_m_t11))

            peak_t1 = time.time()
            rt_list_len = len(ms_rt_list)
            rt_pos_arr = np.tile(np.arange(rt_list_len), (len(each_precursors_list), 1))
            #
            rt_pos_list = rt_pos_arr.tolist()
            ms1_precursors_frag_rt_matrix, ms1_frag_rt_matrix_result_matmul = self.peak_one(
                ms1_frag_moz_matrix_coo_matrix,
                ms1_moz_rt_matrix, rt_list_len,
                rt_pos_list,
                len(each_precursors_list),
                self.input_param.device)
            ms2_precursors_frag_rt_matrix, ms2_frag_rt_matrix_result_matmul = self.peak_one(
                ms2_frag_moz_matrix_coo_matrix,
                ms2_moz_rt_matrix, rt_list_len,
                rt_pos_list,
                len(each_precursors_list),
                self.input_param.device)

            frag_info = self.build_frag_info(ms1_data_tensor, ms2_data_tensor, w_p_arr, frag_repeat_num,
                                             self.input_param.device)

            ms2_frag_info = frag_info[:, 6:26, :].cpu().numpy()
            peak_t2 = time.time()
            logger.debug('[TIME COUNT]: peak time: {}'.format(peak_t2 - peak_t1))
            #
            non_zero_count_matrix = frag_rt_matrix_check_utils.get_none_zero_more_indices_v3(
                ms2_precursors_frag_rt_matrix, ms2_frag_info, open_smooth=self.input_param.open_smooth)
            indices_t2 = time.time()
            logger.debug('[TIME COUNT]: indices time: {}'.format(indices_t2 - peak_t2))

            precursor_info_np_org = np.array(each_precursor_info_list)
            precursors_list_length = len(each_precursors_list)

            #
            fitting_rt_data_list = rt_utils_v5.get_rt_limit(precursor_info_np_org[:, 3], rt_model_params, min_rt,
                                                            max_rt)

            #

            max_rt = max(ms_rt_list)
            new_ms_rt_list = [-1 * max_rt for _ in range(self.input_param.n_cycles)]
            new_ms_rt_list.extend(ms_rt_list)
            new_ms_rt_list.extend([3 * max_rt for _ in range(self.input_param.n_cycles)])

            #
            n_cycle_rt_pos_arr = rt_utils_v5.find_rt_pos(fitting_rt_data_list, new_ms_rt_list,
                                                         self.input_param.n_cycles)

            n_cycle_rt_pos_arr = n_cycle_rt_pos_arr - self.input_param.n_cycles
            #
            n_cycle_rt_pos_arr[n_cycle_rt_pos_arr < 0] = -1
            n_cycle_rt_pos_arr[n_cycle_rt_pos_arr > len(ms_rt_list) - 1] = -1

            assay_rt_kept, score_precursor_index_np, more4_rt_list_pos = self.get_real_rt_post(
                non_zero_count_matrix, precursors_list_length,
                n_cycle_rt_pos_arr, rt_list_len, ms_rt_list, self.input_param.score_scan_peak_type)

            if len(score_precursor_index_np) == 0:
                continue

            #
            score_fitting_rt_data_list = fitting_rt_data_list[score_precursor_index_np]
            delta_rt_kept = score_fitting_rt_data_list.reshape(-1, 1) - assay_rt_kept.reshape(-1, 1)

            rt_time = time.time()
            logger.debug('[TIME COUNT]: get rt time: {}'.format(rt_time - indices_t2))

            model_cycles = self.input_param.model_cycles
            top_n_rt_real_pos_np = np.array(more4_rt_list_pos.flatten().tolist())
            model_cycles_rt_pos_arr_np = rt_utils_v5.find_rt_pos_by_middle_pos_list(top_n_rt_real_pos_np, rt_list_len,
                                                                                    model_cycles)
            logger.debug('model_cycles_rt_pos_arr_np: {}'.format(model_cycles_rt_pos_arr_np.shape))

            adjust_time = time.time()
            logger.debug('[TIME COUNT]: adjust_time: {}'.format(adjust_time - rt_time))

            '''
            '''
            each_parse_frag_rt_matrix_num = self.input_param.each_parse_frag_rt_matrix_num

            ms1_precursors_frag_rt_matrix = self.parse_frag_rt_matrix_v4(
                ms1_precursors_frag_rt_matrix
                , model_cycles_rt_pos_arr_np, score_precursor_index_np, each_parse_frag_rt_matrix_num)
            if ms1_precursors_frag_rt_matrix is None:
                continue
            ms2_precursors_frag_rt_matrix = self.parse_frag_rt_matrix_v4(
                ms2_precursors_frag_rt_matrix
                , model_cycles_rt_pos_arr_np, score_precursor_index_np, each_parse_frag_rt_matrix_num)
            if ms2_precursors_frag_rt_matrix is None:
                continue

            parse_ms_t3 = time.time()
            logger.debug('[TIME COUNT]: parse_ms_time: {}'.format(parse_ms_t3 - adjust_time))

            score_frag_info = frag_info[score_precursor_index_np]

            precursor_info_choose = precursor_info_np_org[:, 0: 5][score_precursor_index_np]

            score_precursor_feat = np.column_stack([precursor_info_choose, assay_rt_kept, delta_rt_kept])
            cat_t2 = time.time()
            logger.debug('[TIME COUNT]: cat time: {}'.format(cat_t2 - parse_ms_t3))


            score_precursor_info = ScorePrecursorInfoNew(self.rawdata_prefix, each_num, pos_index,
                                                         each_precursors_list,
                                                         score_precursor_feat, ms1_precursors_frag_rt_matrix,
                                                         ms2_precursors_frag_rt_matrix, score_frag_info, mzml_rt,
                                                         mzml_instrument, score_precursor_index_np.tolist())
            st1 = time.time()
            score_deal_thread.case_v2(score_precursor_info, True)

            st2 = time.time()
            logger.debug('[TIME COUNT]: [time peak group] only score time: {}'.format(st2 - st1))
            tt2 = time.time()
            logger.debug('[TIME COUNT]: [time peak group] each peak group: {}'.format(tt2 - tt1))

    def peak_one(self, ms_frag_moz_matrix_coo_matrix, ms_moz_rt_matrix, rt_list_len, ms_rt_pos_list,
                 each_precursors_list_length,
                 device):
        ms_frag_rt_matrix_result_matmul = torch.matmul(ms_frag_moz_matrix_coo_matrix.to(device),
                                                       ms_moz_rt_matrix.to(device))
        ad_t1 = time.time()
        ms_frag_rt_matrix_result = lib_tensor_handler.adjust_diagonal_matrix(ms_frag_rt_matrix_result_matmul,
                                                                             rt_list_len)

        ms_frag_rt_matrix_result = lib_tensor_handler.peak2(ms_frag_rt_matrix_result, ms_rt_pos_list,
                                                            each_precursors_list_length, device)
        ad_t2 = time.time()
        self.logger.debug('[TIME COUNT]: ad peak2 time: {}'.format(ad_t2 - ad_t1))
        return ms_frag_rt_matrix_result, ms_frag_rt_matrix_result_matmul

    def build_frag_info(self, ms1_data_tensor, ms2_data_tensor, w_p_arr, frag_repeat_num, device):
        # #
        ttt1 = time.time()
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
        ttt2 = time.time()
        self.logger.debug('[TIME COUNT]: build_frag_info: {}'.format(ttt2 - ttt1))
        return frag_info

    def get_real_rt_post(self, non_zero_count_matrix, precursors_list_length, n_cycle_rt_pos_arr, rt_list_len,
                         ms_rt_list, score_scan_peak_type=0):
        rt_list_rows = np.arange(precursors_list_length)[:, np.newaxis]
        #

        choose_n_cycle_non_zero_count_matrix = non_zero_count_matrix[rt_list_rows, n_cycle_rt_pos_arr]
        choose_n_cycle_non_zero_count_matrix = choose_n_cycle_non_zero_count_matrix.cpu().numpy()

        peak_rt_matrix = np.zeros_like(choose_n_cycle_non_zero_count_matrix, dtype=int)
        if score_scan_peak_type == 1:
            #
            peak_rt_matrix[:, 1::2] = 1
        elif score_scan_peak_type == 2:
            #
            peak_rt_matrix[:, ::2] = 1
        else:
            peak_rt_matrix[:, :] = 1

        more4_pos_index = np.where(
            (choose_n_cycle_non_zero_count_matrix >= 3) & (n_cycle_rt_pos_arr != -1) & (peak_rt_matrix == 1))

        score_precursor_index_np = more4_pos_index[0]

        more4_rt_list_pos = n_cycle_rt_pos_arr[more4_pos_index]

        more8_index = more4_rt_list_pos > 8
        score_precursor_index_np = score_precursor_index_np[more8_index]
        more4_rt_list_pos = more4_rt_list_pos[more8_index]

        less8_index = more4_rt_list_pos < rt_list_len - 8
        score_precursor_index_np = score_precursor_index_np[less8_index]

        more4_rt_list_pos = more4_rt_list_pos[less8_index]
        real_rt_val = np.array(ms_rt_list)[more4_rt_list_pos]
        return real_rt_val, score_precursor_index_np, more4_rt_list_pos

    def parse_frag_rt_matrix_v4(self, ms_matrix, model_cycles_rt_pos_arr_np, score_precursor_index_np, each_choose_num):

        all_num = len(score_precursor_index_np)
        choose_matrix = [ms_matrix[score_precursor_index_np[start_pos:start_pos + each_choose_num][:, None], :,
                         model_cycles_rt_pos_arr_np[start_pos:start_pos + each_choose_num]] for start_pos in
                         range(0, all_num, each_choose_num)]
        choose_matrix = torch.concat(choose_matrix, dim=0)
        choose_matrix = choose_matrix.transpose(1, 2)
        return choose_matrix
