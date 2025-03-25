import copy
import math
import os
import shutil
import time
from queue import Queue

import pandas as pd
import torch
from torch.multiprocessing import Process

from src.common import constant
from src.common_logger import create_new_logger
from src.common import lib_handler
from src.common import model_handler
from src.common import raw_handler
from src.common import runtime_data_info
from src.common import timepoint_handler as timepoint_handler_v1
from src.common import timepoint_handler_v3 as timepoint_handler_v3
from src.common.constant import ProgressStepEnum, ProgressStepStatusEnum
from src.common.model.score_model import FeatureEngineer
from src.common.obj import InputParam
from src.finetune.eval_process import EvalProcess
from src.finetune.finetune_precursor_peak_process import FinetunePrecursorPeakProcess
from src.finetune.finetune_train_process import FinetuneTrainProcess
from src.identify_v8.identify_process import BaseIdentifyProcess
from src.identify_v8.score_predict_thread import ScorePredictThread
from src.lib.lib_process import LibProcess
from src.protein_infer.protein_infer_process import ProteinInferProcess
from src.quant.quant_process import QuantProcess
from src.quantifiction.quantification_process import QuantificationProcess
from src.result_build.result_build_process import ResultBuildProcess
from src.utils import instrument_info_utils
from src.utils import list_utils
from src.utils import msg_send_utils


class IdentifyProcessHandler():
    def __init__(self, input_param: InputParam, logger=None):
        self.input_param = input_param
        self.logger = logger

        self.mzml_files = []
        self.lib_prefix = None
        self.temp_lib_path = None
        self.lib_cols_org = None
        self.lib_data_org = None
        self.deal_num = 0
        self.diart_model = None
        self.lib_max_intensity = None

        self.rawdata_prefix = None
        self.current_mzml_name = None
        self.mzml_rt = None
        self.mzml_instrument = None

    def shell_deal_process(self):
        try:
            torch.multiprocessing.set_start_method('spawn', force=True)
            self.deal_process()
        except Exception:
            self.logger.exception('Process exception')

    def deal_process(self):
        #
        # self.logger.info('Processing param is: {}'.format(self.input_param.__dict__))
        runtime_data_info.runtime_data.mzml_deal_count = 0
        runtime_data_info.runtime_data.start_timestamp = time.time()
        self.get_file_list()

        if self.input_param.gpu_devices == 'auto':
            device_list = []
            for ii in range(torch.cuda.device_count()):
                device_list.append(ii)
            self.input_param.gpu_devices = device_list
        else:
            gpu_device_arr = self.input_param.gpu_devices.split(',')
            device_list = []
            for each_device in gpu_device_arr:
                device_list.append(int(each_device))
            self.input_param.gpu_devices = device_list

        self.input_param.device = 'cuda:' + str(self.input_param.gpu_devices[0])

        self.lib_prefix = os.path.split(self.input_param.lib)[-1].split('.')[0]
        lib_process = LibProcess(self.input_param.lib, self.input_param.decoy_method,
                                 self.input_param.mz_min, self.input_param.mz_max, self.input_param.seed,
                                 self.input_param.n_thread, self.input_param.lib_filter, self.logger)

        self.lib_cols_org, self.lib_data_org, self.temp_lib_path = lib_process.deal_process(
            self.input_param.protein_infer_key, self.input_param.lib_load_version)

        self.lib_max_intensity = self.lib_data_org[self.lib_cols_org['LIB_INTENSITY_COL']].max()

        self.lib_cols, self.lib_data = lib_handler.base_load_lib(self.lib_cols_org, self.lib_data_org, None,
                                                                 intersection=False)
        self.lib_data['transition_group_id'] = pd.Categorical(self.lib_data['transition_group_id'])

        if self.input_param.open_identify:
            for dd, mzml_path in enumerate(self.mzml_files):
                tt1 = time.time()
                self.logger.info('Processing {}/{}, {}'.format(dd, len(self.mzml_files), mzml_path))
                runtime_data_info.runtime_data.current_mzml_index = dd
                runtime_data_info.runtime_data.current_is_success = True
                self.deal_each_mzml(mzml_path, self.lib_max_intensity)
                self.deal_num = dd + 1
                tt2 = time.time()
                self.logger.info('Each mzml spend time: {}'.format(tt2 - tt1))

        #
        if self.input_param.open_quantification:
            qp = QuantificationProcess(self.temp_lib_path, self.input_param.out_path, self.mzml_files,
                                       protein_infer_key=self.input_param.protein_infer_key,
                                       max_workers=self.input_param.n_thread, logger=self.logger)
            qp.deal_process()

        try:
            #
            if self.input_param.clear_data:
                self.logger.info('Clear temp data start')
                #
                for dd, mzml_path in enumerate(self.mzml_files):
                    mzml_name = os.path.split(mzml_path)[-1]
                    rawdata_prefix = mzml_name[:-5]
                    base_raw_out_dir = os.path.join(self.input_param.out_path, rawdata_prefix)
                    #
                    identify_data_dir = os.path.join(base_raw_out_dir, 'identify_data')
                    if os.path.exists(identify_data_dir):
                        shutil.rmtree(identify_data_dir)
                    #
                    finetune_data_dir = os.path.join(base_raw_out_dir, 'finetune', 'data')
                    if os.path.exists(finetune_data_dir):
                        shutil.rmtree(finetune_data_dir)
                    quant_data_dir = os.path.join(base_raw_out_dir, 'quant', 'data')
                    if os.path.exists(quant_data_dir):
                        shutil.rmtree(quant_data_dir)
                self.logger.info('Clear temp data over')
        except Exception:
            self.logger.exception('Clear temp data exception')
        if runtime_data_info.runtime_data.current_is_success:
            msg_send_utils.send_msg(status=ProgressStepStatusEnum.END)
        else:
            msg_send_utils.send_msg(status=ProgressStepStatusEnum.FAIL_END)

    def get_file_list(self):
        with open(self.input_param.rawdata_file_dir_path, mode='r') as f:
            file_path_list = f.readlines()
            for file_path in file_path_list:
                self.mzml_files.append(file_path.strip())

    def deal_each_mzml(self, mzml_path, lib_max_intensity):
        mzml_name = os.path.split(mzml_path)[-1]
        self.current_mzml_name = mzml_name
        self.rawdata_prefix = mzml_name[:-5]
        base_raw_out_dir = os.path.join(self.input_param.out_path, self.rawdata_prefix)
        try:
            self.mzml_instrument = self.input_param.instrument


            self.deal_each_mzml_identify(mzml_path, mzml_name)
            #
            if not self.input_param.open_finetune:
                return
            fe = FeatureEngineer()
            rt_index = fe.get_rt_s2i(self.mzml_rt)
            instrument_index = fe.get_instrument_s2i(self.mzml_instrument)

            fp = FinetunePrecursorPeakProcess(self.current_mzml_name, base_raw_out_dir,
                                              self.input_param.n_thread, self.input_param.finetune_score_limit,
                                              each_pkl_size=self.input_param.train_pkl_size,
                                              rt_index=rt_index, instrument_index=instrument_index,
                                              lib_max_intensity=lib_max_intensity, logger=self.logger)
            if self.input_param.open_finetune_peak:
                fp.peak_score_precursor()
            tp = FinetuneTrainProcess(self.current_mzml_name, base_raw_out_dir,
                                      train_epochs=self.input_param.train_epochs,
                                      base_model_path=self.input_param.finetune_base_model_file,
                                      env=self.input_param.env,
                                      gpu_device_list=self.input_param.gpu_devices,
                                      device=self.input_param.device,
                                      logger=self.logger)
            if self.input_param.open_finetune_train:
                tp.start_train()
            ep = EvalProcess(self.current_mzml_name, base_raw_out_dir, train_epochs=self.input_param.train_epochs,
                             env=self.input_param.env,
                             gpu_device_list=self.input_param.gpu_devices,
                             device=self.input_param.device, logger=self.logger)

            if self.input_param.open_eval:
                ep.eval()
            #
            pp = ProteinInferProcess(self.temp_lib_path, self.input_param.protein_infer_key, base_raw_out_dir,
                                     mzml_name, logger=self.logger)
            if self.input_param.open_protein_infer:
                pp.deal_process()

            qp = QuantProcess(self.rawdata_prefix, mzml_name, base_raw_out_dir, rt_index, instrument_index,
                              each_pkl_size=self.input_param.quant_pkl_size,
                              pred_quant_model_path=self.input_param.quant_model_file, env=self.input_param.env,
                              lib_max_intensity=lib_max_intensity,
                              gpu_device_list=self.input_param.gpu_devices,
                              device=self.input_param.device, logger=self.logger)
            if self.input_param.open_quant:
                qp.deal_process()

            #
            rbp = ResultBuildProcess(base_raw_out_dir, self.rawdata_prefix, mzml_name, self.logger)
            rbp.deal_process()

        except Exception:
            self.logger.exception('Each mzml identify exception')

        try:
            #
            precursor_csv_file = os.path.join(base_raw_out_dir, '{}_precursor.csv'.format(self.rawdata_prefix))
            if not os.path.exists(precursor_csv_file):
                save_df = pd.DataFrame(
                    columns=constant.OUTPUT_PRECURSOR_COLUMN_LIST)
                save_df.to_csv(precursor_csv_file, index=False)

            protein_csv_file = os.path.join(base_raw_out_dir, '{}_protein.csv'.format(self.rawdata_prefix))
            if not os.path.exists(protein_csv_file):
                save_df = pd.DataFrame(columns=constant.OUTPUT_PROTEIN_COLUMN_LIST)
                save_df.to_csv(protein_csv_file, index=False)
        except Exception:
            pass

        try:
            #
            if self.input_param.clear_data:
                self.logger.info('Clear temp data start')
                base_raw_out_dir = os.path.join(self.input_param.out_path, self.rawdata_prefix)
                #
                identify_data_dir = os.path.join(base_raw_out_dir, 'identify_data')
                if os.path.exists(identify_data_dir):
                    shutil.rmtree(identify_data_dir)
                #
                finetune_data_dir = os.path.join(base_raw_out_dir, 'finetune', 'data')
                if os.path.exists(finetune_data_dir):
                    shutil.rmtree(finetune_data_dir)
                quant_data_dir = os.path.join(base_raw_out_dir, 'quant', 'data')
                if os.path.exists(quant_data_dir):
                    shutil.rmtree(quant_data_dir)
                self.logger.info('Clear temp data over')
        except Exception:
            self.logger.exception('Clear temp data exception')
        if runtime_data_info.runtime_data.current_is_success:
            msg_send_utils.send_msg(status=ProgressStepStatusEnum.END)
        else:
            msg_send_utils.send_msg(status=ProgressStepStatusEnum.FAIL_END)

    def deal_each_mzml_identify(self, mzml_path, mzml_name):
        logger = self.logger

        mzml_dir_path = os.path.split(mzml_path)[0]
        logger.info('Processing identify file {}. {}/{}'.format(mzml_name, self.deal_num, len(self.mzml_files)))
        ms1, ms2, win_range = raw_handler.load_and_temp_raw(mzml_dir_path, mzml_name, self.input_param.mz_min,
                                                            self.input_param.mz_max,
                                                            rt_unit=self.input_param.raw_rt_unit,
                                                            skip_no_temp=self.input_param.skip_no_temp,
                                                            logger=self.logger)
        if ms1 is None or ms2 is None or win_range is None:
            self.logger.error('File {} temp raw info not exist'.format(mzml_path))
            return

        mzml_rt = math.ceil(math.ceil(ms1.rt_list[-1]) / 60)
        #
        self.mzml_rt = instrument_info_utils.get_mzml_nearest_rt(mzml_rt)
        self.logger.info('mzml rt is: {}'.format(self.mzml_rt))

        if not self.input_param.open_base_identify:
            return

        pick_rt0 = time.time()
        rt_model_params = self.process_rt_model_params(mzml_path, ms1, ms2, win_range)
        pick_rt1 = time.time()
        logger.debug('pick rt time: {}'.format(pick_rt1 - pick_rt0))
        if rt_model_params is None:
            logger.info('rt model params is None, skip. mzml_name = {}'.format(mzml_name))
            return

        try:
            min_rt, max_rt = self.get_min_max_rt()
            logger.debug('get minmax rt: {}, {}'.format(min_rt, max_rt))

            precursor_id_all = self.lib_data[self.lib_cols['PRECURSOR_ID_COL']].unique()
            precursor_id_list_arr = list_utils.list_split(precursor_id_all, self.input_param.batch_size)

            gpu_device_list = self.input_param.gpu_devices

            all_deal_num = len(precursor_id_list_arr)
            runtime_data_info.runtime_data.current_identify_all_num = all_deal_num

            base_raw_out_dir, raw_out_dir, error_file = self.clear_output_dir(self.rawdata_prefix)
            msg_send_utils.send_msg(step=ProgressStepEnum.SCREEN, status=ProgressStepStatusEnum.RUNNING,
                                    msg='Processing screen, batch num is {}'.format(all_deal_num))

            if len(gpu_device_list) == 1:
                self.each_precursor_iden_process(precursor_id_list_arr, mzml_name, ms1, ms2, win_range, rt_model_params,
                                                 min_rt, max_rt, 0, gpu_device_list[0])
            else:
                #
                #
                device_precursor_id_arr = list_utils.divide_list(precursor_id_list_arr, len(gpu_device_list))
                base_each_num = 0
                processes = []
                for process_num, each_device_precursor_id_list in enumerate(device_precursor_id_arr):
                    process = Process(
                        target=self.each_precursor_iden_process,
                        args=(
                            each_device_precursor_id_list, mzml_name, ms1, ms2, win_range, rt_model_params,
                            min_rt, max_rt, base_each_num, gpu_device_list[process_num])
                    )
                    base_each_num = base_each_num + len(each_device_precursor_id_list)
                    processes.append(process)
                    process.start()

                for process in processes:
                    process.join()

            #
            msg_send_utils.send_msg(step=ProgressStepEnum.SCREEN, status=ProgressStepStatusEnum.SUCCESS,
                                    msg='Screen success')
            with open(os.path.join(raw_out_dir, 'over.txt'), mode='w+') as f:
                f.write('over')
        except Exception as e:
            logger.exception('mzml: {} identify exception.'.format(mzml_path))
            msg_send_utils.send_msg(step=ProgressStepEnum.SCREEN, status=ProgressStepStatusEnum.ERROR,
                                    msg='Screen exception: {}'.format(e))
            runtime_data_info.runtime_data.current_is_success = False
        finally:
            pass

    def get_min_max_rt(self):
        #
        if self.input_param.tp_version == 'v1':
            min_rt, max_rt = timepoint_handler_v1.get_min_max_rt(self.input_param, self.rawdata_prefix)
        elif self.input_param.tp_version == 'v3':
            min_rt, max_rt = timepoint_handler_v3.get_min_max_rt(self.input_param, self.rawdata_prefix)
        else:
            raise Exception('tp version error')
        return min_rt, max_rt

    def process_rt_model_params(self, mzml_path, ms1, ms2, win_range):
        if self.input_param.tp_version == 'v1':
            rt_model_params = timepoint_handler_v1.get_rt_model_params(self.input_param, self.rawdata_prefix,
                                                                       self.lib_prefix,
                                                                       self.lib_cols_org, self.lib_data_org, mzml_path,
                                                                       ms1,
                                                                       ms2, win_range, self.logger)
        elif self.input_param.tp_version == 'v3':
            wait_deal_score_queue = Queue(maxsize=2, )
            #
            first_gpu_device = self.input_param.device
            rt_score_device = torch.device(first_gpu_device)
            rt_diart_model = model_handler.load_model(self.input_param.xrm_model_file, rt_score_device)
            rt_sc_deal_thread = ScorePredictThread('rt score deal', rt_diart_model, rt_score_device,
                                                   self.input_param.out_path, self.input_param.n_cycles,
                                                   self.input_param.model_cycles,
                                                   self.input_param.frag_repeat_num,
                                                   self.input_param.step_size,
                                                   self.lib_max_intensity, wait_deal_score_queue, self.logger,
                                                   self.input_param.ext_frag_quant_fragment_num,
                                                   self.input_param.ext_frag_quant_zero_type,
                                                   self.input_param.ext_quant_data_open_smooth)

            rt_model_params = timepoint_handler_v3.get_rt_model_params(self.input_param, self.rawdata_prefix,
                                                                       self.lib_prefix,
                                                                       self.lib_cols_org, self.lib_data_org, mzml_path,
                                                                       ms1,
                                                                       ms2, win_range, self.logger, rt_sc_deal_thread)
        else:
            raise Exception('tp version error')
        #
        torch.cuda.empty_cache()
        return rt_model_params

    def clear_output_dir(self, rawdata_prefix):
        base_raw_out_dir = os.path.join(self.input_param.out_path, rawdata_prefix)
        raw_out_dir = os.path.join(base_raw_out_dir, 'identify_data')
        if self.input_param.clear_data:
            if os.path.exists(raw_out_dir):
                shutil.rmtree(raw_out_dir)
        if not os.path.exists(raw_out_dir):
            os.makedirs(raw_out_dir)
        error_file = os.path.join(raw_out_dir, 'error.txt')
        if self.input_param.clear_data:
            if os.path.exists(error_file):
                os.remove(error_file)
        return base_raw_out_dir, raw_out_dir, error_file

    def each_precursor_iden_process(self, precursor_id_list_arr, mzml_name, ms1, ms2, win_range, rt_model_params,
                                    min_rt, max_rt, base_each_num, process_gpu_device_num):
        process_param = copy.deepcopy(self.input_param)
        process_param.device = 'cuda:' + str(process_gpu_device_num)
        log_file_name = os.path.basename(process_param.logger_file_path)
        gpu_log_file_name = log_file_name.removesuffix('.log') + f'_CUDA_{process_gpu_device_num}.log'
        logger, logger_file_path = create_new_logger(process_param.out_path, log_file_name=gpu_log_file_name)
        logger.info('process_gpu_device_num: {}'.format(process_gpu_device_num))
        base_process = BaseIdentifyProcess(process_param, self.rawdata_prefix, self.lib_cols, self.lib_data, ms1, ms2,
                                           win_range, rt_model_params, min_rt,
                                           max_rt, self.mzml_rt, self.mzml_instrument, self.lib_max_intensity,
                                           process_gpu_device_num, base_each_num, logger)
        base_process.deal_process(precursor_id_list_arr)
