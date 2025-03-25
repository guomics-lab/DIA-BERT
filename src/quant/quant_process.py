import os
import os.path
import pickle
import shutil

import lightning.pytorch as ptl
import numpy as np
import pandas as pd
import torch
import yaml
from lightning.pytorch.strategies import DDPStrategy
from src.common.model.quant_dataset import Dataset
from src.common.model.quant_eval_model import Evalute
from src.common.model.quant_model import AreaModel
from src.common.model.score_model import FeatureEngineer

from src.common import constant
from src.common import runtime_data_info
from src.common.constant import ProgressStepEnum, ProgressStepStatusEnum
from src.quant.dataset_quant import create_iterable_dataset
from src.utils import msg_send_utils
from src.utils.finetune_utils import set_seeds, mkdir_p


class QuantProcess():
    def __init__(self, rawdata_prefix, mzml_name, base_output, rt_index, instrument_index, each_pkl_size=2048,
                 pred_quant_model_path='./resource/model/quant.ckpt', env='linux', gpu_device_list=None, device=None, lib_max_intensity=None, logger=None):

        self.base_output = base_output
        self.rawdata_prefix = rawdata_prefix
        self.mzml_name = mzml_name
        self.each_pkl_size = each_pkl_size
        self.pkl_dir = os.path.join(self.base_output, 'identify_data')

        self.rt_index = rt_index
        self.instrument_index = instrument_index
        self.lib_max_intensity = lib_max_intensity

        self.pred_quant_config = './config/pred_quant.yaml'
        self.pred_quant_model_path = pred_quant_model_path

        self.gpu_device_list = gpu_device_list
        self.device = device

        self.env = env

        self.model_name = 'sum6'

        self.logger = logger

    def deal_process(self):
        self.logger.info('Processing quant')
        try:
            if not runtime_data_info.runtime_data.current_is_success:
                msg_send_utils.send_msg(step=ProgressStepEnum.QUANT, status=ProgressStepStatusEnum.ERROR)
                self.logger.info('Finished quant')
                return
            self.peak_rsm()
            self.pred_quant()
        except Exception as e:
            self.logger.exception('Processing quant exception')
            msg_send_utils.send_msg(step=ProgressStepEnum.QUANT, status=ProgressStepStatusEnum.ERROR,
                                    msg='Quant exception: {}'.format(e))
        msg_send_utils.send_msg(step=ProgressStepEnum.QUANT, status=ProgressStepStatusEnum.SUCCESS,
                                msg='Finished quant')
        self.logger.info('Finished quant')

    '''
    
    '''

    def pred_quant(self):
        msg_send_utils.send_msg(msg='Processing pred quant')
        with open(self.pred_quant_config) as f_in:
            config = yaml.safe_load(f_in)
        set_seeds(config['seed'])
        device = torch.device(self.device)
        out_path = os.path.join(self.base_output, 'quant', 'output')
        config["out_path"] = out_path
        mkdir_p(config["out_path"])

        config["model_path"] = self.pred_quant_model_path
        config["data_path"] = os.path.join(self.base_output, 'quant', 'data')

        dl = create_iterable_dataset(config['data_path'], self.logger, config, parse='quant')

        #
        one_epoch_iters = int(len(dl))
        config["step_scale"] = int(one_epoch_iters * float(config["step_ratio"]))

        if self.env == constant.env_linux:
            strategy = DDPStrategy(gradient_as_bucket_view=True, find_unused_parameters=True)
        else:
            strategy = DDPStrategy(gradient_as_bucket_view=True, find_unused_parameters=True,
                                   process_group_backend="gloo")
        trainer = ptl.Trainer(
            accelerator="auto",
            devices=[self.gpu_device_list[0]],
            strategy=strategy,
            enable_progress_bar=False,
        )
        # model_name = self.pred_quant_model_path.split('/')[-1].split('.')[0].replace('=', '')
        self.logger.info(f"model_name:  {self.model_name}, device: {device}")
        model = AreaModel.load(self.pred_quant_model_path)
        model.to(device)
        evaluate = Evalute(config, model, self.model_name)
        trainer.test(evaluate, dataloaders=dl)
        msg_send_utils.send_msg(msg='Finished pred quant')

    '''
    
    '''

    def peak_rsm(self):
        self.logger.info('Processing peak precursor rsm info for finetune')
        msg_send_utils.send_msg(msg='Processing peak precursor rsm info for finetune')
        quant_dir = os.path.join(self.base_output, 'quant')
        if os.path.exists(quant_dir):
            shutil.rmtree(quant_dir)
        quant_pkl_dir = os.path.join(quant_dir, 'data')
        os.makedirs(quant_pkl_dir)

        #
        precursor_fdr_path = os.path.join(self.base_output, 'finetune', 'output',
                                          'fdr_{}_eval.csv'.format(self.mzml_name))
        if not os.path.exists(precursor_fdr_path):
            self.logger.error('Precursor fdr file is not exist, {}'.format(precursor_fdr_path))
            msg_send_utils.send_msg(msg='Precursor fdr file is not exist, {}'.format(precursor_fdr_path))
            return
        df = pd.read_csv(precursor_fdr_path)
        need_precursor_ids = set(df['transition_group_id'].tolist())
        if len(need_precursor_ids) == 0:
            self.logger.info('Quant precursor is empty')
            msg_send_utils.send_msg(msg='Quant precursor is empty')
            return
        self.logger.info(f'Quant precursor num is {len(need_precursor_ids)}')

        pkl_list = os.listdir(self.pkl_dir)
        pkl_list = list(filter(lambda entry: entry.endswith('.pkl'), pkl_list))
        #
        precursor_id_list, rsm_list, frag_info_list, precursor_feat_list, target_info_list = [], [], [], [], []
        feature_engineer = FeatureEngineer()
        save_pkl_index = 0
        for pkl_name in pkl_list:
            pkl_path = os.path.join(self.pkl_dir, pkl_name)
            with open(pkl_path, mode='rb') as f:
                precursor, precursor_feat, rsm, frag_info, score = pickle.load(f)

                precursor_np = np.array(precursor)
                pkl_precursor_id_list = precursor_np[:, 0].tolist()
                peak_index_list = [index for index, precursor in enumerate(pkl_precursor_id_list) if
                                   precursor in need_precursor_ids]
                if len(peak_index_list) == 0:
                    continue
                precursor_id = [pkl_precursor_id_list[index] for index in peak_index_list]
                rsm = rsm[peak_index_list]
                frag_info = frag_info[peak_index_list]
                precursor_feat = precursor_feat[peak_index_list]

                precursor_id_list.extend(precursor_id)  # precursor_id
                rsm_list.append(rsm)
                frag_info_list.append(frag_info)
                precursor_feat_list.append(precursor_feat)

                target_info_list.extend([1 - int(x) for x in precursor_np[:, 1][peak_index_list].tolist()])

                if len(precursor_id_list) >= self.each_pkl_size:
                    #
                    all_rsm = np.concatenate(rsm_list, axis=0)
                    all_frag_info = np.concatenate(frag_info_list, axis=0)
                    all_precursor_feat = np.concatenate(precursor_feat_list, axis=0)

                    #
                    save_precursor_id_list = precursor_id_list[:self.each_pkl_size]
                    save_label_list = target_info_list[:self.each_pkl_size]
                    save_rsm = all_rsm[:self.each_pkl_size]
                    save_frag_info = all_frag_info[:self.each_pkl_size]
                    save_precursor_feat = all_precursor_feat[:self.each_pkl_size]

                    save_rsm, rsm_max = FeatureEngineer.process_intensity_np(save_rsm)
                    save_rsm = save_rsm.swapaxes(1, 2)
                    save_frag_info = feature_engineer.process_frag_info(save_frag_info,
                                                                        max_intensity=self.lib_max_intensity)

                    save_precursor_feat = np.column_stack((save_precursor_feat[:, :7], rsm_max))
                    save_precursor_feat = feature_engineer.process_feat_np(save_precursor_feat)

                    #
                    pr_ids = len(save_precursor_id_list)
                    rt_np = np.array([self.rt_index]).repeat(pr_ids).reshape(-1, 1)
                    instrument_np = np.array([self.instrument_index]).repeat(pr_ids).reshape(-1, 1)
                    save_precursor_feat = np.concatenate((save_precursor_feat, rt_np, instrument_np), axis=1)

                    data_set = Dataset()
                    data_set.rsm = save_rsm
                    data_set.feat = save_precursor_feat
                    data_set.frag_info = save_frag_info
                    data_set.label = save_label_list
                    data_set.precursor_id = save_precursor_id_list
                    data_set.file = [self.mzml_name for _ in range(len(save_precursor_id_list))]
                    with open(os.path.join(quant_pkl_dir, 'batch_{}.pkl'.format(save_pkl_index)), mode='wb') as f:
                        f.write(pickle.dumps(data_set, protocol=4))
                    #
                    temp_precursor_id_list = precursor_id_list[self.each_pkl_size:]
                    temp_target_info_list = target_info_list[self.each_pkl_size:]
                    precursor_id_list, rsm_list, frag_info_list, precursor_feat_list, target_info_list = [], [], [], [], []
                    precursor_id_list.extend(temp_precursor_id_list)
                    rsm_list.append(all_rsm[self.each_pkl_size:])
                    frag_info_list.append(all_frag_info[self.each_pkl_size:])
                    precursor_feat_list.append(all_precursor_feat[self.each_pkl_size:])
                    target_info_list.extend(temp_target_info_list)
                    save_pkl_index = save_pkl_index + 1

        #
        if len(precursor_id_list) > 0:
            #
            all_rsm = np.concatenate(rsm_list, axis=0)
            all_frag_info = np.concatenate(frag_info_list, axis=0)
            all_precursor_feat = np.concatenate(precursor_feat_list, axis=0)
            save_rsm, rsm_max = FeatureEngineer.process_intensity_np(all_rsm)
            save_rsm = save_rsm.swapaxes(1, 2)
            save_frag_info = feature_engineer.process_frag_info(all_frag_info, max_intensity=self.lib_max_intensity)
            save_precursor_feat = np.column_stack((all_precursor_feat[:, :7], rsm_max))
            save_precursor_feat = feature_engineer.process_feat_np(save_precursor_feat)
            #
            pr_ids = len(precursor_id_list)
            rt_np = np.array([self.rt_index]).repeat(pr_ids).reshape(-1, 1)
            instrument_np = np.array([self.instrument_index]).repeat(pr_ids).reshape(-1, 1)
            save_precursor_feat = np.concatenate((save_precursor_feat, rt_np, instrument_np), axis=1)
            data_set = Dataset()
            data_set.rsm = save_rsm
            data_set.feat = save_precursor_feat
            data_set.frag_info = save_frag_info
            data_set.label = target_info_list
            data_set.precursor_id = precursor_id_list
            data_set.file = [self.mzml_name for _ in range(len(precursor_id_list))]
            with open(os.path.join(quant_pkl_dir, 'batch_{}.pkl'.format(save_pkl_index)), mode='wb') as f:
                f.write(pickle.dumps(data_set, protocol=4))
        self.logger.info('Finished peak precursor rsm info for finetune')
        msg_send_utils.send_msg(msg='Finished peak precursor rsm info for finetune')
