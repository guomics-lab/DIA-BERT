
import os

import lightning.pytorch as ptl
import pandas as pd
import torch
import yaml
from lightning.pytorch.strategies import DDPStrategy
from src.common.model.eval_model import Evalute
from src.common.model.score_model import DIArtModel
from torch.utils.tensorboard import SummaryWriter

from src.common import constant
from src.common import runtime_data_info
from src.common.constant import ProgressStepEnum, ProgressStepStatusEnum
from src.finetune.dataset import combine_data
from src.finetune.utils import set_seeds, mkdir_p
from src.utils import msg_send_utils

'''

'''
class EvalProcess():

    def __init__(self, mzml_name=None, base_output=None, train_epochs=10, env='linux', gpu_device_list=None,
                 device=None, logger=None):
        self.mzml_name = mzml_name
        self.base_output = base_output
        self.train_epochs = train_epochs
        self.gpu_device_list = gpu_device_list
        self.device = device
        self.env = env
        self.logger = logger

    def eval(self):
        if not runtime_data_info.runtime_data.current_is_success:
            msg_send_utils.send_msg(step=ProgressStepEnum.FINETUNE_EVAL, status=ProgressStepStatusEnum.ERROR)
            return
        self.logger.info('Processing eval train model')
        try:
            msg_send_utils.send_msg(step=ProgressStepEnum.FINETUNE_EVAL, status=ProgressStepStatusEnum.RUNNING,
                                    msg='Processing eval train model')

            torch.set_float32_matmul_precision = 'high'

            base_config_path = './config/finetune/base_train.yaml'

            #
            with open(base_config_path) as f_in:
                config = yaml.safe_load(f_in)

            model_save_folder_path = os.path.join(self.base_output, 'finetune', 'model')

            finetune_data_path = os.path.join(self.base_output, 'finetune', 'data')
            tb_summarywriter = os.path.join(self.base_output, 'finetune', 'logs')
            model_train_output_path = os.path.join(self.base_output, 'finetune', 'output')
            mkdir_p(model_save_folder_path)
            mkdir_p(tb_summarywriter)
            mkdir_p(model_train_output_path)

            config['task_name'] = 'eval'

            sw = SummaryWriter(config["tb_summarywriter"])
            set_seeds(config['seed'])
            config['model_path'] = model_save_folder_path
            config['model_save_folder_path'] = model_save_folder_path
            config['data_path'] = finetune_data_path
            # config['model_path'] = base_model_path
            config['tb_summarywriter'] = tb_summarywriter
            config['out_path'] = model_train_output_path

            torch_device = torch.device(self.device)

            # device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
            metrics_finetune_csv_path = os.path.join(self.base_output, 'metrics_finetune.csv')
            csv_df = pd.read_csv(metrics_finetune_csv_path, header=None)
            csv_df.columns = ['file_name', 'epoch', 'loss', 'auc', 'acc', 'dt', 'model_name']
            model_name = csv_df.loc[csv_df['loss'].idxmin(), 'model_name']
            model_path = os.path.join(config['model_path'], model_name)
            self.logger.info('Get min loss model, {}'.format(model_path))
            msg_send_utils.send_msg(msg='Min loss model is: {}'.format(model_path))

            #
            config['epochs'] = self.train_epochs
            config['phase'] = 'final'

            dl = combine_data(config, phase=config['phase'])
            #
            if self.env == constant.env_linux:
                strategy = DDPStrategy(gradient_as_bucket_view=True, find_unused_parameters=True)
            else:
                strategy = DDPStrategy(gradient_as_bucket_view=True, find_unused_parameters=True, process_group_backend="gloo")
            trainer = ptl.Trainer(
                accelerator="auto",
                devices=[self.gpu_device_list[0]],
                strategy=strategy,
                enable_progress_bar=False,
            )

            model = DIArtModel.load(model_path)
            model.half()
            model.to(torch_device)

            msg_send_utils.send_msg(msg='Start eval model')
            evaluate = Evalute(config, model, model_name, sw)
            trainer.test(evaluate, dataloaders=dl)
            msg_send_utils.send_msg(step=ProgressStepEnum.FINETUNE_EVAL, status=ProgressStepStatusEnum.SUCCESS,
                                    msg='Finish eval model')
        except Exception as e:
            self.logger.exception('Finetune eval exception')
            runtime_data_info.runtime_data.current_is_success = False
            msg_send_utils.send_msg(step=ProgressStepEnum.FINETUNE_EVAL, status=ProgressStepStatusEnum.ERROR, msg='Eval model exception: {}'.format(e))

