import os
from datetime import datetime
from typing import Any

import lightning.pytorch as ptl
import numpy as np
import torch
import yaml
from lightning.pytorch.strategies import DDPStrategy
from src.common.model.score_model import DIArtModel
from src.common.model.train_pt_module import PTModule, WarmupScheduler, Optimizers
from torch.utils.tensorboard import SummaryWriter

from src.common import constant
from src.common.constant import ProgressStepEnum, ProgressStepStatusEnum
from src.finetune.dataset import combine_data
from src.finetune.utils import set_seeds, mkdir_p
from src.utils import msg_send_utils


class FinetuneTrainProcess(object):

    def __init__(self, mzml_name=None, base_output=None, train_epochs=10,
                 base_model_path='./resource/model/finetune_model.ckpt', env='linux', gpu_device_list=None, device=None, logger=None):
        self.mzml_name = mzml_name
        self.base_output = base_output
        self.train_epochs = train_epochs
        self.base_model_path = base_model_path

        self.gpu_device_list = gpu_device_list
        self.device = device

        self.env = env
        self.logger = logger

    def start_train(self):
        self.logger.info('Processing finetune train model')
        msg_send_utils.send_msg(step=ProgressStepEnum.FINETUNE_TRAIN, status=ProgressStepStatusEnum.RUNNING,
                                msg='Processing finetune train model, train epochs: {}'.format(self.train_epochs))

        torch.set_float32_matmul_precision = 'high'

        base_config_path = './config/finetune/base_train.yaml'

        #
        with open(base_config_path) as f_in:
            config = yaml.safe_load(f_in)

        model_save_folder_path = os.path.join(self.base_output, 'finetune', 'model')
        model_train_output_path = os.path.join(self.base_output, 'finetune', 'output')
        final_model_path = os.path.join(model_save_folder_path, 'finetune.ckpt')
        base_model_path = self.base_model_path
        finetune_data_path = os.path.join(self.base_output, 'finetune', 'data')

        #
        sp_train_feat_dir = os.path.join(finetune_data_path, 'sp_train_feat')
        sp_test_feat_dir = os.path.join(finetune_data_path, 'sp_test_feat')
        if not os.path.exists(sp_test_feat_dir) or not os.path.exists(sp_train_feat_dir) or len(
                os.listdir(sp_train_feat_dir)) == 0 or len(os.listdir(sp_test_feat_dir)) == 0:
            self.logger.error('There is no finetune data')
            msg_send_utils.send_msg(step=ProgressStepEnum.FINETUNE_TRAIN, status=ProgressStepStatusEnum.ERROR,
                                    msg='There is no finetune data')
            return False

        tb_summarywriter = os.path.join(self.base_output, 'finetune', 'logs')
        mkdir_p(model_save_folder_path)
        mkdir_p(tb_summarywriter)
        set_seeds(config['seed'])

        #
        config['epochs'] = self.train_epochs
        config['model_save_folder_path'] = model_save_folder_path
        config['final_model_path'] = final_model_path
        config['data_path'] = finetune_data_path
        config['model_path'] = base_model_path
        config['out_path'] = model_train_output_path
        config['tb_summarywriter'] = tb_summarywriter
        config['task_name'] = 'finetune'
        config['metrics_out_path'] = self.base_output

        self.train_process(config, base_model_path)
        self.logger.info('Finish finetune train model')
        msg_send_utils.send_msg(step=ProgressStepEnum.FINETUNE_TRAIN, status=ProgressStepStatusEnum.SUCCESS,
                                msg='Finish finetune train model')

    def train_process(self, config, model_path):
        """Training function."""
        config["tb_summarywriter"] = os.path.join(config["tb_summarywriter"], datetime.now().strftime(
            "diart_train_%y_%m_%d_%H_%M_%S"
        ))
        mkdir_p(config["tb_summarywriter"])
        sw = SummaryWriter(config["tb_summarywriter"])
        self.logger.info(f"Train begin!!! GPU nums: {torch.cuda.device_count()}, epoch: {config['epochs']}")
        msg_send_utils.send_msg(msg=f"Train begin!!! GPU nums: {torch.cuda.device_count()}, epoch: {config['epochs']}")

        train_dl = combine_data(config, phase='train')
        val_dl = combine_data(config, phase='val')
        self.logger.info(f"Updates the iter of per epoch is: train={len(train_dl):,}, val{len(val_dl)}"
                         f", optim_weight_part_decay: {bool(config['optim_weight_part_decay']):,}")
        msg_send_utils.send_msg(
            msg="Updates the iter of per epoch is: train: {}, val: {}, optim_weight_part_decay: {}".format(
                len(train_dl), len(val_dl), bool(config['optim_weight_part_decay'])))

        #
        # torch.cuda.set_device(0)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_device = torch.device(self.device)

        #
        mkdir_p(config["out_path"])
        mkdir_p(config["tb_summarywriter"])
        mkdir_p(config["model_save_folder_path"])
        mkdir_p(config["out_path"])

        #
        one_epoch_iters = len(train_dl)
        config["one_epoch_iters"] = one_epoch_iters
        config["train_step_scale"] = max(int(one_epoch_iters * float(config["train_step_ratio"])), 1)
        config["ckpt_interval"] = one_epoch_iters
        self.logger.info(
            f"Updates train_step_scale is : {config['train_step_scale']}"
            f" ckpt interval={config['ckpt_interval']}"
        )

        # init model
        if bool(config["resume"]):
            model, optim, scheduler = self.resume_model(model_path, torch_device, config)
        else:
            model, optim, scheduler = self.init_model(model_path, torch_device, config)

        ptmodel = PTModule(config, model, sw, optim, scheduler)

        if config["save_model"]:
            callbacks = [
                ptl.callbacks.ModelCheckpoint(
                    dirpath=config["model_save_folder_path"],
                    save_top_k=-1,
                    save_weights_only=config["save_weights_only"],
                    every_n_train_steps=config["ckpt_interval"],
                ),

            ]
        else:
            callbacks = None

        self.logger.info("Initializing PL trainer., epoch: {}".format(config["epochs"]))

        if config["train_strategy"] == 'ddp':
            if self.env == constant.env_linux:
                strategy = DDPStrategy(gradient_as_bucket_view=True, find_unused_parameters=True)
            else:
                strategy = DDPStrategy(gradient_as_bucket_view=True, find_unused_parameters=True,
                                       process_group_backend="gloo")
        else:
            strategy = config["train_strategy"]

        #
        trainer = ptl.Trainer(
            accelerator="auto",
            devices=[self.gpu_device_list[0]],
            precision="16-mixed",  # 混合精度
            callbacks=callbacks,
            max_epochs=config["epochs"],
            num_sanity_val_steps=config["num_sanity_val_steps"],
            accumulate_grad_batches=config["grad_accumulation"],
            gradient_clip_val=config["gradient_clip_val"],
            strategy=strategy,
            enable_progress_bar=False,
        )

        if config["train_strategy"] in ['deepspeed_stage_1', 'deepspeed_stage_2', 'deepspeed_stage_2_offload']:
            trainer.strategy.config["zero_force_ds_cpu_optimizer"] = False

        metrics_finetune_csv_path = os.path.join(self.base_output, 'metrics_finetune.csv')
        if os.path.exists(metrics_finetune_csv_path):
            os.remove(metrics_finetune_csv_path)

        try:
            self.logger.info('Start train model')
            msg_send_utils.send_msg(msg='Start train model')
            # Train the model.
            trainer.fit(ptmodel, train_dl, val_dl)
            msg_send_utils.send_msg(step=ProgressStepEnum.FINETUNE_TRAIN, status=ProgressStepStatusEnum.SUCCESS,
                                    msg='Finish train model')
        except Exception as e:
            self.logger.exception('Finetune train model error')
            # self.logger.info("error: {}".format(e))
            msg_send_utils.send_msg(step=ProgressStepEnum.FINETUNE_TRAIN, status=ProgressStepStatusEnum.ERROR,
                                    msg='Finetune train model error: {}'.format(e))
        finally:
            self.logger.info("model save !!")

    def resume_model(self, ckpt_path: str,
                     device: str,
                     config: dict[str, Any]):
        model = DIArtModel()
        ckpt = torch.load(ckpt_path, map_location=device)

        #
        model_state = {k.replace("model.", ""): v for k, v in ckpt['state_dict'].items()}

        k_missing = np.sum(
            [x not in list(model_state.keys()) for x in list(model.state_dict().keys())]
        )
        if k_missing > 0:
            self.logger.info(f"Model checkpoint is missing {k_missing} keys!")

        k_missing = np.sum(
            [x not in list(model.state_dict().keys()) for x in list(model_state.keys())]
        )
        if k_missing > 0:
            self.logger.info(f"Model state is missing {k_missing} keys!")
        model.load_state_dict(model_state, strict=False)
        model = model.to(device)

        #
        if bool(config["freeze"]):
            self.logger.info(f"freeze model, except out linear!!!")
            model = self.freeze_model(model)

        #
        if bool(config["optim_weight_part_decay"]):
            # optimer
            optim = Optimizers(model, config)
        else:
            # optimer
            optim = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=float(config["learning_rate"]),
                weight_decay=float(config["weight_decay"]),
            )
        optim.load_state_dict(ckpt['optimizer_states'][0])
        self.logger.info("optim load success!!")

        #
        lr_schedulers = ckpt['lr_schedulers'][0]
        base_iter = int(lr_schedulers['last_epoch'])
        scheduler = WarmupScheduler(optim,
                                    int(lr_schedulers['warmup_iter']),
                                    int(lr_schedulers['max_iter']),
                                    float(lr_schedulers['max_lr']),
                                    float(lr_schedulers['min_lr']),
                                    lr_schedulers['warmup_type'],
                                    base_iter)
        self.logger.info(f"scheduler load success!!, base_iter: {base_iter}")
        return model, optim, scheduler

    def init_model(self, ckpt_path: str,
                   device: str,
                   config: dict[str, Any]):

        if (ckpt_path is not None) and (ckpt_path != ''):
            model = DIArtModel.load(ckpt_path)
            self.logger.info(f"model load {ckpt_path} success!!")
        else:
            model = DIArtModel(dropout=float(config["dropout"]),
                               eps=float(config["eps"]))
            self.logger.info("model init success!!")

        # Train on device
        model = model.to(device)
        self.logger.info(
            f"Model init with {np.sum([p.numel() for p in model.parameters()]):,d} parameters"
        )

        #
        if bool(config["freeze"]):
            self.logger.info(f"freeze model, except out linear!!!")
            model = self.freeze_model(model)

        if bool(config["optim_weight_part_decay"]):
            # optimer
            optim = Optimizers(model, config)
        else:
            # optimer
            optim = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=float(config["learning_rate"]),
                weight_decay=float(config["weight_decay"]),
            )
        self.logger.info("optim init success!!")

        #
        max_iters = config["epochs"] * config["one_epoch_iters"]
        warmup_iters = int(float(config["warmup_ratio"]) * max_iters)
        self.logger.info(f"Updates max_iters of per epoch is : {max_iters:,},"
                         f" warmup_iters={warmup_iters}, "
                         )

        #
        scheduler = WarmupScheduler(optim,
                                    warmup_iters,
                                    max_iters,
                                    float(config['learning_rate']),
                                    float(config['min_lr']),
                                    config['warmup_strategy'])
        self.logger.info(f"scheduler init success!!, base_iter: 0")
        return model, optim, scheduler

    def freeze_model(self, model: DIArtModel):
        #
        for name, parameter in model.named_parameters():
            if ('linear_2' in name) or \
                    ('linear_3' in name) or \
                    ('linear_out' in name):
                parameter.requires_grad = True  #
            else:
                parameter.requires_grad = False  #
        return model
