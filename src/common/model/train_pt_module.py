
from __future__ import annotations

import os
from datetime import datetime
from typing import Any
from typing import Union

import lightning.pytorch as ptl
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from torch import Tensor
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from src.common.model.score_model import DIArtModel


class PTModule(ptl.LightningModule):
    """PTL wrapper for model."""

    def __init__(
            self,
            config: dict[str, Any],
            model: DIArtModel,
            sw: SummaryWriter,
            optim: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler._LRScheduler,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model
        self.sw = sw
        self.optim = optim
        self.scheduler = scheduler

        self.train_loss_fn = nn.BCEWithLogitsLoss()
        self.val_loss_fn = nn.BCELoss() # 去掉logit

        self._reset_metrics()

        self.running_loss = None
        self.steps = 0
        self.train_step_scale = config["train_step_scale"]

    def forward(
            self,
            rsm: Tensor,
            frag_info: Tensor,
            feat: Tensor,
    ) -> Tensor:
        """Model forward pass."""
        return self.model(rsm, frag_info, feat)

    def pred(
            self,
            rsm: Tensor,
            frag_info: Tensor,
            feat: Tensor,
    ) -> Tensor:
        """Model forward pass."""
        with torch.no_grad():
            score = self.model(rsm, frag_info, feat)
            sigmod = nn.Sigmoid()
            score = sigmod(score)
        return score

    def training_step(  # need to update this
            self,
            batch: [Tensor, Tensor, Tensor, Tensor, list, list],
    ) -> torch.Tensor:
        """A single training step.

        Args:
            batch (tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.IntTensor, list, list]) :
                A batch of rsm, frag_info, feat, label as torch Tensors, file_name, precursor_id as list.

        Returns:
            torch.FloatTensor: training loss
        """
        try:
            # dataloader
            rsm, frag_info, feat, label, file_name, precursor_id = batch
        except:
            # iterable_dataset
            batch = next(iter(batch))
            rsm, frag_info, feat, label, file_name, precursor_id = batch

        #logger.info(f'rsm shape: {rsm.shape}')

        rsm = rsm.to(self.device)
        frag_info = frag_info.to(self.device)
        feat = feat.to(self.device)
        label = label.to(self.device)

        # pred： (batch , score)
        # truth： (batch , 0/1)
        pred = self.forward(rsm, frag_info, feat)
        loss = self.train_loss_fn(pred, label)
        self.log('train_loss', loss)

        if self.running_loss is None:
            self.running_loss = loss.item()
        else:
            self.running_loss = 0.99 * self.running_loss + (1 - 0.99) * loss.item()

        # skip first iter
        if ((self.steps + 1) % int(self.train_step_scale)) == 0:
            lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]

            self.sw.add_scalar("train/train_loss_raw", loss.item(), self.steps - 1)
            self.sw.add_scalar("train/train_loss_smooth", self.running_loss, self.steps - 1)
            self.sw.add_scalar("optim/lr", lr, self.steps - 1)
            self.sw.add_scalar("optim/epoch", self.trainer.current_epoch, self.steps - 1)
        self.steps += 1
        return loss

    def on_train_epoch_end(self) -> None:
        """Log the training loss at the end of each epoch."""
        epoch = self.trainer.current_epoch
        self.sw.add_scalar(f"eval/train_loss", self.running_loss, epoch)
        self.running_loss = None

    def validation_step(
            self,
            batch: [Tensor, Tensor, Tensor, Tensor, list, list],
    ) -> torch.Tensor:
        """Single test step."""
        try:
            # dataloader
            rsm, frag_info, feat, label, file_name, precursor_id = batch
        except:
            # iterable_dataset
            batch = next(iter(batch))
            rsm, frag_info, feat, label, file_name, precursor_id = batch

        rsm = rsm.to(self.device).to(torch.float16)
        frag_info = frag_info.to(self.device).to(torch.float16)
        feat = feat.to(self.device).to(torch.float16)
        label = label.to(self.device)

        # preds： (batch , score)
        # truth： (batch , 0/1)
        pred = self.pred(rsm, frag_info, feat).to(torch.float32)
        pred = pred.cpu().data.numpy()
        label = label.cpu().data.numpy()

        self.precursor_id_list.extend(precursor_id)
        self.file_name_list.extend(file_name)
        self.pred_list.extend(pred)
        self.label_list.extend(label)

    def on_validation_epoch_end(self) -> None:
        loss = self.val_loss_fn(torch.tensor(self.pred_list),
                                torch.tensor(self.label_list))
        auc, acc = eval_predict(self.pred_list, self.label_list)

        epoch = self.trainer.current_epoch

        self.sw.add_scalar("eval/val_loss", loss.item(), epoch)
        self.sw.add_scalar("eval/val_auc", auc, epoch)
        self.sw.add_scalar("eval/val_acc", acc, epoch)

        # save eval metrics
        file_name = self.file_name_list[0]
        model_name = 'epoch={}-step={}.ckpt'.format(epoch, self.trainer.global_step)
        metrics = pd.DataFrame([file_name, epoch, loss.item(), auc, acc, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), model_name]).T
        # metrics.columns = ['file_name', 'epoch', 'loss', 'auc', 'acc', 'dt', 'model_name']
        # metrics_path = '/'.join(.split('/')[:-2])
        metrics.to_csv(os.path.join(self.config['metrics_out_path'], "metrics_finetune.csv"), mode='a+', header=False, index=False)
        self._reset_metrics()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Save config with checkpoint."""
        checkpoint["config"] = self.config
        checkpoint["epoch"] = self.trainer.current_epoch

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Attempt to load config with checkpoint."""
        self.config = checkpoint["config"]
        self.optim = checkpoint["optim"]

    def configure_optimizers(
            self,
    ) -> [torch.optim.Optimizer, dict[str, Any]]:
        """Initialize the optimizer.

        This is used by pytorch-lightning when preparing the model for training.

        Returns
        -------
        Tuple[torch.optim.Optimizer, Dict[str, Any]]
            The initialized Adam optimizer and its learning rate scheduler.
        """
        return [self.optim], {"scheduler": self.scheduler, "interval": "step"}

    def _reset_metrics(self) -> None:
        self.precursor_id_list = []
        self.file_name_list = []
        self.pred_list = []
        self.label_list = []



def Optimizers(model, config) -> torch.optim.Optimizer:
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (nn.Linear, )
    blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}

    # merge other
    other = param_dict.keys() - (decay | no_decay)
    decay = decay | other

    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                       % (str(param_dict.keys() - union_params), )

    # 仅优化requires_grad=True的网络层
    # need_optim = set([name for name, parameter in model.named_parameters() if parameter.requires_grad])
    # decay = decay & need_optim
    # no_decay = no_decay & need_optim

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": float(config["weight_decay"])},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optim = torch.optim.Adam(optim_groups,
                             lr=float(config["learning_rate"]))
    return optim

class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup scheduler."""

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_iter: int,
                 max_iter: int,
                 max_lr: float,
                 min_lr: float,
                 warmup_type: str,
                 base_iter: int = 0,
                 ) -> None:
        """
        Args:


        """
        self.warmup_iter = warmup_iter
        self.max_iter = max_iter
        self.warmup_type = warmup_type

        self.base_iter = base_iter

        self.max_lr = max_lr
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        """Get the learning rate at the current step."""
        if self.warmup_type == 'exp':
            lr_factor = self.get_exponential_lr_factor(epoch=self.last_epoch + self.base_iter)
        elif self.warmup_type == 'cos':
            lr_factor = self.get_cosine_lr_factor(epoch=self.last_epoch + self.base_iter)
        else:
            lr_factor = 1.0

        if isinstance(lr_factor, float):
            lr_factor = min(1.0, lr_factor)
        else:
            # when lr_factor is complex, designate lr_factor equal 0 where lr equal min_lr
            lr_factor = 0.0

        return [float(base_lr * lr_factor) if float(base_lr * lr_factor) > self.min_lr else self.min_lr
                for base_lr in self.base_lrs]

    def get_exponential_lr_factor(self, epoch: int) -> float:
        """Get the LR factor at the current step."""
        lr_factor = 1.0
        if epoch <= self.warmup_iter:
            lr_factor *= epoch / self.warmup_iter
        elif epoch <= self.max_iter:
            lr_factor = (1 - (epoch - self.warmup_iter) / (self.max_iter - self.warmup_iter)) ** 0.9
        else:
            lr_factor = 0.0
        return lr_factor

    def get_cosine_lr_factor(self, epoch: int) -> float:
        """Get the LR factor at the current step."""
        lr_factor = 1.0
        if epoch <= self.warmup_iter:
            lr_factor *= epoch / self.warmup_iter
        elif epoch <= self.max_iter:
            lr = self.min_lr + \
                 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(epoch / (self.max_iter - self.warmup_iter) * np.pi))
            lr_factor = lr / self.max_lr
        else:
            lr_factor = 0.0
        return lr_factor


def eval_predict(predictions, targets):
    fpr, tpr, threshold = metrics.roc_curve(targets, predictions, pos_label=1)
    auc_results = metrics.auc(fpr, tpr)
    round_pred = np.round(predictions)
    correct_count = 0
    for i in range(len(round_pred)):
        if round_pred[i] == targets[i]:
            correct_count += 1
    correctness = float(correct_count) / len(round_pred)
    return auc_results, correctness

