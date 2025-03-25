
import os
from datetime import datetime
from typing import Any

import lightning.pytorch as ptl
import pandas as pd
import torch
from torch import Tensor
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from src.common.model.score_model import DIArtModel
from src.finetune.utils import eval_predict, get_prophet_result


class Evalute(ptl.LightningModule):
    """evaluate for model."""

    def __init__(
            self,
            config: dict[str, Any],
            model: DIArtModel,
            model_name: str,
            sw: SummaryWriter,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model
        self.model_name = model_name
        self.sw = sw
        self._reset_metrics()


        self.loss_fn = nn.BCELoss()

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

    def test_step(
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

        self.irt_list.extend(feat[:, 3].cpu().data.numpy() * 600)
        self.rt_list.extend(feat[:, 5].cpu().data.numpy() * 6400)

    def on_test_end(self) -> None:
        loss = self.loss_fn(torch.tensor(self.pred_list),
                            torch.tensor(self.label_list))
        auc, acc = eval_predict(self.pred_list, self.label_list)

        epoch = self.trainer.current_epoch

        self.sw.add_scalar("eval/loss", loss.item(), epoch)
        self.sw.add_scalar("eval/auc", auc, epoch)
        self.sw.add_scalar("eval/acc", acc, epoch)

        # prophet
        if bool(self.config['prophet']):
            df = pd.DataFrame({"transition_group_id": self.precursor_id_list,
                               "score": self.pred_list,
                               "label": self.label_list,
                               "file_name": self.file_name_list,
                               "iRT": self.irt_list,
                               "RT": self.rt_list})

            assert df['file_name'].nunique() == 1
            # ['transition_group_id', 'score', 'label', 'file_name']
            df.to_csv(os.path.join(self.config['out_path'], f"result_{self.file_name_list[0]}_{self.config['task_name']}.csv"), index=False)

            fdr_df, _, fdr01 = get_prophet_result(df)

            file_name = self.file_name_list[0]
            self.sw.add_scalar(f"prophet_1%fdr/{file_name}", fdr01, epoch)
            # ['transition_group_id', 'score', 'label', 'decoy', 'q_value', 'file_name']
            fdr_df.columns = ['transition_group_id', 'score', 'label', 'decoy', 'q_value', 'file_name', 'iRT', 'RT']
            fdr_df.to_csv(os.path.join(self.config['out_path'], f"fdr_{self.file_name_list[0]}_{self.config['task_name']}.csv"), index=False)

            # save eval metrics
            result = pd.DataFrame([file_name, self.model_name, fdr01, loss.item(), auc, acc, datetime.now().strftime('%Y-%m-%d %H:%M:%S')]).T
            result.columns = ['file_name', 'model_name', 'fdr1%', 'loss', 'auc', 'acc', 'dt']
            # metrics.columns=['file_name', 'model_name', 'fdr1%', 'loss', 'auc', 'acc', 'dt']
            # result_path = '/'.join(self.config['out_path'].split('/')[:-1])
            result.to_csv(os.path.join(self.config['out_path'], f"result_{self.config['task_name']}.csv"), index=False)

        self._reset_metrics()

    def _reset_metrics(self) -> None:
        self.precursor_id_list = []
        self.file_name_list = []
        self.pred_list = []
        self.label_list = []
        self.irt_list = []
        self.rt_list = []
