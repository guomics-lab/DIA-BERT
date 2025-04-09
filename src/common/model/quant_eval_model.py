from __future__ import annotations

import logging
import os
import warnings
from typing import Any

import lightning.pytorch as ptl
import pandas as pd
import torch
from torch import Tensor

from src.common.model.quant_model import AreaModel

warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

torch.set_float32_matmul_precision = 'high'


class Evalute(ptl.LightningModule):
    """evaluate for model."""

    def __init__(
            self,
            config: dict[str, Any],
            model: AreaModel,
            model_name: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model
        self.model_name = model_name
        self._reset_metrics()

    def test_step(
            self,
            batch: [Tensor, Tensor, Tensor, Tensor, list, list],
    ) -> torch.Tensor:
        """Single test step."""
        try:
            # dataloader
            file_name, rsm, precursor_id, feat, frag_info = batch
        except:
            # iterable_dataset
            batch = next(iter(batch))
            file_name, rsm, precursor_id, feat, frag_info = batch

        rsm = rsm.to(self.device)
        frag_info = frag_info.to(self.device)
        feat = feat.to(self.device)

        # preds： (batch , score)
        # truth： (batch , 0/1)
        pred_1, pred_3, pred_6 = AreaModel.pred(self.model, rsm, frag_info, feat)
        pred_6 = pred_6.cpu().data.numpy()

        self.precursor_id_list.extend(precursor_id)
        self.file_name_list.extend(file_name)
        self.pred_6_list.extend(pred_6)

    def on_test_end(self) -> None:
        df = pd.DataFrame({"transition_group_id": self.precursor_id_list,
                           "pred_6": self.pred_6_list,
                           "file_name": self.file_name_list})
        df.to_csv(os.path.join(self.config['out_path'], f"quant_{self.model_name}.csv"), mode='w+', header=True, index=False)

    def _reset_metrics(self) -> None:
        self.precursor_id_list = []
        self.file_name_list = []
        self.pred_6_list = []
