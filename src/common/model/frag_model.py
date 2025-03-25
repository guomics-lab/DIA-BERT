import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as pt_data

warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision = 'high'


class Dataset(pt_data.Dataset):
    def __init__(self, file_name):
        self.file_name = file_name
        self.file = None
        self.precursor_id = None
        self.label = None
        self.rsm = None
        self.frag_info = None
        self.feat = None
        self.pos = None

    def __getitem__(self, idx):
        return_dict = {"file": self.file[idx],
                       "precursor_id": self.precursor_id[idx],
                       "label": self.label[idx],
                       "rsm": self.rsm[idx],
                       "frag_info": self.frag_info[idx],
                       "feat": self.feat[idx],
                       "pos": self.pos[idx]}
        return return_dict

    def __len__(self):
        return len(self.precursor_id)

    def fit_scale(self, scaler_list, embedding=False):
        pass

class NLinearMemoryEfficient(nn.Module):
    def __init__(self, n: int, d_in: int, d_out: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for _ in range(n)])

    def forward(self, x):
        return torch.stack([l(x[:, i]) for i, l in enumerate(self.layers)], 1)


class NumEmbeddings(nn.Module):
    def __init__(
            self,
            n_features: int,
            d_embedding: int,
            embedding_arch: list,
            d_feature: int,
    ) -> None:
        super().__init__()
        assert embedding_arch
        assert set(embedding_arch) <= {
            'linear',
            'shared_linear',
            'relu',
            'layernorm',
            'batchnorm',
        }

        # NLinear_ =  NLinear
        layers: list[nn.Module] = []

        if embedding_arch[0] == 'linear':
            assert d_embedding is not None
            layers.append(
                NLinearMemoryEfficient(n_features, d_feature, d_embedding)
            )
        elif embedding_arch[0] == 'shared_linear':
            layers.append(
                nn.Linear(d_feature, d_embedding)
            )
        d_current = d_embedding

        for x in embedding_arch[1:]:
            layers.append(
                nn.ReLU()
                if x == 'relu'
                else NLinearMemoryEfficient(n_features, d_current, d_embedding)  # type: ignore[code]
                if x == 'linear'
                else nn.Linear(d_current, d_embedding)  # type: ignore[code]
                if x == 'shared_linear'
                else nn.LayerNorm([n_features, d_current])
                if x == 'layernorm'
                else nn.BatchNorm1d(n_features)
                if x == 'batchnorm'
                else nn.Identity()
            )
            if x in ['linear']:
                d_current = d_embedding
            assert not isinstance(layers[-1], nn.Identity)
        self.d_embedding = d_current
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x) 
        return output * self.weight


class ConvLayer(nn.Module):
    def __init__(self,
                 embedding_len,
                 n_rsm_features,
                 n_frags=72,
                 in_channels=1,
                 mid_channels=4,
                 out_channels=16,
                 kernel_size=(1, 3, 3),
                 padding=(0, 1, 1),
                 eps=1e-5):
        super(ConvLayer, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels,
                               out_channels=mid_channels,
                               kernel_size=kernel_size,
                               padding=padding)

        # layer norm
        self.norm1 = nn.LayerNorm([mid_channels, n_frags, embedding_len, n_rsm_features], eps=eps)
#         self.norm1 = RMSNorm([mid_channels, n_frags, embedding_len, n_rsm_features], eps=eps)


        self.conv2 = nn.Conv3d(in_channels=mid_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               padding=padding)
        # layer norm
        self.norm2 = nn.LayerNorm([out_channels, n_frags, embedding_len // 2, n_rsm_features // 2], eps=eps)
#         self.norm2 = RMSNorm([out_channels, n_frags, embedding_len // 2, n_rsm_features // 2], eps=eps)

        # pool
        self.pool_0 = nn.AvgPool3d((1, 2, 2))
        self.pool_1 = nn.AvgPool3d((1, 2, 4))
        self.activation = nn.ReLU()

    # @torch.compile
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)  # (batch, 1, 72, 8, 16) -> (batch, 4, 72, 8, 16)
        x = self.activation(x)
        x = self.norm1(x)  # (batch, 4, 72, 8, 16) -> (batch, 4, 72, 8, 16)
        
        x = self.pool_0(x)  # (batch, 4, 72, 8, 16) -> (batch, 4, 72, 4, 8)
        x = self.conv2(x)  # (batch, 4, 72, 4, 8) -> (batch, 16, 72, 4, 8)
        x = self.activation(x)
        x = self.norm2(x)  # (batch, 16, 72, 4, 8) -> (batch, 16, 72, 4, 8)
        x = self.pool_1(x)  # (batch, 16, 72, 4, 8) -> (batch, 16, 72, 2, 2)

        x = x.transpose(1, 2).contiguous().view(-1, 72, 64)
        # (batch, 16, 72, 2, 2) -> (batch, 72, 16, 2, 2) -> (batch, 72, 64)
        return x


class AreaModel(nn.Module):
    def __init__(self, 
                 n_num_features=8,
                 n_rsm_features=16,
                 embedding_len=8,
                 channels=72,
                 embedding_dim=8,
                 eps=1e-5):
        """
        Args:
            n_num_features:
            n_rsm_features：rsm
            embedding_len：
            channels:
            embedding_dim：
            eps: norm的eps

        """
        super(AreaModel, self).__init__()

        self.n_num_features = n_num_features
        self.n_rsm_features = n_rsm_features
        self.embedding_len = embedding_len
        self.embedding_dim = embedding_dim
        self.channels = channels
        self.eps = eps

        self.conv = ConvLayer(self.embedding_len, self.n_rsm_features, in_channels=1, mid_channels=4, out_channels=16, eps=self.eps)

        self.transformer_layer_1 = nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=64*4,
                                                              batch_first=True, layer_norm_eps=self.eps)
        self.transformer_encoder_1 = nn.TransformerEncoder(self.transformer_layer_1, num_layers=4)

        self.transformer_layer_2 = nn.TransformerEncoderLayer(d_model=72, nhead=8, dim_feedforward=72*4,
                                                              batch_first=True, layer_norm_eps=self.eps)
        self.transformer_encoder_2 = nn.TransformerEncoder(self.transformer_layer_2, num_layers=4)

        embedding_arch = ['shared_linear', 'batchnorm', 'relu']
        self.num_embeddings_1 = NumEmbeddings(n_features=channels, d_embedding=16,
                                              embedding_arch=embedding_arch,
                                              d_feature=4)
        self.num_embeddings_2 = NumEmbeddings(n_features=1, d_embedding=8,
                                              embedding_arch=embedding_arch,
                                              d_feature=1)

        self.rt_embedding = nn.Embedding(44, self.embedding_dim)
        self.instructment_embedding = nn.Embedding(7, self.embedding_dim)

        self.linear_0 = nn.Linear(104, 64)
        self.linear_1 = nn.Linear(64, 1)
        self.activation = nn.ReLU()

        def init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                torch.nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Embedding):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

        self.apply(init_weights)

    def forward(self, rsm, frag_info, feats, pos=0):
        """
        Args:

        """
        ########################################################################################################
        # conv + LayerNorm
        rsm = self.conv(rsm)  # (batch, 72, 8, 16) ==> (batch, 72, 64)

        rsm = self.transformer_encoder_1(rsm)  # (batch, 72, 64)

        frag_info = self.num_embeddings_1(frag_info)

        rsm = torch.cat((rsm, frag_info), dim=-1)

        rsm = rsm.transpose(1, 2)  # (batch, 72, 80) ==> (batch, 80, 72)(batch, rt_feat, frag)
        rsm = self.transformer_encoder_2(rsm)  # (batch, 80, 72)

        rsm_lib_pos = rsm[:, :, 6 + pos]    # (batch, 80)

        max_intensity = feats[:, 7].unsqueeze(dim=-1).unsqueeze(dim=-1) # (batch, 1) ==> # (batch, 1, 1)
        max_intensity = self.num_embeddings_2(max_intensity).squeeze(dim=-2)  # (batch, 1, 1) ==> (batch, 1, 8) ==> (batch, 8)

        rt = self.rt_embedding(feats[:, 8].long()) # (batch, 1) ==> (batch, 8) 
        instructment = self.instructment_embedding(feats[:, 9].long()) # (batch, 1) ==> (batch, 8) 

        # (batch, 8) + + (batch, 8) + + (batch, 8)  ==> (batch, 24)
        feats = torch.cat((max_intensity, rt, instructment), dim=1)
        
        ########################################################################################################
        dense_rep_lib_pos = torch.cat((rsm_lib_pos, feats), dim=1) # (batch, 80) + (batch, 24) ==> (batch, 104)

        dense_rep_lib_pos = self.activation(self.linear_0(dense_rep_lib_pos))  # (batch, 104)  ==> (batch, 64)
        pred = self.linear_1(dense_rep_lib_pos).squeeze()  # (batch, 64)  ==> (batch, 1) ==> (batch)
        pred = torch.nan_to_num(pred, nan=1e-5)
        return pred
    
    @classmethod
    def load(cls, path: str) -> nn.Module:
        """Load model from checkpoint."""
        area_model = cls()
        state_dict = torch.load(path, map_location="cpu")
        # check if PTL checkpoint
        if "state_dict" in state_dict.keys():
            state_dict = state_dict['state_dict']
        model_state = {k.replace("model.", ""): v for k, v in state_dict.items()}

        k_missing = np.sum(
            [x not in list(model_state.keys()) for x in list(area_model.state_dict().keys())]
        )
        if k_missing > 0:
            print(f"Model checkpoint is missing {k_missing} keys!")
        k_missing = np.sum(
            [x not in list(area_model.state_dict().keys()) for x in list(model_state.keys())]
        )
        if k_missing > 0:
            print(f"Model state is missing {k_missing} keys!")
            # print(f" missing keys: {list(set(model_state.keys()) - set(diart_model.state_dict().keys()))[:2]}")
        area_model.load_state_dict(model_state, strict=False)
        return area_model
    
    @classmethod
    def load_ema(cls, path: str) -> nn.Module:
        """Load model from checkpoint."""
        area_model = cls()
        state_dict = torch.load(path, map_location="cpu")
        # check if PTL checkpoint
        if "model_ema" in state_dict.keys():
            state_dict = state_dict['model_ema']
        model_state = {k.replace("module.", ""): v for k, v in state_dict.items()}

        k_missing = np.sum(
            [x not in list(model_state.keys()) for x in list(area_model.state_dict().keys())]
        )
        if k_missing > 0:
            print(f"Model checkpoint is missing {k_missing} keys!")
        k_missing = np.sum(
            [x not in list(area_model.state_dict().keys()) for x in list(model_state.keys())]
        )
        if k_missing > 0:
            print(f"Model state is missing {k_missing} keys!")
            # print(f" missing keys: {list(set(model_state.keys()) - set(diart_model.state_dict().keys()))[:2]}")
        area_model.load_state_dict(model_state, strict=False)
        return area_model

    def pred(self,
             rsm,
             frag_info,
             feats,
             pos_tensor):
        """model pred."""
        pred = torch.zeros(frag_info.shape[0]).to(rsm.device)

        for pos in pos_tensor.unique().tolist():
            indices = torch.nonzero(torch.eq(pos_tensor, pos)).squeeze()
            indices = torch.LongTensor(indices)

            rsm_part = rsm[indices]
            frag_info_part = frag_info[indices]
            feats_part = feats[indices]
            
            pred_part = self.forward(rsm_part, frag_info_part, feats_part, int(pos)).float().to(rsm.device)
            pred = pred.scatter(0, indices.to(rsm.device), pred_part)

        return pred
