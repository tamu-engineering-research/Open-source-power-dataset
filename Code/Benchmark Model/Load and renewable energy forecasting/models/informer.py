# Created by xunannancy at 2021/9/25
"""
refer to codes: https://github.com/zhouhaoyi/Informer2020.git
"""
import  warnings
warnings.filterwarnings('ignore')
import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
import pandas as pd
import itertools
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from utils import merge_parameters, Pytorch_DNN_exp, Pytorch_DNN_testing, Pytorch_DNN_validation, print_network, \
    task_prediction_horizon, run_evaluate_V3
import json
import os
import yaml
from sklearn.model_selection import ParameterGrid
from collections import OrderedDict
import argparse

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_inp, d_model):
        super(TimeFeatureEmbedding, self).__init__()
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
        return self.embed(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, time_features, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbedding(d_inp=len(time_features), d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)

        return self.dropout(x)

class ProbMask():
    def __init__(self, B, H, L, index, scores):
        device = scores.device
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q)

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1./math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2,1).contiguous(), attn

class TriangularCausalMask():
    def __init__(self, B, L):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1)

    @property
    def mask(self):
        return self._mask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm2(x+y), attn

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,2)
        return x

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y)

class Informer(nn.Module):
    def __init__(self,
                 sliding_window, time_features, external_features, history_column_names, target_val_column_names, dropout, autoregressive, normalization,
                 label_len, attn='prob', d_model=512, factor=5, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 activation='gelu', distil=True, mix=True):
        super(Informer, self).__init__()
        self.history_column_names = history_column_names
        self.sliding_window, self.label_len = sliding_window, label_len
        self.time_features, self.external_features = time_features, external_features
        self.target_val_column_names = target_val_column_names
        self.autoregressive = autoregressive
        self.normalization = normalization
        # TODO: mode
        self.prediction_horizon = task_prediction_horizon['wind']

        if self.autoregressive:
            self.dec_len = max(self.prediction_horizon) # [1, 12] => 12
            self.selected_prediction_horizon = self.label_len+np.array(self.prediction_horizon)-1 # [label+0, label+11]
        else:
            self.dec_len = len(self.prediction_horizon) # [1, 12] => 2
            self.selected_prediction_horizon = self.label_len+np.arange(len(self.prediction_horizon)) #=> [label+0, label+1]
        self.attn = attn
        c_out = len(self.history_column_names)

        # Encoding
        self.enc_embedding = DataEmbedding(
            c_in=len(self.history_column_names)+len(self.external_features),
            d_model=d_model,
            time_features=time_features,
            dropout=dropout)
        self.dec_embedding = DataEmbedding(
            c_in=len(self.history_column_names)+len(self.external_features),
            d_model=d_model,
            time_features=time_features,
            dropout=dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=False),
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_mark_dec):
        """
        :param x_enc: [32, 96, 7], [batch_size, sliding_window+1, loc_index+external_features]
        :param x_mark_enc: [32,96, 4], [batch_size, sliding_window+1, time_features]
        :param x_dec: [32, 72, 7], [batch_size, prediction_length, loc_index+external_features]
        :param x_mark_dec: [32, 72, 4], [batch_size, prediction_length, time_features]
        :return: [batch_size, prediction_horizon, loc_index]
        """
        cur_device, batch_size = x_enc.device, x_enc.shape[0]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out)

        # zero impadding
        x_dec = torch.cat([x_enc[:, -self.label_len:, :], torch.zeros([x_enc.shape[0], self.dec_len, x_enc.shape[-1]]).to(cur_device)], dim=1).float()
        x_mark_dec = torch.cat([x_mark_enc[:, -self.label_len:, :], x_mark_dec], dim=1)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out)
        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        pred = dec_out[:, self.selected_prediction_horizon, :].reshape([batch_size, len(self.target_val_column_names)]) # [B, L, D]
        if self.normalization == 'minmax':
            pred = torch.sigmoid(pred)
        return pred

    def loss_function(self, batch):
        x_enc, x_mark_enc, x_mark_dec, y, flag = batch
        pred = self.forward(x_enc, x_mark_enc, x_mark_dec)

        if self.normalization == 'none':
            loss = torch.mean(nn.MSELoss(reduction='none')(pred, torch.log(y)) * flag)
            pred = torch.exp(pred)
        else:
            loss = torch.mean(nn.MSELoss(reduction='none')(pred, y) * flag)
        return loss, pred

class InformerTrainDataset(Dataset):
    def __init__(self, x_enc, x_mark_enc, x_mark_dec, y, flag):
        self.x_enc, self.x_mark_enc = x_enc, x_mark_enc
        self.x_mark_dec = x_mark_dec
        self.y = y
        self.flag = flag

    def __len__(self):
        return len(self.x_enc)

    def __getitem__(self, idx):
        return self.x_enc[idx], self.x_mark_enc[idx], self.x_mark_dec[idx], self.y[idx], self.flag[idx]

class InformerTestDataset(Dataset):
    def __init__(self, ID, x_enc, x_mark_enc, x_mark_dec):
        self.ID = ID
        self.x_enc, self.x_mark_enc = x_enc, x_mark_enc
        self.x_mark_dec = x_mark_dec

    def __len__(self):
        return len(self.ID)

    def __getitem__(self, idx):
        return self.ID[idx], self.x_enc[idx], self.x_mark_enc[idx], self.x_mark_dec[idx]

class InformerLoader():
    """
    training/validation:
    - x_enc: [batch_size, sliding_window, loc_index+external_features]
    - x_mark_enc: [batch_size, sliding_window, time_features]
    - x_dec: [batch_size, label_len+prediction_horizon, loc_index+external_features]
    - x_mark_dec: [batch_size, label_len+prediction_horizon, time_features]
    - y: [batch_size, len(prediction_horizon), loc_index]
    - flag: [batch_size, len(prediction_horizon), loc_index]
    """
    def __init__(self, file, param_dict, config):
        self.file = file
        self.config = config
        self.param_dict = param_dict

        self.batch_size = self.param_dict['batch_size']
        self.sliding_window = self.param_dict['sliding_window']

        self.time_features = self.config['exp_params']['time_features']
        self.external_features = self.config['exp_params']['external_features']
        self.autoregressive = self.param_dict['autoregressive']
        self.label_len = self.param_dict['label_len']

        # self.prediction_horizon = config['exp_params']['prediction_horizon']
        self.prediction_horizon = task_prediction_horizon['wind']
        self.train_valid_ratio = config['exp_params']['train_valid_ratio']
        self.num_workers = config['exp_params']['num_workers']

        self.variate = 'multi'
        self.external_flag = True

        self.set_dataset()

    def set_dataset(self):
        data = pd.read_csv(self.file)
        if self.autoregressive:
            # x_dec, x_mark_dec: [batch_size, label_len+max(prediction_horizon), loc_index+external_features/time_features]
            actual_prediction_horizon = range(1, max(self.prediction_horizon)+1)
        else:
            # x_dec, x_mark_dec: [batch_size, label_len+len(prediction_horizon), loc_index+external_features/time_features]
            actual_prediction_horizon = self.prediction_horizon

        """
        prepare training & validation datasets
        """
        train_flag = data['train_flag'].to_numpy()
        training_validation_index = np.sort(np.argwhere(train_flag == 1).reshape([-1]))[self.sliding_window:]

        self.history_column_names = list()
        self.target_val_column_names = list()
        for task_name, task_prediction_horizon_list in task_prediction_horizon.items():
            self.history_column_names.append(f'y{task_name[0]}_t')
            for horizon_val in task_prediction_horizon_list:
                self.target_val_column_names.append(f'y{task_name[0]}_t+{horizon_val}(val)')
        self.target_flag_column_names = [i.replace('val', 'flag') for i in self.target_val_column_names]


        y_t, external_features, time_features = data[self.history_column_names].to_numpy(), data[self.external_features].to_numpy(), data[self.time_features].to_numpy()
        # padding time_features, [#sequences, time_features]
        time_features = np.concatenate([time_features, np.zeros([max(actual_prediction_horizon), len(self.time_features)])], axis=0)

        x_enc, x_mark_enc = list(), list()
        for index in range(self.sliding_window, -1, -1):
            x_enc.append(y_t[training_validation_index-index])
            x_enc.append(external_features[training_validation_index-index])
            x_mark_enc.append(time_features[training_validation_index-index])
        # [batch_size, (sliding_window+1)*(loc_index+external_features)] => [batch_size, sliding_window+1, loc_index+external_features]
        x_enc = np.concatenate(x_enc, axis=-1).reshape([len(training_validation_index), self.sliding_window+1, len(self.history_column_names)+len(self.external_features)])
        # (sliding_window+1)*[batch_size, time_features] => [batch_size, sliding_window+1, time_features]
        x_mark_enc = np.stack(x_mark_enc, axis=1)

        x_mark_dec = list()
        for index in actual_prediction_horizon:
            x_mark_dec.append(time_features[training_validation_index+index])
        # [batch_size, prediction_horizon * (loc+external)] => [batch_size, prediction_horizon, loc+external]
        x_mark_dec = np.stack(x_mark_dec, axis=1)

        training_validation_target_val = data[self.target_val_column_names].to_numpy()[training_validation_index[self.sliding_window:]]
        training_validation_target_flag = data[self.target_flag_column_names].to_numpy()[training_validation_index[self.sliding_window:]]
        selected_index = np.argwhere(np.prod(training_validation_target_flag, axis=-1) == 1).reshape([-1])
        num_train = int(len(selected_index) * self.config['exp_params']['train_valid_ratio'])

        train_x_enc, train_x_mark_enc, train_x_mark_dec, train_y = x_enc[selected_index[:num_train]], x_mark_enc[selected_index[:num_train]], x_mark_dec[selected_index[:num_train]], training_validation_target_val[selected_index[:num_train]]
        valid_x_enc, valid_x_mark_enc, valid_x_mark_dec, valid_y = x_enc[selected_index[num_train:]], x_mark_enc[selected_index[num_train:]], x_mark_dec[selected_index[num_train:]], training_validation_target_val[selected_index[num_train:]]
        if 'normalization' in self.param_dict and self.param_dict['normalization'] != 'none':
            if self.param_dict['normalization'] == 'minmax':
                self.scalar_x_enc, self.scalar_x_mark_enc, self.scalar_x_mark_dec, self.scalar_y = MinMaxScaler(), MinMaxScaler(), MinMaxScaler(), MinMaxScaler()
            elif self.param_dict['normalization'] == 'standard':
                self.scalar_x_enc, self.scalar_x_mark_enc, self.scalar_x_mark_dec, self.scalar_y = StandardScaler(), StandardScaler(), StandardScaler(), StandardScaler()
            x_enc_last_dim, x_mark_enc_last_dim = x_enc.shape[-1], x_mark_enc.shape[-1]
            self.scalar_x_enc, self.scalar_x_mark_enc = self.scalar_x_enc.fit(x_enc.reshape([-1, x_enc_last_dim])), self.scalar_x_mark_enc.fit(x_mark_enc.reshape([-1, x_mark_enc_last_dim]))
            x_mark_dec_last_dim = x_mark_dec.shape[-1]
            self.scalar_x_mark_dec = self.scalar_x_mark_dec.fit(x_mark_dec.reshape([-1, x_mark_dec_last_dim]))
            training_validation_target_val_last_dim = training_validation_target_val.shape[-1]
            self.scalar_y = self.scalar_y.fit(training_validation_target_val.reshape([-1, training_validation_target_val_last_dim]))

            train_x_enc_shape, valid_x_enc_shape = train_x_enc.shape, valid_x_enc.shape
            train_x_enc, valid_x_enc = self.scalar_x_enc.transform(train_x_enc.reshape([-1, train_x_enc_shape[2]])).reshape(train_x_enc_shape), self.scalar_x_enc.transform(valid_x_enc.reshape([-1, valid_x_enc_shape[2]])).reshape(valid_x_enc_shape)

            train_x_mark_enc_shape, valid_x_mark_enc_shape = train_x_mark_enc.shape, valid_x_mark_enc.shape
            train_x_mark_enc, valid_x_mark_enc = self.scalar_x_mark_enc.transform(train_x_mark_enc.reshape([-1, train_x_mark_enc_shape[2]])).reshape(train_x_mark_enc_shape), self.scalar_x_mark_enc.transform(valid_x_mark_enc.reshape([-1, valid_x_mark_enc_shape[2]])).reshape(valid_x_mark_enc_shape)

            train_x_mark_dec_shape, valid_x_mark_dec_shape = train_x_mark_dec.shape, valid_x_mark_dec.shape
            train_x_mark_dec, valid_x_mark_dec = self.scalar_x_mark_dec.transform(train_x_mark_dec.reshape([-1, train_x_mark_dec_shape[2]])).reshape(train_x_mark_dec_shape), self.scalar_x_mark_dec.transform(valid_x_mark_dec.reshape([-1, valid_x_mark_dec_shape[2]])).reshape(valid_x_mark_dec_shape)

            train_y, valid_y = self.scalar_y.transform(train_y), self.scalar_y.transform(valid_y)

        self.train_dataset = InformerTrainDataset(
            torch.from_numpy(train_x_enc).to(torch.float),
            torch.from_numpy(train_x_mark_enc).to(torch.float),
            torch.from_numpy(train_x_mark_dec).to(torch.float),
            torch.from_numpy(train_y).to(torch.float),
            torch.from_numpy(training_validation_target_flag[selected_index[:num_train]]).to(torch.float))
        self.valid_dataset = InformerTrainDataset(
            torch.from_numpy(valid_x_enc).to(torch.float),
            torch.from_numpy(valid_x_mark_enc).to(torch.float),
            torch.from_numpy(valid_x_mark_dec).to(torch.float),
            torch.from_numpy(valid_y).to(torch.float),
            torch.from_numpy(training_validation_target_flag[selected_index[num_train:]]).to(torch.float))

        """
        prepare testing datasets        
        """
        testing_index = np.sort(np.argwhere(train_flag == 0).reshape([-1]))
        testing_data = data.iloc[testing_index]
        testing_ID = testing_data['ID'].to_numpy()
        x_enc, x_mark_enc = list(), list()
        for index in range(self.sliding_window, -1, -1):
            x_enc.append(y_t[testing_index-index])
            x_enc.append(external_features[testing_index-index])
            x_mark_enc.append(time_features[testing_index-index])
        x_enc = np.concatenate(x_enc, axis=-1).reshape([len(testing_index), self.sliding_window+1, len(self.history_column_names)+len(self.external_features)])
        x_mark_enc = np.stack(x_mark_enc, axis=1)

        x_mark_dec = list()
        for index in actual_prediction_horizon:
            x_mark_dec.append(time_features[testing_index+index])
        x_mark_dec = np.stack(x_mark_dec, axis=1)

        if 'normalization' in self.param_dict and self.param_dict['normalization'] != 'none':
            x_enc_shape = x_enc.shape
            x_enc = self.scalar_x_enc.transform(x_enc.reshape([-1, x_enc_shape[2]])).reshape(x_enc_shape)
            x_mark_enc_shape = x_mark_enc.shape
            x_mark_enc = self.scalar_x_mark_enc.transform(x_mark_enc.reshape([-1, x_mark_enc_shape[2]])).reshape(x_mark_enc_shape)
            x_mark_dec_shape = x_mark_dec.shape
            x_mark_dec = self.scalar_x_mark_dec.transform(x_mark_dec.reshape([-1, x_mark_dec_shape[2]])).reshape(x_mark_dec_shape)

        self.test_dataset = InformerTestDataset(
            torch.from_numpy(testing_ID).to(torch.int),
            torch.from_numpy(x_enc).to(torch.float),
            torch.from_numpy(x_mark_enc).to(torch.float),
            torch.from_numpy(x_mark_dec).to(torch.float)
        )

        return

    def load_train(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True, drop_last=True)

    def load_valid(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True, drop_last=False)

    def load_test(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True, drop_last=False)


class informer_exp(Pytorch_DNN_exp):
    def __init__(self, file, param_dict, config):
        super().__init__(file, param_dict, config)

        self.dataloader = InformerLoader(
            file,
            param_dict,
            config
        )
        self.model = self.load_model()
        print_network(self.model)

    def load_model(self):
        model = Informer(
            sliding_window=self.param_dict['sliding_window'],
            time_features=self.config['exp_params']['time_features'],
            external_features=self.config['exp_params']['external_features'],
            history_column_names=self.dataloader.history_column_names,
            target_val_column_names=self.dataloader.target_val_column_names,
            dropout=self.param_dict['dropout'],
            autoregressive=self.param_dict['autoregressive'],
            label_len=self.param_dict['label_len'],
            attn=self.param_dict['attn'],
            d_model=self.param_dict['d_model'],
            factor=self.param_dict['factor'],
            n_heads=self.param_dict['n_heads'],
            e_layers=self.param_dict['e_layers'],
            d_layers=self.param_dict['d_layers'],
            d_ff=self.param_dict['d_ff'],
            activation=self.param_dict['activation'],
            distil=self.param_dict['distil'],
            mix=self.param_dict['mix'],
            normalization=self.param_dict['normalization']
        )
        return model

    def oracle_loss(self, batch):
        loss, pred = self.model.loss_function(batch)
        if self.eval():
            # validation
            _, _, _, y, flag = batch
            pred, y = pred.detach().cpu().numpy(), y.detach().cpu().numpy()
            if self.param_dict['normalization'] != 'none':
                pred = self.dataloader.scalar_y.inverse_transform(pred)
                y = self.dataloader.scalar_y.inverse_transform(y)
            # selected_index = torch.where(flag.reshape([-1]) == 1)[0].detach().cpu().numpy()
            pred_list, y_list = list(), list()
            for column_index in range(len(self.dataloader.target_val_column_names)):
                selected_index = torch.where(flag[:, column_index] == 1)[0].detach().cpu().numpy()
                pred_list.append(pred[selected_index, column_index])
                y_list.append(y[selected_index, column_index])
            return {'loss': loss, 'pred': pred_list, 'y': y_list}

            # return {'loss': loss, 'pred': pred.reshape([-1])[selected_index], 'y': y.reshape([-1])[selected_index]}
        else:
            # training
            return {'loss': loss}

    def test_step(self, batch, batch_idx):
        ID, x_enc, x_mark_enc, x_dec = batch
        prediction = self.model.forward(x_enc, x_mark_enc, x_dec)
        if self.param_dict['normalization'] == 'none':
            prediction = torch.exp(prediction)
        return ID, prediction


def grid_search_informer(config, num_files):
    # set random seed
    torch.manual_seed(config['logging_params']['manual_seed'])
    torch.cuda.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])

    saved_folder = os.path.join(config['logging_params']['save_dir'], config['logging_params']['name'])
    flag = True
    while flag:
        if config['exp_params']['test_flag']:
            last_version = config['exp_params']['last_version'] - 1
        else:
            if not os.path.exists(saved_folder):
                os.makedirs(saved_folder)
                last_version = -1
            else:
                last_version = sorted([int(i.split('_')[1]) for i in os.listdir(saved_folder) if i.startswith('version_')])[-1]
        log_dir = os.path.join(saved_folder, f'version_{last_version+1}')
        if config['exp_params']['test_flag']:
            assert os.path.exists(log_dir)
            flag = False
        else:
            try:
                os.makedirs(log_dir)
                flag = False
            except:
                flag = True
    print(f'log_dir: {log_dir}')


    data_folder = config['exp_params']['data_folder']
    file_list = sorted([i for i in os.listdir(data_folder) if 'zone' in i and i.endswith('.csv')])[:num_files]

    param_grid = {
        'sliding_window': config['exp_params']['sliding_window'],
        'batch_size': config['exp_params']['batch_size'],
        'learning_rate': config['exp_params']['learning_rate'],
        'dropout': config['model_params']['dropout'],
        'normalization': config['exp_params']['normalization'],

        'autoregressive': config['model_params']['autoregressive'],
        'label_len': config['model_params']['label_len'],
        'attn': config['model_params']['attn'],
        'd_model': config['model_params']['d_model'],
        'factor': config['model_params']['factor'],
        'n_heads': config['model_params']['n_heads'],
        'e_layers': config['model_params']['e_layers'],
        'd_layers': config['model_params']['d_layers'],
        'd_ff': config['model_params']['d_ff'],
        'activation': config['model_params']['activation'],
        'distil': config['model_params']['distil'],
        'mix': config['model_params']['mix'],
    }
    param_dict_list = list(ParameterGrid(param_grid))

    """
    getting validation results
    """
    for file in file_list:
        cur_log_dir = os.path.join(log_dir, file.split('.')[0])
        if not config['exp_params']['test_flag']:
            if not os.path.exists(cur_log_dir):
                os.makedirs(cur_log_dir)
            Pytorch_DNN_validation(os.path.join(data_folder, file), param_dict_list, cur_log_dir, config, informer_exp)
            """
            hyperparameters selection
            """
            summary = OrderedDict()
            for param_index, param_dict in enumerate(param_dict_list):
                param_dict = OrderedDict(param_dict)
                setting_name = 'param'
                for key, val in param_dict.items():
                    setting_name += f'_{key[0].capitalize()}{val}'

                model_list = [i for i in os.listdir(os.path.join(cur_log_dir, setting_name, 'version_0')) if i.endswith('.ckpt')]
                assert len(model_list) == 1
                perf = float(model_list[0][model_list[0].find('avg_val_metric=')+len('avg_val_metric='):model_list[0].find('.ckpt')])
                with open(os.path.join(cur_log_dir, setting_name, 'version_0', 'std.txt'), 'r') as f:
                    std_text = f.readlines()
                    std_list = [[int(i.split()[0]), list(map(float, i.split()[1].split('_')))] for i in std_text]
                    std_dict = dict(zip(list(zip(*std_list))[0], list(zip(*std_list))[1]))
                best_epoch = int(model_list[0][model_list[0].find('best-epoch=')+len('best-epoch='):model_list[0].find('-avg_val_metric')])
                std = std_dict[best_epoch]
                # perf = float(model_list[0][model_list[0].find('avg_val_metric=')+len('avg_val_metric='):model_list[0].find('-std')])
                # std = float(model_list[0][model_list[0].find('-std=')+len('-std='):model_list[0].find('.ckpt')])
                summary['_'.join(map(str, list(param_dict.values())))] = [perf, std]
            with open(os.path.join(cur_log_dir, 'val_summary.json'), 'w') as f:
                json.dump(summary, f, indent=4)
            selected_index = np.argmin(np.array(list(summary.values()))[:, 0])
            selected_params = list(summary.keys())[selected_index]
            param_dict = {
                'activation': selected_params.split('_')[0],
                'attn': selected_params.split('_')[1],
                'autoregressive': eval(selected_params.split('_')[2]),
                'batch_size': int(selected_params.split('_')[3]),
                'd_ff': int(selected_params.split('_')[4]),
                'd_layers': int(selected_params.split('_')[5]),
                'd_model': int(selected_params.split('_')[6]),
                'distil': eval(selected_params.split('_')[7]),
                'dropout': float(selected_params.split('_')[8]),
                'e_layers': int(selected_params.split('_')[9]),
                'factor': int(selected_params.split('_')[10]),
                'label_len': int(selected_params.split('_')[11]),
                'learning_rate': float(selected_params.split('_')[12]),
                'mix': eval(selected_params.split('_')[13]),
                'n_heads': int(selected_params.split('_')[14]),
                'normalization': selected_params.split('_')[15],
                'sliding_window': int(selected_params.split('_')[16]),
                'std': np.array(list(summary.values()))[selected_index][-1],
            }
            # save param
            with open(os.path.join(cur_log_dir, 'param.json'), 'w') as f:
                json.dump(param_dict, f, indent=4)

        """
        prediction on testing
        """
        with open(os.path.join(cur_log_dir, 'param.json'), 'r') as f:
            param_dict = json.load(f)
        Pytorch_DNN_testing(os.path.join(data_folder, file), param_dict, cur_log_dir, config, informer_exp)

    if not os.path.exists(os.path.join(log_dir, 'config.yaml')):
        with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)

    # run evaluate
    evaluate_config = {
        'exp_params': {
            'prediction_path': log_dir,
            'prediction_interval': config['exp_params']['prediction_interval'],
        }
    }
    run_evaluate_V3(config=evaluate_config, verbose=False)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--manual_seed', '-manual_seed', type=int, help='random seed')
    parser.add_argument('--num_files', '-num_files', type=int, default=3, help='number of files to predict')

    parser.add_argument('--sliding_window', '-sliding_window', type=str, help='list of sliding_window for arima')
    parser.add_argument('--selection_metric', '-selection_metric', type=str, help='metrics to select hyperparameters, one of [RMSE, MAE, MAPE]',)
    parser.add_argument('--train_valid_ratio', '-train_valid_ratio', type=float, help='select hyperparameters on validation set')
    parser.add_argument('--time_features', '-time_features', type=str, help='list of time feature names')
    parser.add_argument('--external_features', '-external_features', type=str, help='list of external feature names')

    # model-specific features
    parser.add_argument('--batch_size', '-batch_size', type=str, help='list of batch_size')
    parser.add_argument('--max_epochs', '-max_epochs', type=int, help='number of epochs')
    parser.add_argument('--learning_rate', '-learning_rate', type=int, help='list of learning rate')
    parser.add_argument('--gpus', '-g', type=str, default='[1]')
    parser.add_argument('--dropout', '-dropout', type=str, help='list of dropout rates')

    parser.add_argument('--normalization', '-normalization', type=str, help='list of normalization options')
    parser.add_argument('--autoregressive', '-autoregressive', type=str, help='list of autoregressive options')
    parser.add_argument('--label_len', '-label_len', type=str, help='list of label_len options')
    parser.add_argument('--attn', '-attn', type=str, help='list of attn options')
    parser.add_argument('--d_model', '-d_model', type=str, help='list of d_model options')
    parser.add_argument('--factor', '-factor', type=str, help='list of factor options')
    parser.add_argument('--n_heads', '-n_heads', type=str, help='list of n_heads options')
    parser.add_argument('--e_layers', '-e_layers', type=str, help='list of e_layers options')
    parser.add_argument('--d_layers', '-d_layers', type=str, help='list of d_layers options')
    parser.add_argument('--d_ff', '-d_ff', type=str, help='list of d_ff options')
    parser.add_argument('--activation', '-activation', type=str, help='list of activation options')
    parser.add_argument('--distil', '-distil', type=str, help='list of distil options')
    parser.add_argument('--mix', '-mix', type=str, help='list of mix options')

    args = vars(parser.parse_args())
    with open('./../configs/informer.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    config = merge_parameters(args, config)
    print(f'after merge: config, {config}')

    print('gpus: ', config['trainer_params']['gpus'])
    if np.sum(config['trainer_params']['gpus']) < 0:
        config['trainer_params']['gpus'] = 0

    grid_search_informer(config, num_files=args['num_files'])