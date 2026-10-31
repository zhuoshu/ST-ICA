import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from layers.STEmbedding import DynamicAdapter
from layers.PromptNet import PromptNet
from einops import rearrange

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

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

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
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
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns



class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) 
        # x: [Batch Variate d_model]
        return self.dropout(x)


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, args):
        super(Model, self).__init__()
        self.seq_len = args.input_length
        self.pred_len = args.predict_length
        self.output_attention = False
        self.use_norm = True

        self.prompt_type = args.prompt_type
        self.hid_dim = args.hid_dim
        
        if self.prompt_type == 'FlashST':
            self.getNodeP = PromptNet(args)
            
            input_len = args.FlashST_embed_dim * 4 * args.input_length
            # first_layer_embedding_size = 64
            # self.getNodeP_first_layer_embedding = nn.Sequential(nn.Linear(args.FlashST_embed_dim*4, first_layer_embedding_size),nn.ReLU())
            # self.input_dim = first_layer_embedding_size
        else:
            input_len = args.input_length
        
        if self.prompt_type=='DA':
            self.se_emb_dropout = args.se_emb_dropout
            self.prior_feat_dim = args.prior_feat_dim
            self.support_len = args.support_len
            self.getNodeP = DynamicAdapter(args.input_length, args.in_dim, self.prior_feat_dim, args.hid_dim, support_len=self.support_len, 
                                dropout_rate=self.se_emb_dropout)
            
        
        # Embedding

        self.enc_embedding = DataEmbedding_inverted(input_len, args.hid_dim, args.emb_dropout)
        # self.class_strategy = args.class_strategy

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=args.att_dropout,
                                      output_attention=self.output_attention), args.hid_dim, args.n_heads),
                    args.hid_dim,
                    d_ff=2048,
                    dropout=args.enc_dropout,
                    activation='gelu'
                ) for l in range(args.num_layers)
            ],
            norm_layer=torch.nn.LayerNorm(args.hid_dim)
        )
        self.projector = nn.Linear(args.hid_dim, args.predict_length, bias=True)

    def forecast(self, x_enc, x_mark_enc, aux_data=None):

        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates
        if self.prompt_type =='FlashST':
            x_prompt = self.getNodeP(x_enc.unsqueeze(-1), x_mark_enc, nadj=aux_data['nadj'], lpls=aux_data['lpls'], useGNN=True)
            x_enc = rearrange(x_prompt, 'b t n d-> b (t d) n')

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, None) # covariates (e.g timestamp) can be also embedded as tokens
        if self.prompt_type == 'DA':
            adj_mx = aux_data['adj_mx']
            prior = aux_data['prior']
            prompt_emb = self.getNodeP(x_enc.unsqueeze(-1), prior, adj_mx)
            prompt_emb = prompt_emb.squeeze(1)
            enc_out = enc_out + prompt_emb
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


    def forward(self, x, mode=None):
        x_enc, x_mark_enc, aux_data = x
        x_enc = x_enc.squeeze(-1) # b t n 1 -> b t n
        dec_out = self.forecast(x_enc, x_mark_enc, aux_data)
        output = dec_out[:, -self.pred_len:, :].unsqueeze(-1)  # [B, L, D] -> B L D 1
        return output, None
    
    def set_finetuning_parameters(self, online='all'):
        for p in self.parameters():
            p.requires_grad = False
        if online == 'all':
            for p in self.parameters():
                p.requires_grad = True
        elif online == 'getNodeP':
            for name, p in self.named_parameters():
                if 'getNodeP' in name:
                    p.requires_grad = True
        elif online == 'DA_dynamic_prior':
            for name, p in self.named_parameters():
                if 'getNodeP.local_dynamic_adapter' in name or 'getNodeP.prior_projection' in name:
                    p.requires_grad = True
        elif online == 'DA_dynamic_prior_gate':
            for name, p in self.named_parameters():
                if 'getNodeP.local_dynamic_adapter' in name or 'getNodeP.prior_projection' in name or 'getNodeP.static_gate' in name:
                    p.requires_grad = True
        elif online == 'DA_all':
            for name, p in self.named_parameters():
                if 'getNodeP' in name:
                    p.requires_grad = True