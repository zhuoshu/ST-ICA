
import torch
import torch.nn as nn
from layers.PatchTST_backbone import PatchTST_backbone_TCN
from layers.PatchTST_layers import series_decomp

from typing import Callable, Optional
from torch import Tensor

class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = None #configs.enc_in
        context_window = configs.input_length
        target_window = configs.predict_length
        
        n_layers = configs.num_layers
        n_heads = configs.n_heads
        d_model = configs.hid_dim
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = False # configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        self.channel_cross = False # configs.channel_cross
            
        
        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone_TCN(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, channel_cross=self.channel_cross, tcn_output_dim=configs.tcn_output_dim, tcn_layer=configs.tcn_layer,tcn_hidden=configs.tcn_hidden, **kwargs)
            self.model_res = PatchTST_backbone_TCN(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, channel_cross=self.channel_cross,tcn_output_dim=configs.tcn_output_dim, tcn_layer=configs.tcn_layer,tcn_hidden=configs.tcn_hidden, **kwargs)
        else:
            self.model = PatchTST_backbone_TCN(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, channel_cross=self.channel_cross,tcn_output_dim=configs.tcn_output_dim, tcn_layer=configs.tcn_layer,tcn_hidden=configs.tcn_hidden,**kwargs)
        
        # self.to(device)
    
    
    def forward(self, input, mode='train'):           
        x, x_mark, aux_data = input
        x = x.squeeze(-1) # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        return x.unsqueeze(-1), None

    def store_grad(self):
        if self.decomposition:
            for name, layer in self.model_trend.TCN_backbone.named_modules():    
                if 'PadConv' in type(layer).__name__:
                    #print('{} - {}'.format(name, type(layer).__name__))
                    layer.store_grad()
            for name, layer in self.model_res.TCN_backbone.named_modules():    
                if 'PadConv' in type(layer).__name__:
                    #print('{} - {}'.format(name, type(layer).__name__))
                    layer.store_grad()
        else:
            for name, layer in self.model.TCN_backbone.named_modules():    
                if 'PadConv' in type(layer).__name__:
                    #print('{} - {}'.format(name, type(layer).__name__))
                    layer.store_grad()
    
    def set_finetuning_parameters(self, online='all'):
        for p in self.parameters():
            p.requires_grad = True
        if online == 'regressor':
            if self.decomposition:
                for p in self.model_trend.backbone.parameters():
                    p.requires_grad = False 
                for p in self.model_res.backbone.parameters():
                    p.requires_grad = False 
                for p in self.model_trend.TCN_backbone.parameters():
                    p.requires_grad = False 
                for p in self.model_res.TCN_backbone.parameters():
                    p.requires_grad = False 
            else:
                for p in self.model.backbone.parameters():
                    p.requires_grad = False 
                for p in self.model.TCN_backbone.parameters():
                    p.requires_grad = False 
        if online == 'regressor_tcn':
            if self.decomposition:
                for p in self.model_trend.backbone.parameters():
                    p.requires_grad = False 
                for p in self.model_res.backbone.parameters():
                    p.requires_grad = False 
            else:
                for p in self.model.backbone.parameters():
                    p.requires_grad = False 
        elif online == 'none':
            for p in self.parameters():
                p.requires_grad = False
        elif online == 'encoder':
            if self.decomposition:
                for p in self.model_trend.head.parameters():
                    p.requires_grad = False 
                for p in self.model_res.head.parameters():
                    p.requires_grad = False 
            else:
                for p in self.model.head.parameters():
                    p.requires_grad = False 
        elif online == 'tcn':
            if self.decomposition:
                for p in self.model_trend.backbone.parameters():
                    p.requires_grad = False 
                for p in self.model_res.backbone.parameters():
                    p.requires_grad = False 
                for p in self.model_trend.head.parameters():
                    p.requires_grad = False 
                for p in self.model_res.head.parameters():
                    p.requires_grad = False 
            else:
                for p in self.model.backbone.parameters():
                    p.requires_grad = False 
                for p in self.model.head.parameters():
                    p.requires_grad = False 
        