import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.STEmbedding import DynamicAdapter
from layers.PromptNet import PromptNet

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, args):
        super(Model, self).__init__()
        self.seq_len = args.input_length
        self.pred_len = args.predict_length
        
        # Decompsition Kernel Size
        kernel_size = 3
        self.decompsition = series_decomp(kernel_size)
        self.individual = False
        self.channels = None # not used

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
            
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))


        self.prompt_type = args.prompt_type
        self.hid_dim = args.hid_dim

        if self.prompt_type == 'FlashST':
            self.getNodeP = PromptNet(args)
            self.input_dim = args.FlashST_embed_dim * 4  
            self.getNodeP_first_layer_embedding = nn.Sequential(nn.Linear(args.FlashST_embed_dim*4, 1),nn.ReLU())
            
        ## for SAFA
        if self.prompt_type=='DA':
            self.se_emb_dropout = args.se_emb_dropout
            self.prior_feat_dim = args.prior_feat_dim
            self.support_len = args.support_len
            self.getNodeP = DynamicAdapter(args.input_length, args.in_dim, self.prior_feat_dim, self.seq_len, support_len=self.support_len, 
                                dropout_rate=self.se_emb_dropout)
            


    def forward(self, x, mode=None):
        x, x_mark,  aux_data = x

        if self.prompt_type == 'FlashST':
            x_prompt = self.getNodeP(x, x_mark, nadj=aux_data['nadj'], lpls=aux_data['lpls'], useGNN=True)
            x = self.getNodeP_first_layer_embedding(x_prompt)

        x = x.squeeze(-1) #BTN



        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1) # B N T


        enc_out = x.permute(0,2,1) # B N T
        if self.prompt_type == 'DA':
            adj_mx = aux_data['adj_mx']
            prior = aux_data['prior']                
            prompt_emb = self.getNodeP(x.unsqueeze(-1), prior, adj_mx)
            prompt_emb = prompt_emb.squeeze(1) # B N t
            seasonal_init = seasonal_init + prompt_emb
            trend_init = trend_init + prompt_emb

    
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        predy = seasonal_output + trend_output
        predy = predy.permute(0,2,1).unsqueeze(-1)
        
        return predy, None

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