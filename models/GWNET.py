import torch
from torch import nn
from einops import rearrange, repeat
from layers.STEmbedding import DynamicAdapter
from layers.PromptNet import PromptNet

import torch.nn.functional as F
import numpy as np


class NConv(nn.Module):
    def __init__(self):
        super(NConv, self).__init__()

    def forward(self, x, adj):
        if len(adj.shape)==2:
            x = torch.einsum('ncvl,vw->ncwl', (x, adj))
        elif len(adj.shape) == 3:
            x = torch.einsum('ncvl,nvw->ncwl', (x, adj))
        return x.contiguous()


class Linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(Linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(GCN, self).__init__()
        self.nconv = NConv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = Linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        # self.adj_mx = args.adj_mx
        self.num_nodes = args.num_nodes
        self.feature_dim = args.in_dim

        self.dropout = 0.3 #0.3
        self.blocks = 4 #4
        self.layers = 2 #2
        self.gcn_bool = True #True
        self.addaptadj = bool(args.addaptadj) #True
        self.adjtype = 'doubletransition' #'doubletransition'
        self.randomadj = True #True
        self.aptonly = bool(args.aptonly) #True
        self.kernel_size = 2 #2
        self.nhid = args.hid_dim #32
        self.residual_channels = args.hid_dim #self.nhid
        self.dilation_channels = args.hid_dim #self.nhid
        self.skip_channels = args.hid_dim * 8 # self.nhid * 8
        self.end_channels = args.hid_dim * 16 # self.nhid * 16
        self.input_window = args.input_length #12
        self.output_window = args.predict_length #12
        self.output_dim = args.pre_dim
        self.device = args.device #torch.device('cpu')
        self.embed_dim = args.hid_dim
        self.prompt_type = args.prompt_type

        self.emb_dropout = args.emb_dropout

        self.hid_dim = args.hid_dim
       
        
        
        self.prior_feat_dim = args.prior_feat_dim
        self.use_SAFA_as_adjadp = bool(args.use_SAFA_as_adjadp)
        self.supports = None # args.support 



        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        if self.prompt_type == 'FlashST':
            self.getNodeP = PromptNet(args)
            self.feature_dim = args.FlashST_embed_dim * 4 

        self.start_conv = nn.Conv2d(in_channels=self.feature_dim,
                                    out_channels=self.residual_channels,
                                    kernel_size=(1, 1))
        
        if self.randomadj:
            self.aptinit = None
        else:
            self.aptinit = self.supports[0]
        if self.aptonly: 
            self.supports = None

        receptive_field = self.output_dim

        self.supports_len = args.support_len
        if self.supports is not None:
            self.supports_len += len(self.supports)

        if self.gcn_bool and self.addaptadj:
            if self.aptinit is None:
                if self.supports is None:
                    self.supports = []
                if self.use_SAFA_as_adjadp == True:
                    # static_feats_dim = self.prior_feat_dim
                    # self.nodevec1_linear = nn.Sequential(nn.Linear(static_feats_dim, 64),nn.ReLU(),nn.Linear(64,10))
                    # self.nodevec2_linear = nn.Sequential(nn.Linear(static_feats_dim, 64),nn.ReLU(),nn.Linear(64,10))
                    self.nodevec1_linear = nn.Sequential(nn.Linear(64, 64),nn.ReLU(),nn.Linear(64,10))
                    self.nodevec2_linear = nn.Sequential(nn.Linear(64, 64),nn.ReLU(),nn.Linear(64,10))
                    # self.embed_dim = 20
                    
                else:
                    self.nodevec1 = nn.Parameter(torch.randn(self.num_nodes, 10).to(self.device),
                                                requires_grad=True).to(self.device)
                    self.nodevec2 = nn.Parameter(torch.randn(10, self.num_nodes).to(self.device),
                                                requires_grad=True).to(self.device)
                

                self.supports_len += 1
            else:
                if self.supports is None:
                    self.supports = []
                m, p, n = torch.svd(self.aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(self.device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(self.device)
                self.supports_len += 1

        for b in range(self.blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            for i in range(self.layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=self.residual_channels,
                                                   out_channels=self.dilation_channels,
                                                   kernel_size=(1, self.kernel_size), dilation=new_dilation))
                # print(self.filter_convs[-1])
                self.gate_convs.append(nn.Conv2d(in_channels=self.residual_channels,
                                                 out_channels=self.dilation_channels,
                                                 kernel_size=(1, self.kernel_size), dilation=new_dilation))
                # print(self.gate_convs[-1])
                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=self.dilation_channels,
                                                     out_channels=self.residual_channels,
                                                     kernel_size=(1, 1)))
                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=self.dilation_channels,
                                                 out_channels=self.skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(self.residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(GCN(self.dilation_channels, self.residual_channels,
                                          self.dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                    out_channels=self.end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                    out_channels=self.output_window,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.receptive_field = receptive_field
        
        self.use_SAFA_as_adjadp = bool(args.use_SAFA_as_adjadp)
        self.addSAFAemb = bool(args.addSAFAemb_GWN)
        
        if self.prompt_type=='DA':
            self.prior_feat_dim = args.prior_feat_dim
            self.se_emb_dropout = args.se_emb_dropout
            self.support_len = args.support_len
            self.getNodeP = DynamicAdapter(self.input_window, args.in_dim, self.prior_feat_dim, self.embed_dim, support_len=self.support_len, 
                                dropout_rate=self.se_emb_dropout)
          

    def forward(self, x, mode=None):
        
        history_data, x_mark, aux_data = x
        # ic(x_mark.shape, x_mark.dtype)
        supports = aux_data['adj_mx']
        if self.aptonly:
            supports = []

        inputs = history_data  # (batch_size, input_window, num_nodes, feature_dim)
        
        if self.prompt_type =='FlashST':
            x_prompt = self.getNodeP(history_data, x_mark, nadj=aux_data['nadj'], lpls=aux_data['lpls'], useGNN=True)
            inputs = x_prompt
        
        inputs = inputs.transpose(1, 3)  # (batch_size, feature_dim, num_nodes, input_window)
        inputs = nn.functional.pad(inputs, (1, 0, 0, 0))  # (batch_size, feature_dim, num_nodes, input_window+1)

        in_len = inputs.size(3)
        if in_len < self.receptive_field:
            x_in = nn.functional.pad(inputs, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x_in = inputs


        x = self.start_conv(x_in)  # (batch_size, residual_channels, num_nodes, self.receptive_field)
        
        if self.prompt_type in ['DA']:
            adj_mx = aux_data['adj_mx']
            prior = aux_data['prior']
            

            time_series_emb_ready = rearrange(x, 'b d n t -> b t n d')
            
            if self.prompt_type == 'DA':
                inductive_semb = self.getNodeP(history_data, prior, adj_mx)
            
            inductive_semb = rearrange(inductive_semb, 'b t n d -> b d n t')
            
            if self.addSAFAemb:
                x = x + inductive_semb

            if self.addaptadj and self.use_SAFA_as_adjadp:
                inductive_semb = rearrange(inductive_semb, 'b d n 1 -> b n d')
                self.nodevec1 = self.nodevec1_linear(inductive_semb)
                self.nodevec2 = self.nodevec2_linear(inductive_semb)
                self.nodevec2 = rearrange(self.nodevec2, 'b n d -> b d n')

                # self.nodevec1 = self.nodevec1_linear(prior)
                # self.nodevec2 = self.nodevec2_linear(prior).t()
                # self.nodevec2 = rearrange(self.nodevec2, 'b n d -> b d n')

            
        skip = 0


        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and supports is not None:
            if len(self.nodevec1.shape) == 2:
                adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            elif len(self.nodevec1.shape) ==3:
                adp = torch.einsum('bnd,bdk->bnk', self.nodevec1, self.nodevec2)
                adp = F.softmax(F.relu(adp), dim=-1)
            new_supports = supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*
            # (dilation, init_dilation) = self.dilations[i]
            # residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # (batch_size, residual_channels, num_nodes, self.receptive_field)
            # dilated convolution
            filter = self.filter_convs[i](residual)
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            gate = torch.sigmoid(gate)
            x = filter * gate
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            # parametrized skip connection
            s = x
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            s = self.skip_convs[i](s)
            # (batch_size, skip_channels, num_nodes, receptive_field-kernel_size+1)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except(Exception):
                skip = 0
            skip = s + skip
            # (batch_size, skip_channels, num_nodes, receptive_field-kernel_size+1)
            if self.gcn_bool and supports is not None:
                # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, supports)
                # (batch_size, residual_channels, num_nodes, receptive_field-kernel_size+1)
            else:
                # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
                x = self.residual_convs[i](x)
                # (batch_size, residual_channels, num_nodes, receptive_field-kernel_size+1)
            # residual: (batch_size, residual_channels, num_nodes, self.receptive_field)
            x = x + residual[:, :, :, -x.size(3):]
            # (batch_size, residual_channels, num_nodes, receptive_field-kernel_size+1)
            x = self.bn[i](x)
        x = F.relu(skip)
        # (batch_size, skip_channels, num_nodes, self.output_dim)
        x = F.relu(self.end_conv_1(x))
        # (batch_size, end_channels, num_nodes, self.output_dim)
        x = self.end_conv_2(x)
        # (batch_size, output_window, num_nodes, self.output_dim)
        prediction = x

        return prediction, None

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
        elif online == 'DA_dynamic_prior_adj':
            for name, p in self.named_parameters():
                if 'getNodeP.local_dynamic_adapter' in name or 'getNodeP.prior_projection' in name or 'nodevec1_linear' in name or 'nodevec2_linear' in name:
                    p.requires_grad = True
        elif online == 'DA_dynamic_prior_gate':
            for name, p in self.named_parameters():
                if 'getNodeP.local_dynamic_adapter' in name or 'getNodeP.prior_projection' in name or 'getNodeP.static_gate' in name:
                    p.requires_grad = True
        elif online == 'DA_all':
            for name, p in self.named_parameters():
                if 'getNodeP' in name:
                    p.requires_grad = True