import torch
import torch.nn as nn
import copy
from abc import abstractmethod
from einops import repeat
from layers.STEmbedding import DynamicAdapter
from layers.PromptNet import PromptNet

class BaseModel(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len=12, horizon=12):
        super(BaseModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon


    @abstractmethod
    def forward(self): # (x, label=None, adj=None)
        raise NotImplementedError


    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class MLP(nn.Module):
    def __init__(self, num_layer, model_dim, prompt_dim, tod_size, kernel_size, args):
        super(MLP, self).__init__()
        self.input_len = args.input_length
        self.output_len = args.predict_length
        self.input_dim = args.in_dim
        self.output_dim = args.pre_dim

        self.embed_dim = model_dim # 64
        self.num_layer = num_layer
        self.node_dim = prompt_dim
        self.temp_dim_tod = prompt_dim
        self.temp_dim_dow = prompt_dim
        self.temp_dim_moy = prompt_dim

        self.kernel_size = kernel_size
        
        self.time_of_day_size = tod_size
        self.day_of_week_size = 7
        self.month_of_year_size = 12
        self.if_time_in_day = 1
        self.if_day_in_week = 1
        self.if_spatial = 1
        self.if_month_in_year = 0
        
        self.prompt_type = args.prompt_type
        if self.prompt_type == 'FlashST':
            self.getNodeP = PromptNet(args)
            
            # self.input_len = args.FlashST_embed_dim * 4 
            first_layer_embedding_size = 1
            self.getNodeP_first_layer_embedding = nn.Sequential(nn.Linear(args.FlashST_embed_dim*4, first_layer_embedding_size),nn.ReLU())
            # self.input_dim = first_layer_embedding_size


        if self.if_time_in_day:
            print('day')
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tod))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            print('week')
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_dow))
            nn.init.xavier_uniform_(self.day_in_week_emb)
        if self.if_month_in_year:
            print('month')
            self.month_in_year_emb = nn.Parameter(
                torch.empty(self.month_of_year_size, self.temp_dim_moy))
            nn.init.xavier_uniform_(self.month_in_year_emb)
        
        # decomposing
        self.decompsition = series_decomp(self.kernel_size)
        
        self.time_series_emb_layer1 = nn.Conv1d(
            in_channels=self.input_len, out_channels=self.embed_dim, kernel_size=1, bias=True)
        self.time_series_emb_layer2 = nn.Conv1d(
            in_channels=self.input_len, out_channels=self.embed_dim, kernel_size=1, bias=True)
        
        self.hidden_dim = self.embed_dim + \
            self.temp_dim_tod*int(self.if_day_in_week) + \
            self.temp_dim_dow*int(self.if_time_in_day) + \
            self.temp_dim_moy*int(self.if_month_in_year)
        
        self.encoder = nn.Sequential(
            *[FeedForward(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])
        
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)
    
    def forward(self, history_data, label=None, x_mark=None, aux_data=None,  adj=None):
        if self.if_time_in_day:
            t_i_d_data = history_data[..., 1]
            time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :]).type(torch.LongTensor)]
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            d_i_w_data = history_data[..., 2]
            day_in_week_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :]).type(torch.LongTensor)]
        else:
            day_in_week_emb = None
        if self.if_month_in_year:
            m_i_y_data = history_data[..., 3]
            month_in_year_emb = self.month_in_year_emb[(m_i_y_data[:, -1, :] * self.month_of_year_size - 1).type(torch.LongTensor)]
        else:
            month_in_year_emb = None

        if self.prompt_type =='FlashST':
            x_prompt = self.getNodeP(history_data, x_mark, nadj=aux_data['nadj'], lpls=aux_data['lpls'], useGNN=True)
            input_data = self.getNodeP_first_layer_embedding(x_prompt) # b t n 1
            seasonal_init, trend_init = self.decompsition(input_data[..., 0])
        else:
            seasonal_init, trend_init = self.decompsition(history_data[..., 0])


        seasonal_output = self.time_series_emb_layer1(seasonal_init)
        trend_output = self.time_series_emb_layer2(trend_init)
        time_series_emb = (seasonal_output + trend_output).unsqueeze(-1)

        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))
        if month_in_year_emb is not None:
            tem_emb.append(month_in_year_emb.transpose(1, 2).unsqueeze(-1))

        hidden = torch.cat([time_series_emb] + tem_emb, dim=1)
        h = hidden.transpose(1, -1)

        hidden = self.encoder(hidden)
        z = hidden.transpose(1, -1)
        prediction = self.regression_layer(hidden)

        return  h, z, prediction
    
    def module(self, hidden, label=None, adj=None):
        hidden = self.encoder(hidden.transpose(1, -1))
        return hidden.transpose(1, -1)  
    

class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=input_dim,  
                      out_channels=4*hidden_dim, 
                      kernel_size=(1, 1), 
                      bias=True),
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.Conv2d(in_channels=4*hidden_dim, 
                      out_channels=hidden_dim, 
                      kernel_size=(1, 1), 
                      bias=True)
        )

    def forward(self, input_data):
        hidden = self.fc(input_data)
        hidden = hidden + input_data
        return hidden


class Model(nn.Module):
    def __init__(self, args, stmodel):
    # def __init__(self, model_args, stmodel, dim, core, ssie_dim, head, **args):
        super(Model, self).__init__()
        ## base spatio-temporal model
        self.stmodel = stmodel
        ssie_dim = None # not used
        head = 8
        hid_dim = 256
        core = 8
        # input_dim = args.in_dim
        # output_dim = args.pre_dim
        # self.inverse = stmodel.if_inverse()

        ## training type parameters
        self.extra_type = True #model_args['extra_type']
        self.same = False #model_args['same']
        if self.extra_type and not self.same:
            self.stmodel_detach = copy.deepcopy(stmodel)
        
        self.prompt_type = args.prompt_type


        ## decouple parameters and networks
        self.in_dim = 128 # equals to the dim of hidden emb before entering decoder/predictor/out-put layer.
        self.out_dim = 128 # equals to the dim of shallow emb. 
        self.backcast_hidden_dim = hid_dim # the dim in the backcast network. 
        if core:
            # self.backcast = Core_Adaptive(self.in_dim, self.in_dim, self.out_dim, self.node_num, core)
            self.backcast = Core_Adaptive(self.in_dim, self.in_dim, self.out_dim, ssie_dim, core, head)
        else:
            self.backcast = nn.Sequential(
                nn.Linear(self.in_dim, 4*self.in_dim), 
                nn.GELU(),
                nn.Linear(4*self.in_dim, self.out_dim), 
            )
        
        if self.prompt_type=='DA':
            self.prior_feat_dim = args.prior_feat_dim
            self.support_len = args.support_len
            self.getNodeP = DynamicAdapter(args.input_length, args.in_dim, self.prior_feat_dim, self.in_dim, support_len=self.support_len, 
                                dropout_rate=args.se_emb_dropout)
            

        ## decoder parameters and nerworks
        self.decoder_hidden_dim = hid_dim # the dim in the decoder network. 
        self.horizon = args.predict_length # the horizon
        self.decoder = nn.Sequential(
            nn.Linear(self.in_dim, self.decoder_hidden_dim), 
            nn.GELU(),
            nn.Linear(self.decoder_hidden_dim, self.horizon),  
        )

    
    def forward(self, x, label=None, adj=None, ssie=None,mode=None): 
        x, x_time, aux_data = x
        B,T,N,c = x.shape
        prior = aux_data['prior']
        supports = aux_data['adj_mx']
        if self.extra_type:
            history_data =[]
            history_data.append(x)
            history_data.append(repeat(x_time[...,0:1].unsqueeze(2),'b t 1 d -> b t n d', n = N))
            history_data.append(repeat(x_time[...,1:2].unsqueeze(2),'b t 1 d -> b t n d', n = N))
            history_data = torch.concat(history_data, dim=-1)
            h, z, y = self.stmodel(history_data, label, x_time, aux_data)

            if self.prompt_type == 'DA':
                time_series_emb_ready = x # b 1 n d
                inductive_semb = self.getNodeP(time_series_emb_ready, prior, supports)
                z = z + inductive_semb
            # ic(h.shape, z.shape, y.shape)
            h_res = self.backcast(z, ssie)
            z_res = self.stmodel.module(h - h_res, label) if self.same else self.stmodel_detach.module(h - h_res, label)
            # z_res = self.residual_information_propagation(z_res)
            y_res = self.decoder(z_res)
            return y + y_res.transpose(1, -1), None
        else:
            return self.stmodel(x, label)[-1], None
        
        
    def set_finetuning_parameters(self, online='all'):
        for p in self.parameters():
            p.requires_grad = False
        if online == 'backcast':
            for p in self.backcast.parameters():
                p.requires_grad = True 
        elif online == 'stmodel':
            for p in self.stmodel.parameters():
                p.requires_grad = True 
        elif online == 'all':
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
    

class Core_Adaptive(nn.Module):
    def __init__(self, d_in, d_core, d_out, ssie_dim, core_num, head=4, nndropout=0.3, dropout=0.08):
        super(Core_Adaptive, self).__init__()
        if core_num == 0:
            raise 
        else:
            print('core number is', core_num)
        if head == 0:
            raise 
        else:
            print('head number is', head)
        self.head_dim = d_core // head
        self.cores = nn.Parameter(torch.randn((head, core_num, self.head_dim)))
        # self.affiliation = nn.Parameter(torch.randn(core_num, node_num))
        self.value = nn.Conv2d(d_in, d_core, kernel_size=(1, 1))
        self.ffn = nn.Sequential(
            nn.Conv2d(d_in + d_core, 4*(d_in + d_core), kernel_size=(1, 1)),
            nn.GELU(),
            nn.Dropout(nndropout),
            nn.Conv2d(4*(d_in + d_core), d_out, kernel_size=(1, 1)),
        )
        self.d_core = d_core
        self.core_num = core_num
        self.head = head
        self.nndropout = nn.Dropout(nndropout)
        self.dropout = dropout
        self.norm = nn.BatchNorm2d(d_out)


    def forward(self, input, ssie, adj=None, *args, **kwargs): 
        input = input.permute(0, 3, 1, 2) # (b, f, t, n)
        b, f, t, n = input.shape
        q = self.value(input) # (b, f, t, n)
        q = torch.stack(torch.split(q, self.head_dim, dim=1), dim=1) # (b, f, t, n) = (b, hd, t, n) -> (b, h, d, t, n)
        
        affiliation = torch.einsum('hcd, bhdtn -> bhctn', self.cores, q).transpose(-2, -3) / self.head_dim**0.5 # (b, h, c, t, n) -> (b, h, t, c, n)
        affiliation_node_to_core = torch.softmax(affiliation, dim=-1)
        affiliation_core_to_node = torch.softmax(affiliation, dim=-2)

        v = torch.stack(torch.split(input, self.head_dim, dim=1), dim=1) # (b, f, t, n) -> (h, b, d', t, n)
        v = torch.einsum('bhftn, bhtcn -> bhftc', v, affiliation_node_to_core)
        v = torch.einsum('bhftc, bhtcn -> bhftn', v, affiliation_core_to_node)

        v = v.transpose(0, 1).reshape(b, f, t, n)
        output = torch.cat([input-v, v], dim=1)
        output = self.ffn(output)
        output = output + input
        output = self.norm(output)
        return output.permute(0, 2, 3, 1)



