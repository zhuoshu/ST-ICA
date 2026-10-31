import torch
from torch import nn
from einops import rearrange
from layers.STEmbedding import DynamicAdapter
from layers.PromptNet import PromptNet


class Convs1D(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, mlp_width, mlp_depth, mlp_dropout, act=nn.ReLU()):
        super(Convs1D, self).__init__()

        if mlp_depth == 1:
            mlp_width = n_outputs

        self.input = nn.Conv1d(in_channels=n_inputs,
                               out_channels=mlp_width,
                               kernel_size=1)
        self.dropout = nn.Dropout(mlp_dropout)

        self.mlp_depth = mlp_depth

        if mlp_depth > 1:
            self.output = nn.Conv1d(in_channels=mlp_width,
                                    out_channels=n_outputs,
                                    kernel_size=1)
        if mlp_depth > 2:
            self.hiddens = nn.ModuleList([
                nn.Conv1d(in_channels=mlp_width,
                          out_channels=mlp_width,
                          kernel_size=1)
                for _ in range(mlp_depth-2)])

        self.n_outputs = n_outputs
        self.act = act

    def forward(self, x):
        x = self.input(x)
        if self.training:
            x = self.dropout(x)

        if self.mlp_depth > 1:
            x = self.act(x)

        if self.mlp_depth > 2:
            for hidden in self.hiddens:
                x = hidden(x)
                if self.training:
                    x = self.dropout(x)
                x = self.act(x)

        if self.mlp_depth > 1:
            x = self.output(x)
        return x

class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        hidden = hidden + input_data                           # residual
        return hidden




class Model(nn.Module):
    """
    Paper: Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting
    Link: https://arxiv.org/abs/2208.05233
    Official Code: https://github.com/zezhishao/STID
    Venue: CIKM 2022
    Task: Spatial-Temporal Forecasting
    """

    def __init__(self, args):
        super().__init__()
        # attributes
        self.num_nodes = args.num_nodes
        self.node_dim = args.hid_dim
        self.input_len = args.input_length
        self.input_dim = 1
        self.output_dim = 1
        self.embed_dim = args.hid_dim
        self.output_len = args.predict_length
        self.num_layer = args.num_layers
        self.temp_dim_tid = args.hid_dim
        self.temp_dim_diw = args.hid_dim
        self.time_of_day_size = args.slice_size_per_day
        self.day_of_week_size = 7
        
        self.if_time_in_day = True
        self.if_day_in_week = True
        self.if_spatial = bool(args.hasRawSemb)

        self.emb_dropout =args.emb_dropout
        self.prompt_type = args.prompt_type
        self.te_emb_dropout = args.te_emb_dropout
        self.se_emb_dropout = args.se_emb_dropout
        self.hid_dim = args.hid_dim
        
        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        

        if self.prompt_type == 'FlashST':
            self.getNodeP = PromptNet(args)
            
            # self.input_dim = args.FlashST_embed_dim * 4 
            first_layer_embedding_size = 8
            self.getNodeP_first_layer_embedding = nn.Sequential(nn.Linear(args.FlashST_embed_dim*4, first_layer_embedding_size),nn.ReLU())
            self.input_dim = first_layer_embedding_size



        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

        # encoding
        self.hidden_dim = self.embed_dim+self.node_dim * \
            int(self.if_spatial)+self.temp_dim_tid*int(self.if_time_in_day) + \
            self.temp_dim_diw*int(self.if_day_in_week) 
        
        # ST-ICA
        if self.prompt_type=='DA':
            self.prior_feat_dim = args.prior_feat_dim
            self.support_len = args.support_len
            self.getNodeP =DynamicAdapter(self.input_len, self.input_dim, self.prior_feat_dim, self.hid_dim, support_len=self.support_len, 
                                dropout_rate=self.se_emb_dropout, mlp_layers= args.adapter_mlp_layers,
                                identity_func_type=args.static_func_type, dynamic_func_type=args.dynamic_func_type)
            
            self.hidden_dim += self.hid_dim
        
        
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])




        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_len*self.output_dim, kernel_size=(1, 1), bias=True)


    def forward(self, input, supports=None, mode='train'):
        """Feed forward of STID.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """

        history_data, x_mark, aux_data = input
        # ic(x_mark.shape, x_mark.dtype)
        # ic(history_data.shape)
        # ic(x_mark.shape)
        # ic(prior.shape)
        supports = aux_data['adj_mx']
        prior = aux_data['prior']
        B,T,N,C = history_data.shape
        # if mode == 'train':


        # hidden, inductive_semb = self.get_rep([history_data, x_mark, prior, supports])
        input_data = history_data
        B,T,N,d = history_data.shape
        if self.if_time_in_day:
            # t_i_d_data = history_data[..., 1]
            t_i_d_data = x_mark[:,:,0:1]
            time_in_day_emb = self.time_in_day_emb[t_i_d_data[:, -1, :].type(torch.LongTensor)]
            time_in_day_emb = time_in_day_emb.repeat(repeats=[1,N,1])
            # ic(t_i_d_data.shape)
            # ic('repeats',time_in_day_emb.shape)
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            # d_i_w_data = history_data[..., 2]
            d_i_w_data = x_mark[..., 1:2]
            day_in_week_emb = self.day_in_week_emb[d_i_w_data[:, -1, :].type(torch.LongTensor)]
            day_in_week_emb = day_in_week_emb.repeat(repeats=[1,N,1])
            # ic(d_i_w_data.shape)
            # ic(day_in_week_emb.shape)
        else:
            day_in_week_emb = None

        # ic(input_data.shape)
        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        if self.prompt_type =='FlashST':
            x_prompt = self.getNodeP(history_data, x_mark, nadj=aux_data['nadj'], lpls=aux_data['lpls'], useGNN=True)
            input_data = self.getNodeP_first_layer_embedding(x_prompt)
            inductive_semb = None
        input_data = rearrange(input_data, 'b t n c ->b (t c) n 1')

        time_series_emb = self.time_series_emb_layer(input_data)
        # ic(time_series_emb.shape)

        node_emb = []
        semb = None
        if self.if_spatial:
            # expand node embeddings
            semb = self.node_emb
            node_emb.append(semb.unsqueeze(0).expand(
                batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))
        

        if self.prompt_type in ['DA']:
            tx = None
            tx = torch.concat([time_in_day_emb[:,0:1], day_in_week_emb[:,0:1]], dim = -1) # B, 1, 2d
            x_ready = rearrange(input_data, 'b d n 1 -> b 1 n d')
            inductive_semb= self.getNodeP(x_ready, prior, supports, tx=tx)
            inductive_semb = rearrange(inductive_semb, 'b 1 n d -> b d n 1')
            node_emb.append(inductive_semb)
            semb = inductive_semb 
            

        elif self.prompt_type.lower() == 'none':
            inductive_semb = None


        # temporal embeddings
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        # concate all embeddings
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)

        # encoding
        hidden = self.encoder(hidden)
        hidden = rearrange(hidden, 'b c n l -> b l n c')
        if inductive_semb is not None:
            inductive_semb = rearrange(inductive_semb, 'b d n l -> b l n d')
        hidden = rearrange(hidden, 'b l n c -> b c n l')

        # regression
        prediction = self.regression_layer(hidden)
        if self.output_dim>1:
            prediction = rearrange(prediction, 'b (t c) n 1 -> b t n c', t=self.output_len, c=self.output_dim)

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
        elif online == 'DA_dynamic_prior_gate':
            for name, p in self.named_parameters():
                if 'getNodeP.local_dynamic_adapter' in name or 'getNodeP.prior_projection' in name or 'getNodeP.static_gate' in name:
                    p.requires_grad = True
                    
        elif online == 'DA_all':
            for name, p in self.named_parameters():
                if 'getNodeP' in name:
                    p.requires_grad = True