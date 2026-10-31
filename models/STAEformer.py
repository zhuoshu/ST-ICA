import torch.nn as nn
import torch
# from torchinfo import summary
from einops import rearrange, repeat

from layers.STEmbedding import DynamicAdapter
from layers.PromptNet import PromptNet

class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class Model(nn.Module):
    def __init__(
        self, args
    ):
        super().__init__()

        self.num_nodes = args.num_nodes
        self.in_steps = args.input_length
        self.out_steps = args.predict_length
        self.steps_per_day = args.slice_size_per_day
        self.input_dim = args.in_dim
        self.output_dim = args.pre_dim
        self.slice_size_per_day = args.slice_size_per_day

        self.dropout = 0.1
        self.se_emb_dropout = args.se_emb_dropout

        self.input_embedding_dim = 24
        self.tod_embedding_dim = 24
        self.dow_embedding_dim = 24
        self.spatial_embedding_dim = 0
        self.adaptive_embedding_dim = 80
        self.feed_forward_dim = 256

        self.stie_embed_dim = 80

        self.hasRawSemb = bool(args.hasRawSemb)
        self.prompt_type = args.prompt_type

        if self.hasRawSemb == False:
            self.adaptive_embedding_dim = 0
            self.spatial_embedding_dim = 0

    
        
        self.num_heads = args.n_heads
        self.num_layers = args.num_layers
        # self.num_heads = 4
        # self.num_layers = 3
        self.use_mixed_proj = True

        if self.prompt_type == 'FlashST':
            self.getNodeP = PromptNet(args)
            
            self.input_dim = args.FlashST_embed_dim * 4 
            # first_layer_embedding_size = 64
            # self.getNodeP_first_layer_embedding = nn.Sequential(nn.Linear(args.FlashST_embed_dim*4, first_layer_embedding_size),nn.ReLU())
            # self.input_dim = first_layer_embedding_size

        self.input_proj = nn.Linear(self.input_dim, self.input_embedding_dim)
        if self.tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(self.slice_size_per_day, self.tod_embedding_dim)
        if self.dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, self.dow_embedding_dim)
        
        if self.spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if self.adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(self.in_steps, self.num_nodes, self.adaptive_embedding_dim))
            )

        self.model_dim = (
            self.input_embedding_dim
            + self.tod_embedding_dim
            + self.dow_embedding_dim
            + self.spatial_embedding_dim
            + self.adaptive_embedding_dim
        )

        if self.prompt_type=='DA':
            self.prior_feat_dim = args.prior_feat_dim
            self.support_len = args.support_len
            self.snip_emb_dim = 80
            self.getNodeP = DynamicAdapter(args.input_length, args.in_dim, self.prior_feat_dim, self.snip_emb_dim, support_len=self.support_len, 
                                dropout_rate=self.se_emb_dropout)
            self.model_dim += self.snip_emb_dim
            
            
        if self.use_mixed_proj:
            self.output_proj = nn.Linear(
                self.in_steps * self.model_dim, self.out_steps * self.output_dim
            )
        else:
            self.temporal_proj = nn.Linear(self.in_steps, self.out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, self.feed_forward_dim, self.num_heads, self.dropout)
                for _ in range(self.num_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, self.feed_forward_dim, self.num_heads, self.dropout)
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, x, mode=None):
        x, x_time,  aux_data = x
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3 or 4)
        batch_size = x.shape[0]
        B,T,N,C = x.shape
        # print("x.shape:",x.shape)

        if self.tod_embedding_dim > 0:
            tod = x_time[...,0:1]
            tod = repeat(tod, 'b t 1 -> b t n', n = N)
            # tod = x_time[..., 0].unsqueeze(2)
            # tod = torch.repeat(tod,(1,1,self.num_nodes))
        if self.dow_embedding_dim > 0:
            dow = x_time[...,1:2]
            dow = repeat(dow, 'b t 1 -> b t n', n = N)
            # dow = x_time[..., 1].unsqueeze(2)
            # dow = torch.repeat(dow,(1,1,self.num_nodes))
        x_in = x[..., :1]

        if self.prompt_type =='FlashST':
            x_prompt = self.getNodeP(x_in, x_time, nadj=aux_data['nadj'], lpls=aux_data['lpls'], useGNN=True)
            x = self.input_proj(x_prompt)
        else:
            x = self.input_proj(x_in)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = [x]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                tod
                # (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
            # print('tod_emb.shape:', tod_emb.shape)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
            # print('dow_emb.shape:', dow_emb.shape)
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )
            features.append(spatial_emb)
            # print('spatial_emb.shape:', spatial_emb.shape)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
            # print('adp_emb.shape:', adp_emb.shape)
        
        if self.prompt_type == 'DA':
            adj_mx = aux_data['adj_mx']
            prior = aux_data['prior']
            time_series_emb_ready = x_in
            # ic(x_in.shape)
            inductive_semb = self.getNodeP(time_series_emb_ready, prior, adj_mx)
            # ic(inductive_semb.shape)
            inductive_semb = repeat(inductive_semb, 'b 1 n d-> b t n d', t=T)
            features.append(inductive_semb)
            
        elif self.prompt_type.lower() == 'none':
            inductive_semb = None

        x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)

        for attn in self.attn_layers_t:
            x = attn(x, dim=1)
        for attn in self.attn_layers_s:
            x = attn(x, dim=2)
        # (batch_size, in_steps, num_nodes, model_dim)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                batch_size, N, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, N, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)

        return out, None
    
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