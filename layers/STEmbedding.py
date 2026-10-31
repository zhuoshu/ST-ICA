import torch
from torch import nn
from einops import rearrange,repeat

class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout=0.1, support_len=1, order=2):
        super(GCN, self).__init__()
        c_in = (order * support_len + 1) * c_in
        self.mlp = nn.Linear(c_in, c_out)
        self.order = order
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, support):
        out = [x]
        if type(support)!=list:
            support = [support]
        for a in support:
            x1 = torch.einsum('btkd,nk->btnd', x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = torch.einsum('btkd,nk->btnd', x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=-1)
        h = self.mlp(h)
        h = self.dropout(h)
        return h



class Core_Adaptive(nn.Module):
    def __init__(self, d_in, d_core, d_out, core_num, head=4, nndropout=0.3, dropout=0.08):
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


    def forward(self, input):
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




class featMLP(nn.Module):
    def __init__(self, num_feat_dim, hid_dim, dropout_rate=0.1):
        super().__init__()
        self.meta_projection = nn.Linear(num_feat_dim, hid_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(hid_dim,hid_dim)
    
    def forward(self, feat):
        h = self.meta_projection(feat)
        emb = self.fc(self.drop(self.act((h))))
        return emb




class getSTembedding_adp(nn.Module):
    def __init__(self, num_feat_dim, hid_dim, static_func_type='featMLP', dropout_rate=0.1):
        super().__init__()
        if static_func_type == 'featMLP':
            self.embedder = featMLP(num_feat_dim, hid_dim, dropout_rate)

    def forward(self, aux_data):
        emb = self.embedder(aux_data)
        return emb

class getSTembedding_dynamic(nn.Module):
    def __init__(self, hid_dim, feat_type, support_len):
        super().__init__()
        assert feat_type in ['gcn', 'gcn2', 'sgcn', 'sgcn2', 'core_adapt', 'att_t2n', 'gate_t2n']
        self.feat_type = feat_type
        self.in_dim = hid_dim

        if feat_type == 'gcn':
            self.gcn_layer = GCN(hid_dim, hid_dim, support_len=support_len, order=1)
        elif feat_type == 'gcn2':
            self.gcn_layer = GCN(hid_dim, hid_dim, support_len=support_len, order=2)
        elif feat_type == 'core_adapt': # derive from [ICML 2025] STOP
            self.att_layer = Core_Adaptive(hid_dim, hid_dim, hid_dim, 8)
        elif feat_type == 'att_t2n':
            self.att_layer = TimeNodeCrossAttnBlock(hid_dim, 4, 1)
        elif feat_type == 'gate_t2n':
            self.gate_layer = TimeGatedNodeBlock(hid_dim, 1)

    def forward(self, x, adj_mx, tx=None):
        if self.feat_type in ['gcn', 'gcn2', 'sgcn']:
            h = self.gcn_layer(x, adj_mx)
        elif self.feat_type =='sgcn2':
            h = self.act(self.gcn_layer(x, adj_mx))
            h = self.gcn_layer2(h, adj_mx)
        elif self.feat_type == 'core_adapt': # derive from [ICML 2025] STOP
            h = self.att_layer(x)
        return h

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hid_dim, out_dim)
    
    def forward(self, feat):
        h = self.fc1(feat)
        h = self.fc2(self.drop(self.act((h))))
        return h

class DynamicAdapter(nn.Module):
    def __init__(self, input_length, in_dim, prior_feat_dim, emb_dim, support_len,  
                   mlp_layers = 1, dynamic_func_type='gcn2', identity_func_type = 'mlp',
                   dropout_rate=0.1):
        super(DynamicAdapter, self).__init__()
        self.emb_dim = emb_dim      
        self.identity_func_type = identity_func_type
        self.dynamic_func_type = dynamic_func_type  
        
        if self.identity_func_type != 'none':
            self.prior_projection = nn.Sequential(MLP(prior_feat_dim, emb_dim, emb_dim, dropout_rate), *[MLP(emb_dim, emb_dim, emb_dim, dropout_rate) for _ in range(1, mlp_layers-1)])
            
        if dynamic_func_type != 'none':
            self.window_projection = nn.Sequential(MLP(input_length*in_dim, emb_dim, emb_dim), *[MLP(emb_dim, emb_dim, emb_dim) for _ in range(1, mlp_layers-1)])
            # if dynamic_input_dim is not None and dynamic_input_dim!= emb_dim:
            #     self.input_emb_linear = nn.Linear(dynamic_input_dim, emb_dim, bias=False)
            self.local_dynamic_adapter = getSTembedding_dynamic(emb_dim, dynamic_func_type, support_len)
        
        

    def forward(self, x, prior, adj_list, tx=None):
        B,T,N,d = x.shape
        identity_emb = None
        if self.identity_func_type != 'none':
            identity_emb = self.prior_projection(prior)

            if len(identity_emb.shape) == 2:
                identity_emb = rearrange(identity_emb, 'n d -> 1 1 n d')
                identity_emb = repeat(identity_emb, '1 t n d -> b t n d', b= B)
            elif len(identity_emb.shape) == 3:
                identity_emb = rearrange(identity_emb, 'b n d -> b 1 n d')
        
        if self.dynamic_func_type != 'none':
            series_emb = self.window_projection(rearrange(x, 'b t n c -> b 1 n (t c)'))
            # if d != identity_emb.shape[-1]:
            #     series_emb = self.input_emb_linear(x)
            residual_emb = self.local_dynamic_adapter(series_emb, adj_list, tx)
            adapt_emb = identity_emb + residual_emb if identity_emb is not None else residual_emb
        else:
            assert identity_emb is not None
            adapt_emb = identity_emb

        return adapt_emb

class AttentionPrompt(nn.Module):
    def __init__(self, num_prompt, embed_dim, prompt_threshold=0.2, dynamic_input_dim=None) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.dynamic_input_dim = dynamic_input_dim
        if self.dynamic_input_dim is not None and dynamic_input_dim != embed_dim:
            self.input_emb_linear =nn.Linear(dynamic_input_dim, embed_dim, bias=False)
        self.prompt = nn.Embedding(num_prompt, embed_dim)
        nn.init.kaiming_uniform_(self.prompt.weight, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        self.prompt_threshold = prompt_threshold

    def forward(self, patches):
        """
        :param patches: [B, P, N, D]
        :return: prompted_patches
        """
        # spatial dimension
        if self.dynamic_input_dim is not None and self.dynamic_input_dim != self.embed_dim:
            patches = self.input_emb_linear(patches)
        score_map = torch.sigmoid(torch.matmul(patches, self.prompt.weight.t()))  # [B, P, N, num_prompt]
        score_map = score_map * (score_map > self.prompt_threshold).float()
        prompt_emb = torch.sum(score_map.unsqueeze(-1) * self.prompt.weight, dim=-2)
        return prompt_emb
    