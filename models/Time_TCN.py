from layers.ncca import TSEncoderTime
import torch.nn as nn
from einops import rearrange

from layers.STEmbedding import DynamicAdapter
from layers.PromptNet import PromptNet

class Model(nn.Module):
    def __init__(self, args, revin=False):
        super().__init__()
        # self.device = device
        self.prompt_type = args.prompt_type

        if self.prompt_type=='DA':
            self.se_emb_dropout = args.se_emb_dropout
            self.prior_feat_dim = args.prior_feat_dim
            self.support_len = args.support_len
            self.getNodeP =DynamicAdapter(args.input_length, args.in_dim, self.prior_feat_dim, args.hid_dim, support_len=self.support_len, 
                                dropout_rate=self.se_emb_dropout)
            
        if self.prompt_type == 'FlashST':
            self.getNodeP = PromptNet(args)
            
            # self.input_dim = args.FlashST_embed_dim * 4 
            first_layer_embedding_size = 8
            self.getNodeP_first_layer_embedding = nn.Sequential(nn.Linear(args.FlashST_embed_dim*4, first_layer_embedding_size),nn.ReLU())
            self.input_dim = first_layer_embedding_size
            self.encoder = TSEncoderTime(input_dims=self.input_dim * args.input_length,
                                output_dims=320,  # standard ts2vec backbone value
                                hidden_dims=64, # standard ts2vec backbone value
                                depth=10
                                )
        else:
            self.encoder = TSEncoderTime(input_dims=args.input_length,
                                output_dims=320,  # standard ts2vec backbone value
                                hidden_dims=64, # standard ts2vec backbone value
                                depth=10
                                )

        self.dim = args.predict_length
        # from layers.RevIN import RevIN
        self.revin = False #revin
        # if revin: self.revin_layer = RevIN(args.enc_in, affine=False, subtract_last=True, last_n=5)
        
        #self.regressor = nn.Sequential(nn.Linear(320, 320), nn.ReLU(), nn.Linear(320, self.dim)).to(self.device)
        self.regressor = nn.Linear(320, self.dim)#.to(self.device)
        
    def forward(self, input, mode='train'):# z: [bs x seq_len x nvars]
        z, x_mark, aux_data = input
        z = z.squeeze(-1)
        if self.revin: 
            z = self.revin_layer(z, 'norm')
        
        if self.prompt_type == 'DA':
            adj_mx = aux_data['adj_mx']
            prior = aux_data['prior']
            time_series_emb_ready = rearrange(z,'b t n -> b 1 n t')
            inductive_semb = self.getNodeP(time_series_emb_ready, prior, adj_mx)
            inductive_semb = rearrange(inductive_semb, 'b 1 n d -> b n d')
            
        elif self.prompt_type.lower() == 'none':
            inductive_semb = None

        if self.prompt_type =='FlashST':
            x_prompt = self.getNodeP(z.unsqueeze(-1), x_mark, nadj=aux_data['nadj'], lpls=aux_data['lpls'], useGNN=True)
            z = self.getNodeP_first_layer_embedding(x_prompt)
            z = rearrange(z, 'b t n d -> b (t d) n')
            inductive_semb = None
        z = self.encoder(z, inductive_semb = inductive_semb)
        z = self.regressor(z)
        
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        return z.transpose(1, 2).unsqueeze(-1), None
    
    def set_finetuning_parameters(self, online='all'):
        for p in self.parameters():
            p.requires_grad = False
        if online == 'regressor':
            for p in self.regressor.parameters():
                p.requires_grad = True 
        if online == 'encoder':
            for p in self.encoder.parameters():
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