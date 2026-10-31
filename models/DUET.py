import torch.nn as nn
from einops import rearrange
from layers.DUET_clutser import Linear_extractor_cluster 
from layers.DUET_utils import Mahalanobis_mask, Encoder, EncoderLayer, FullAttention, AttentionLayer
import torch

from layers.STEmbedding import DynamicAdapter
from layers.PromptNet import PromptNet
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.cluster = Linear_extractor_cluster(args)
        self.CI = bool(args.CI)
        # self.n_vars = args.enc_in
        self.prompt_type = args.prompt_type

        ### for SNIP calculation
        if self.prompt_type == 'FlashST':
            self.getNodeP = PromptNet(args)
            
            # self.input_dim = args.FlashST_embed_dim * 4 
            first_layer_embedding_size = 8
            self.getNodeP_first_layer_embedding = nn.Sequential(nn.Linear(args.FlashST_embed_dim*4, first_layer_embedding_size),nn.ReLU())
            self.input_dim = first_layer_embedding_size


            

        
        if self.prompt_type=='DA':
            self.se_emb_dropout = args.se_emb_dropout
            self.prior_feat_dim = args.prior_feat_dim
            self.support_len = args.support_len
            self.getNodeP =DynamicAdapter(args.input_length, args.in_dim, self.prior_feat_dim, args.hid_dim, support_len=self.support_len, 
                                dropout_rate=self.se_emb_dropout)
            

        self.mask_generator = Mahalanobis_mask(args.input_length)
        self.Channel_transformer = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,
                            None,
                            attention_dropout=args.att_dropout,
                            output_attention=False,
                        ),
                        args.hid_dim,
                        args.n_heads,
                    ),
                    args.hid_dim,
                    d_ff=None,
                    dropout=args.enc_dropout,
                    activation=args.activation_enc,
                )
                for _ in range(args.num_layers)
            ],
            norm_layer=torch.nn.LayerNorm(args.hid_dim)
        )

        self.linear_head = nn.Sequential(nn.Linear(args.hid_dim, args.predict_length), nn.Dropout(args.enc_dropout))

    def forward(self, input, mode=None):
        
        x, x_time, aux_data = input
        input = x.squeeze(-1)
        B,T,N = input.shape

        if self.prompt_type == 'DA':
            adj_mx = aux_data['adj_mx']
            prior = aux_data['prior']
            time_series_emb_ready = rearrange(input,'b t n -> b 1 n t')
            inductive_semb = self.getNodeP(time_series_emb_ready, prior, adj_mx)
            inductive_semb = rearrange(inductive_semb, 'b 1 n d -> b n d')
          
        elif self.prompt_type.lower() == 'none':
            inductive_semb = None

        # x: [batch_size, seq_len, n_vars]
        if self.CI:
            if self.prompt_type == 'FlashST':
                x_prompt = self.getNodeP(input.unsqueeze(-1), x_time, nadj=aux_data['nadj'], lpls=aux_data['lpls'], useGNN=True)
                input_data = self.getNodeP_first_layer_embedding(x_prompt)
                channel_independent_input = rearrange(input_data, 'b l n 1 -> (b n) l 1')
            else:
                channel_independent_input = rearrange(input, 'b l n -> (b n) l 1')
            reshaped_output, L_importance = self.cluster(channel_independent_input, n_vars=N)
            temporal_feature = rearrange(reshaped_output, '(b n) l 1 -> b l n', b=input.shape[0])

            
        else:
            temporal_feature, L_importance = self.cluster(input, n_vars=N)

        # B x d_model x n_vars -> B x n_vars x d_model
        temporal_feature = rearrange(temporal_feature, 'b d n -> b n d')
        # if self.n_vars > 1:
        if self.prompt_type in ['DA']:
            temporal_feature = temporal_feature + inductive_semb

        changed_input = rearrange(input, 'b l n -> b n l')
        channel_mask = self.mask_generator(changed_input)

        channel_group_feature, attention = self.Channel_transformer(x=temporal_feature, attn_mask=channel_mask)

        output = self.linear_head(channel_group_feature)

        output = rearrange(output, 'b n d -> b d n')
        output = self.cluster.revin(output, "denorm")
        
        output = output.unsqueeze(-1)

        return output, L_importance
    
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