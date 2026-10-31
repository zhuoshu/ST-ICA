from layers.fsnet import TSEncoder
import torch
import torch.nn as nn


class TS2VecEncoderWrapper(nn.Module):
    def __init__(self, encoder, mask):
        super().__init__()
        self.encoder = encoder
        self.mask = mask

    def forward(self, input):
        return self.encoder(input, mask=self.mask)[:, -1]

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.device = device
        encoder = TSEncoder(input_dims=args.input_length,
                             output_dims=320,  # standard ts2vec backbone value
                             hidden_dims=64, # standard ts2vec backbone value
                             depth=10) 
        self.encoder = TS2VecEncoderWrapper(encoder, mask='all_true')
        self.dim = args.predict_length
        
        #self.regressor = nn.Sequential(nn.Linear(320, 320), nn.ReLU(), nn.Linear(320, self.dim)).to(self.device)
        self.regressor = nn.Linear(320, self.dim)
        
    def forward(self, input, mode='train'):
        x, x_mark, aux_data = input
        x = x.squeeze(-1)
        rep = self.encoder.encoder.forward_time(x, mask='all_true')
        y = self.regressor(rep)
        return y.transpose(1, 2).unsqueeze(-1), None
    
    def store_grad(self):
        for name, layer in self.encoder.named_modules():    
            if 'PadConv' in type(layer).__name__:
                #print('{} - {}'.format(name, type(layer).__name__))
                layer.store_grad()
        
    def set_finetuning_parameters(self, online='all'):
        for p in self.parameters():
            p.requires_grad = True
        if online == 'regressor':
            for p in self.encoder.parameters():
                p.requires_grad = False 
        elif online == 'none':
            for p in self.parameters():
                p.requires_grad = False