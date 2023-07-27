import torch
import torch.nn as nn
from util.pos_embed import get_1d_sincos_pos_embed_from_grid
import numpy as np

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)
    
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)

class swish(nn.Module):
    '''
    The swish layer: implements the swish activation function
    '''
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class LogVarLayer(nn.Module):
    '''
    The log variance layer: calculates the log variance of the data along given 'dim'
    (natural logarithm)
    '''
    def __init__(self, dim):
        super(LogVarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(x.var(dim = self.dim, keepdim= True), 1e-6, 1e6))

class Bio_FSTNet(nn.Module):
    def SCB(self, m, nChan, nBands, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
                Conv2dWithConstraint(nBands, m*nBands, (nChan, 1), groups= nBands, bias=True ,
                                     max_norm = 2, doWeightNorm = doWeightNorm,padding = 0),
                nn.BatchNorm2d(m*nBands),
                swish()
                )

    def LastBlock(self, inF, outF, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
                LinearWithConstraint(inF, outF, max_norm = 0.5, doWeightNorm = doWeightNorm, *args, **kwargs),
                nn.LogSoftmax(dim = 1))
    
    def convBlock(self, inF, outF, dropoutP, kernalSize,  *args, **kwargs):
        return nn.Sequential(
            nn.Dropout(p=dropoutP),
            Conv2dWithConstraint(inF, outF, kernalSize, bias= False, max_norm = 2, *args, **kwargs),
            nn.BatchNorm2d(outF),
            nn.ELU(),
            nn.MaxPool2d((2,2), stride = (2,2)),
            )

    def __init__(self, nChan, nTime, nClass, nBands = 9, m = 48, doWeightNorm = True, strideFactor= 4, *args, **kwargs):
        super(Bio_FSTNet, self).__init__()
        self.nBands = nBands
        self.channels = nChan
        self.m = m
        self.scb = self.SCB(m, nChan, self.nBands, doWeightNorm = doWeightNorm)
        self.strideFactor = strideFactor
        self.temporalLayer = LogVarLayer(dim = 3)
        self.multiHeadAttention = nn.ModuleList([nn.MultiheadAttention(embed_dim = 1,\
            num_heads = 1) for i in range(m*nBands)])
        self.pos_embed = get_1d_sincos_pos_embed_from_grid(int(432), int(4), cls_token=False).T
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.lastLayer = self.LastBlock(self.m*self.nBands*self.strideFactor, nClass, doWeightNorm = doWeightNorm)
        self.residual_factor = 0.1
        self.linear_residual = nn.Linear(nChan * nBands, nBands * m, bias = True)
        nFiltLaterLayer = [2, 5, 10, 20]
        dropoutP = 0.25
        kernalSize = (3, 3)
        self.stft_conv = nn.ModuleList([nn.Sequential(*[self.convBlock(inF, outF, dropoutP, kernalSize)
            for inF, outF in zip(nFiltLaterLayer, nFiltLaterLayer[1:])]) for i in range(m*nBands)])
        self.stft_linear =  nn.ModuleList([nn.Linear(120, 20)for i in range(m*nBands)])
        self.flatten = nn.Flatten()
        
    def making_suitable_pos_embedded_array(self, batch_size, spectral_dim):
            pos_embedded_used = np.expand_dims(self.pos_embed,0).repeat(batch_size, axis= 0)
            pos_embedded_used = np.expand_dims(pos_embedded_used,3).repeat(spectral_dim, axis= 3)
            return torch.from_numpy(pos_embedded_used).to(torch.float).to(self.device)
    
    def forward(self, x):
        new_channel_size = self.nBands * self.channels
        x_reshaped = x.reshape(x.size()[0], new_channel_size, x.size()[-1])
        x_reshaped = torch.einsum('nhq->nqh', x_reshaped)
        x_reshaped = self.linear_residual(x_reshaped) * self.residual_factor
        x_reshaped = torch.einsum('nqh->nhq', x_reshaped)

        x_ = self.scb(x)  
        x_ = torch.squeeze(x_,2)
        x_ = x_ + x_reshaped
        x_ = torch.unsqueeze(x_, 2)
        x_ = x_.reshape([*x_.shape[0:2], self.strideFactor, int(x_.shape[3]/self.strideFactor)])
        x_ = self.temporalLayer(x_)

        # x_results = torch.zeros((x_.size()[0], x_.size()[1], self.strideFactor, 20), requires_grad = True, dtype = torch.float).to(self.device)
        # for i in range(x_.size(1)):
        #     x_singel_chan =x_[:,i,:] 
        #     x_stft_singel_chan = torch.stft(x_singel_chan, n_fft = 128, hop_length = 8)
        #     seg_len = int(x_stft_singel_chan.size()[2] / self.strideFactor)
        #     # print(seg_len)
        #     if(x_stft_singel_chan.size()[2] % seg_len!= 0):
        #         for j in range(self.strideFactor -1):
        #             temp_stft_data = x_stft_singel_chan[:,:,j*seg_len:(j+1)*seg_len,:]
        #             temp_stft_data = torch.einsum('nhwc->nchw', temp_stft_data)
        #             x_stft_singel_out = self.stft_conv[i](temp_stft_data)
        #             x_stft_singel_out = self.flatten(x_stft_singel_out)
        #             x_stft_singel_out = self.stft_linear[i](x_stft_singel_out)
        #             x_results[:,i,j,:] = x_stft_singel_out
        #         temp_stft_data = x_stft_singel_chan[:,:,(self.strideFactor -1)*seg_len:,:]
        #         temp_stft_data = torch.einsum('nhwc->nchw', temp_stft_data)
        #         x_stft_singel_out = self.stft_conv[i](temp_stft_data)
        #         x_stft_singel_out = self.flatten(x_stft_singel_out)
        #         x_stft_singel_out = self.stft_linear[i](x_stft_singel_out)
        #         x_results[:,i,-1,:] = x_stft_singel_out
        
        # batch_size = x_results.size()[0]
        # spectral_dim = x_results.size()[-1]
        # pos_embedded_used = self.making_suitable_pos_embedded_array(batch_size, spectral_dim)
        # for i, mh_attention in enumerate(self.multiHeadAttention):
        #     x_sliced = x_results[:,i,:,:] + pos_embedded_used[:,i,:,:]
        #     x_sliced = torch.einsum('nhw->hnw',x_sliced)
        #     x_sliced, _ = mh_attention(x_sliced,x_sliced,x_sliced)
        #     x_sliced = torch.einsum('nhw->hnw',x_sliced)
        #     x_results[:,i,:,:] = x_sliced
        # x_results = torch.flatten(x_results, start_dim= 1)
        # x_results = self.lastLayer(x_results)

        batch_size = x_.size()[0]
        spectral_dim = x_.size()[-1]
        pos_embedded_used = self.making_suitable_pos_embedded_array(batch_size, spectral_dim)
        for i, mh_attention in enumerate(self.multiHeadAttention):
            x_sliced = x_[:,i,:,:] + pos_embedded_used[:,i,:,:]
            x_sliced = torch.squeeze(x_[:,i,:,:],dim=1)
            x_sliced = torch.einsum('nhw->hnw',x_sliced)
            x_sliced, _ = mh_attention(x_sliced,x_sliced,x_sliced)
            x_sliced = torch.einsum('nhw->hnw',x_sliced)
            x_[:,i,:,:] = x_sliced
        x_ = torch.flatten(x_, start_dim= 1)
        x_ = self.lastLayer(x_)
        return x_

# from torchsummary import summary
# from prettytable import PrettyTable

# device = str("cuda:"+str(0)) if torch.cuda.is_available() else "cpu"
# model = Bio_FSTNet(nChan = 22, nTime = 800, nClass = 4, nBands = 9, m = 48, doWeightNorm = True, strideFactor= 4).to(device)
# def count_parameters(model):
#     table = PrettyTable(["Modules", "Parameters"])
#     total_params = 0
#     for name, parameter in model.named_parameters():
#         if not parameter.requires_grad: continue
#         params = parameter.numel()
#         table.add_row([name, params])
#         total_params+=params
#     print(table)
#     print(f"Total Trainable Params: {total_params}")
#     return total_params
    
# count_parameters(model)
# data = nn.Parameter(torch.rand(16, 9, 22, 800)).to(device)
# model(data)