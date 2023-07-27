#!/usr/bin/env python
# coding: utf-8
"""
All network architectures: FBCNet, EEGNet, DeepConvNet
@author: Ravikiran Mane
"""
import torch
import torch.nn as nn
import sys
import math
current_module = sys.modules[__name__]

debug = False

#%% Deep convnet - Baseline 1
class deepConvNet(nn.Module):
    def convBlock(self, inF, outF, dropoutP, kernalSize,  *args, **kwargs):
        print("In: "+str(inF), " Out: "+str(outF))
        return nn.Sequential(
            nn.Dropout(p=dropoutP),
            Conv2dWithConstraint(inF, outF, kernalSize, bias= False, max_norm = 2, *args, **kwargs),
            nn.BatchNorm2d(outF),
            nn.ELU(),
            nn.MaxPool2d((1,3), stride = (1,3))
            )

    def firstBlock(self, outF, dropoutP, kernalSize, nChan, *args, **kwargs):
        return nn.Sequential(
                Conv2dWithConstraint(1,outF, kernalSize, padding = 0, max_norm = 2, *args, **kwargs),
                Conv2dWithConstraint(25, 25, (nChan, 1), padding = 0, bias= False, max_norm = 2),
                nn.BatchNorm2d(outF),
                nn.ELU(),
                nn.MaxPool2d((1,3), stride = (1,3))
                )

    def lastBlock(self, inF, outF, kernalSize, *args, **kwargs):
        return nn.Sequential(
                Conv2dWithConstraint(inF, outF, kernalSize, max_norm = 0.5,*args, **kwargs),
                nn.LogSoftmax(dim = 1))

    def calculateOutSize(self, model, nChan, nTime):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1,1,nChan, nTime)
        model.eval()
        out = model(data).shape
        return out[2:]

    def __init__(self, nChan, nTime, nClass, dropoutP = 0.25, *args, **kwargs):
        super(deepConvNet, self).__init__()
        kernalSize = (1,10)
        nFilt_FirstLayer = 25
        nFiltLaterLayer = [25, 50, 100, 200]
        firstLayer = self.firstBlock(nFilt_FirstLayer, dropoutP, kernalSize, nChan)
        middleLayers = nn.Sequential(*[self.convBlock(inF, outF, dropoutP, kernalSize)
            for inF, outF in zip(nFiltLaterLayer, nFiltLaterLayer[1:])])

        self.allButLastLayers = nn.Sequential(firstLayer, middleLayers)
        self.fSize = self.calculateOutSize(self.allButLastLayers, nChan, nTime)
        self.lastLayer = self.lastBlock(nFiltLaterLayer[-1], nClass, (1, self.fSize[1]))
        self.flatten_my = nn.Flatten()

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.allButLastLayers(x)
        x = self.lastLayer(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)
        return x

#%% EEGNet Baseline 2
class eegNet(nn.Module):
    def initialBlocks(self, dropoutP, *args, **kwargs):
        block1 = nn.Sequential(
                nn.Conv2d(1, self.F1, (1, self.C1),
                          padding = (0, self.C1 // 2 ), bias =False),
                nn.BatchNorm2d(self.F1),
                Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.nChan, 1),
                                     padding = 0, bias = False, max_norm = 1,
                                     groups=self.F1),
                nn.BatchNorm2d(self.F1 * self.D),
                nn.ELU(),
                nn.AvgPool2d((1,4), stride = 4),
                nn.Dropout(p = dropoutP))
        block2 = nn.Sequential(
                nn.Conv2d(self.F1 * self.D, self.F1 * self.D,  (1, 22),
                                     padding = (0, 22//2) , bias = False,
                                     groups=self.F1* self.D),
                nn.Conv2d(self.F1 * self.D, self.F2, (1,1),
                          stride =1, bias = False, padding = 0),
                nn.BatchNorm2d(self.F2),
                nn.ELU(),
                nn.AvgPool2d((1,8), stride = 8),
                nn.Dropout(p = dropoutP)
                )
        return nn.Sequential(block1, block2)

    def lastBlock(self, inF, outF, kernalSize, *args, **kwargs):
        return nn.Sequential(
                nn.Conv2d(inF, outF, kernalSize, *args, **kwargs),
                nn.LogSoftmax(dim = 1))

    def calculateOutSize(self, model, nChan, nTime):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1,1,nChan, nTime)
        model.eval()
        out = model(data).shape
        return out[2:]

    def __init__(self, nChan, nTime, nClass = 4,
                 dropoutP = 0.25, F1=8, D = 2,
                 C1 = 125, *args, **kwargs):
        super(eegNet, self).__init__()
        self.F2 = D*F1
        self.F1 = F1
        self.D = D
        self.nTime = nTime
        self.nClass = nClass
        self.nChan = nChan
        self.C1 = C1

        self.firstBlocks = self.initialBlocks(dropoutP)
        self.fSize = self.calculateOutSize(self.firstBlocks, nChan, nTime)
        self.lastLayer = self.lastBlock(self.F2, nClass, (1, self.fSize[1]))

    def forward(self, x):
        # print(x.size())
        x = torch.unsqueeze(x, 1)
        x = self.firstBlocks(x)
        # print(x.size())
        x = self.lastLayer(x)
        # print(x.size())
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)
        # print(x.size())
        return x

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

#%% Support classes for FBNet Implementation
class VarLayer(nn.Module):
    '''
    The variance layer: calculates the variance of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(VarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.var(dim = self.dim, keepdim= True)

class StdLayer(nn.Module):
    '''
    The standard deviation layer: calculates the std of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(StdLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.std(dim = self.dim, keepdim=True)

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

class MeanLayer(nn.Module):
    '''
    The mean layer: calculates the mean of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(MeanLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim = self.dim, keepdim=True)

class MaxLayer(nn.Module):
    '''
    The max layer: calculates the max of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(MaxLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        ma ,ima = x.max(dim = self.dim, keepdim=True)
        return ma

class swish(nn.Module):
    '''
    The swish layer: implements the swish activation function
    '''
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class FBCNet(nn.Module):
    # just a FBCSP like structure : chan conv and then variance along the time axis
    '''
        FBNet with seperate variance for every 1s. 
        The data input is in a form of batch x 1 x chan x time x filterBand
    '''
    def SCB(self, m, nChan, nBands, doWeightNorm=True, *args, **kwargs):
        '''
        The spatial convolution block
        m : number of sptatial filters.
        nBands: number of bands in the data
        '''
        return nn.Sequential(
                Conv2dWithConstraint(nBands, m*nBands, (nChan, 1), groups= nBands,
                                     max_norm = 2 , doWeightNorm = doWeightNorm,padding = 0),
                nn.BatchNorm2d(m*nBands),
                swish()
                )

    def LastBlock(self, inF, outF, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
                LinearWithConstraint(inF, outF, max_norm = 0.5, doWeightNorm = doWeightNorm, *args, **kwargs),
                nn.LogSoftmax(dim = 1))

    def __init__(self, nChan, nTime, nClass, nBands = 9, m = 32,
                 temporalLayer = 'LogVarLayer', strideFactor= 4, doWeightNorm = True, *args, **kwargs):
        super(FBCNet, self).__init__()

        self.nBands = nBands
        self.m = m
        self.strideFactor = strideFactor

        # create all the parrallel SCBc
        self.scb = self.SCB(m, nChan, self.nBands, doWeightNorm = doWeightNorm)
        
        # Formulate the temporal agreegator
        self.temporalLayer = current_module.__dict__[temporalLayer](dim = 3)

        # The final fully connected layer
        self.lastLayer = self.LastBlock(self.m*self.nBands*self.strideFactor, nClass, doWeightNorm = doWeightNorm)

    def forward(self, x):
        x = self.scb(x)
        # print(x.size())
        x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3]/self.strideFactor)])
        x = self.temporalLayer(x)
        x = torch.flatten(x, start_dim= 1)
        # features = x.clone()
        x = self.lastLayer(x)
        return x

class PowerLayer_me(nn.Module):
    '''
    The standard deviation layer: calculates the std of the data along given 'dim'
    '''

    def __init__(self, dim, length, step):
        super(PowerLayer, self).__init__()
        self.dim = dim
        self.pooling = nn.AvgPool2d(kernel_size=(1, length), stride=(1, step))

    def forward(self, x):
        return torch.log(self.pooling(x.pow(3)))
    
class FBSCNet(nn.Module):
    # just a FBCSP like structure : chan conv and then variance along the time axis
    '''
        FBNet with seperate variance for every 1s. 
        The data input is in a form of batch x 1 x chan x time x filterBand
    '''
    
    def temporal_learner(self, in_chan, out_chan, kernel, pool, pool_step_rate):
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=kernel, stride=(1, 1)),
            PowerLayer_me(dim=-1, length=pool, step=int(pool_step_rate*pool))
        )
    def SCB(self, m, nChan, nBands, doWeightNorm=True, *args, **kwargs):
        '''
        The spatial convolution block
        m : number of sptatial filters.
        nBands: number of bands in the data
        '''
        return nn.Sequential(
                Conv2dWithConstraint(nBands, m*nBands, (nChan, 1), groups= nBands,
                                     max_norm = 2 , doWeightNorm = doWeightNorm,padding = 0),
                nn.BatchNorm2d(m*nBands),
                swish()
                )

    def LastBlock(self, inF, outF, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
                LinearWithConstraint(inF, outF, max_norm = 0.5, doWeightNorm = doWeightNorm, *args, **kwargs),
                nn.LogSoftmax(dim = 1))

    def __init__(self, nChan, nTime, nClass, nBands = 9, m = 32,
                 temporalLayer = 'LogVarLayer', strideFactor= 4,sampling_rate = 250, pool=16, pool_step_rate = 0.25, doWeightNorm = True, *args, **kwargs):
        super(FBCNet, self).__init__()
        self.pool = pool
        self.nBands = nBands
        self.m = m
        self.strideFactor = strideFactor
        self.window = [0.5, 0.25, 0.125]
        # create all the parrallel SCBc
        self.scb = self.SCB(m, nChan, self.nBands, doWeightNorm = doWeightNorm)
        self.temporalLayer_me = self.temporal_learner(in_chan=m*nBands, out_chan=int(m*nBands*0.75),
                                               kernel=(1, int(self.window[0] * sampling_rate)),
                                               pool=self.pool, pool_step_rate=pool_step_rate)
        
        # Formulate the temporal agreegator
        self.temporalLayer = current_module.__dict__[temporalLayer](dim = 3)

        # The final fully connected layer
        self.lastLayer = self.LastBlock(self.m*self.nBands*self.strideFactor, nClass, doWeightNorm = doWeightNorm)

    def forward(self, x):
        x = self.scb(x)
        x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3]/self.strideFactor)])
        x = self.temporalLayer_me(x)
        print(x.shape)
        sys.exit()
        x = self.temporalLayer(x)
        x = torch.flatten(x, start_dim= 1)
        # features = x.clone()
        x = self.lastLayer(x)
        return x
# This is the networks script
import torch.nn.functional as F
from layers import GraphConvolution
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class PowerLayer(nn.Module):
    '''
    The standard deviation layer: calculates the std of the data along given 'dim'
    '''

    def __init__(self, dim, length, step):
        super(PowerLayer, self).__init__()
        self.dim = dim
        self.pooling = nn.AvgPool2d(kernel_size=(1, length), stride=(1, step))

    def forward(self, x):
        return torch.log(self.pooling(x.pow(3)))


class LGG(nn.Module):
    def temporal_learner(self, in_chan, out_chan, kernel, pool, pool_step_rate):
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=kernel, stride=(1, 1)),
            PowerLayer(dim=-1, length=pool, step=int(pool_step_rate*pool))
        )

    def __init__(self, num_classes, input_size, sampling_rate, num_T,
                 out_graph, dropout_rate, pool, pool_step_rate, idx_graph,device):
        # input_size: EEG frequency x channel x datapoint
        super(LGG, self).__init__()
        self.idx = idx_graph
        self.window = [0.5, 0.25, 0.125]
        self.pool = pool
        self.channel = input_size[1]
        self.brain_area = len(self.idx)
        self.device = device
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[0] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.Tception2 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[1] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.Tception3 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[2] * sampling_rate)),
                                               self.pool, pool_step_rate)

        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_t_ = nn.BatchNorm2d(num_T)
        self.OneXOneConv = nn.Sequential(
            nn.Conv2d(num_T, num_T, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.AvgPool2d((1, 2)))
        # diag(W) to assign a weight to each local areas
        size = self.get_size_temporal(input_size)
        self.local_filter_weight = nn.Parameter(torch.FloatTensor(self.channel, size[-1]),
                                                requires_grad=True)
        nn.init.xavier_uniform_(self.local_filter_weight)
        self.local_filter_bias = nn.Parameter(torch.zeros((1, self.channel, 1), dtype=torch.float32),
                                              requires_grad=True)

        # aggregate function
        self.aggregate = Aggregator(self.idx)

        # trainable adj weight for global network
        self.global_adj = nn.Parameter(torch.FloatTensor(self.brain_area, self.brain_area), requires_grad=True)
        nn.init.xavier_uniform_(self.global_adj)
        # to be used after local graph embedding
        self.bn = nn.BatchNorm1d(self.brain_area)
        self.bn_ = nn.BatchNorm1d(self.brain_area)
        # learn the global network of networks
        self.GCN = GraphConvolution(size[-1], out_graph)

        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(int(self.brain_area * out_graph), num_classes))

    def forward(self, x):
        # x = torch.unsqueeze(x, 1)
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        out = self.OneXOneConv(out)
        out = self.BN_t_(out)
        out = out.permute(0, 2, 1, 3)
        out = torch.reshape(out, (out.size(0), out.size(1), -1))
        out = self.local_filter_fun(out, self.local_filter_weight)
        out = self.aggregate.forward(out)
        adj = self.get_adj(out)
        out = self.bn(out)
        print(out.size(), adj.size())
        out = self.GCN(out, adj)
        print(out.size())
        out = self.bn_(out)
        out = out.view(out.size()[0], -1)
        # features = out.clone()
        out = self.fc(out)

        return out

    def get_size_temporal(self, input_size):
        # input_size: frequency x channel x data point
        data = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
        z = self.Tception1(data)
        out = z
        z = self.Tception2(data)
        out = torch.cat((out, z), dim=-1)
        z = self.Tception3(data)
        out = torch.cat((out, z), dim=-1)
        out = self.BN_t(out)
        out = self.OneXOneConv(out)
        out = self.BN_t_(out)
        out = out.permute(0, 2, 1, 3)
        out = torch.reshape(out, (out.size(0), out.size(1), -1))
        size = out.size()
        return size

    def local_filter_fun(self, x, w):
        w = w.unsqueeze(0).repeat(x.size()[0], 1, 1)
        x = F.relu(torch.mul(x, w) - self.local_filter_bias)
        return x

    def get_adj(self, x, self_loop=True):
        # x: b, node, feature
        adj = self.self_similarity(x)   # b, n, n
        num_nodes = adj.shape[-1]
        adj = F.relu(adj * (self.global_adj + self.global_adj.transpose(1, 0)))
        if self_loop:
            adj = adj + torch.eye(num_nodes).to(self.device)
        rowsum = torch.sum(adj, dim=-1)
        mask = torch.zeros_like(rowsum)
        mask[rowsum == 0] = 1
        rowsum += mask
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        adj = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj

    def self_similarity(self, x):
        # x: b, node, feature
        x_ = x.permute(0, 2, 1)
        s = torch.bmm(x, x_)
        return s


class Aggregator():

    def __init__(self, idx_area):
        # chan_in_area: a list of the number of channels within each area
        self.chan_in_area = idx_area
        self.idx = self.get_idx(idx_area)
        self.area = len(idx_area)

    def forward(self, x):
        # x: batch x channel x data
        data = []
        for i, area in enumerate(range(self.area)):
            if i < self.area - 1:
                data.append(self.aggr_fun(x[:, self.idx[i]:self.idx[i + 1], :], dim=1))
            else:
                data.append(self.aggr_fun(x[:, self.idx[i]:, :], dim=1))
        return torch.stack(data, dim=1)

    def get_idx(self, chan_in_area):
        idx = [0] + chan_in_area
        idx_ = [0]
        for i in idx:
            idx_.append(idx_[-1] + i)
        return idx_[1:]

    def aggr_fun(self, x, dim):
        # return torch.max(x, dim=dim).values
        return torch.mean(x, dim=dim)

import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from util.pos_embed import get_1d_sincos_pos_embed_from_grid
import numpy as np

class My_GraphConvolution(Module):
    """
    simple GCN layer
    """
    def __init__(self, in_features, out_features,adj, bias=True):
        super(My_GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.adj = adj
        print('Weight size: ', self.weight.size())
        torch.nn.init.xavier_uniform_(self.weight, gain=1.414)
        if bias:
            self.bias = Parameter(torch.zeros((1, 1, out_features), dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        adj = self.adj.repeat(x.shape[0],0)
        print(adj.size())
        x = torch.einsum('nhq->nqh',x)
        output = torch.matmul(x, self.weight)-self.bias
        output = F.relu(torch.matmul(adj, output))
        output = torch.einsum('nhq->nqh',output)
        return output

class LinearProjectionLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearProjectionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.Linear_Layer = nn.Linear(self.in_features, self.out_features)

    def forward(self, x):
        x = torch.einsum('nhqw->nqwh',x)
        x = self.Linear_Layer(x)
        x = torch.einsum('nqwh->nhqw',x)
        print('AFter projection:', x.size())
        return x

class tempoalNet(nn.Module):
    def SCB(self, m, nChan, nBands, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
                # Conv2dWithConstraint(nBands, m*nBands, (1, 1), groups= nBands,
                #                      max_norm = 2 , doWeightNorm = doWeightNorm,padding = 0),
                Conv2dWithConstraint(nBands, m*nBands, (nChan, 1), groups= nBands, bias=True ,
                                     max_norm = 2, doWeightNorm = doWeightNorm,padding = 0),
                # LinearProjectionLayer(nBands, nBands * m),
                # My_GraphConvolution(nChan, nChan, adj),
                nn.BatchNorm2d(m*nBands),
                swish()
                )

    def LastBlock(self, inF, outF, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
                LinearWithConstraint(inF, outF, max_norm = 0.5, doWeightNorm = doWeightNorm, *args, **kwargs),
                nn.LogSoftmax(dim = 1))

    def __init__(self, nChan, nTime, nClass, nBands = 9, m = 48,
                 temporalLayer = 'LogVarLayer', doWeightNorm = True,\
                 strideFactor= 4, *args, **kwargs):
        super(tempoalNet, self).__init__()
        self.nBands = nBands
        self.m = m
        # self.adj = adj
        self.scb = self.SCB(m, nChan, self.nBands, doWeightNorm = doWeightNorm)
        self.strideFactor = strideFactor
        self.temporalLayer = LogVarLayer(dim = 3)
        self.multiHeadAttention = nn.ModuleList([nn.MultiheadAttention(embed_dim = 1,\
            num_heads = 1) for i in range(m*nBands)])
        self.pos_embed = get_1d_sincos_pos_embed_from_grid(int(432), int(4), cls_token=False).T
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lastLayer = self.LastBlock(self.m*self.nBands*self.strideFactor, nClass, doWeightNorm = doWeightNorm)
        self.GCNs = nn.ModuleList([GraphConvolution(nChan, nChan) for i in range(nBands)])
        
    def making_suitable_pos_embedded_array(self, batch_size, spectral_dim):
            pos_embedded_used = np.expand_dims(self.pos_embed,0).repeat(batch_size, axis= 0)
            pos_embedded_used = np.expand_dims(pos_embedded_used,3).repeat(spectral_dim, axis= 3)
            return torch.from_numpy(pos_embedded_used).to(self.device)
            
    def forward(self, x):
        x = self.scb(x)    
        x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3]/self.strideFactor)])
        x = self.temporalLayer(x)
        
        batch_size = x.size()[0]
        spectral_dim = x.size()[-1]
        pos_embedded_used = self.making_suitable_pos_embedded_array(batch_size, spectral_dim)
        for i, mh_attention in enumerate(self.multiHeadAttention):
            x_sliced = x[:,i,:,:] + pos_embedded_used[:,i,:,:]
            x_sliced = torch.squeeze(x[:,i,:,:],dim=1)
            x_sliced = torch.einsum('nhw->hnw',x_sliced)
            x_sliced, _ = mh_attention(x_sliced,x_sliced,x_sliced)
            x_sliced = torch.einsum('nhw->hnw',x_sliced)
            x[:,i,:,:] = x_sliced
        x = torch.flatten(x, start_dim= 1)
        x = self.lastLayer(x)
        return x

class LogVarLayer_(nn.Module):
    def __init__(self, dim):
        super(LogVarLayer_, self).__init__()
        self.dim = dim

    def forward(self, x):
        var = torch.log(torch.clamp(x.var(dim = self.dim, keepdim= True), 1e-6, 1e6))
        mean = torch.log(torch.clamp(x.mean(dim = self.dim, keepdim= True), 1e-6, 1e6))
        output = torch.cat((var, mean),dim=-1)
        return output

class tempoalNet_(nn.Module):
    def SCB(self, m, nChan, nBands, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
                Conv2dWithConstraint(nBands, m*nBands, (nChan, 1), groups= nBands,
                                     max_norm = 2 , doWeightNorm = doWeightNorm,padding = 0),
                nn.BatchNorm2d(m*nBands),
                swish()
                )

    def LastBlock(self, inF, outF, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
                LinearWithConstraint(inF, outF, max_norm = 0.5, doWeightNorm = doWeightNorm, *args, **kwargs),
                nn.LogSoftmax(dim = 1))

    def __init__(self, nChan, nTime, nClass, nBands = 9, m = 32,
                 temporalLayer = 'LogVarLayer', doWeightNorm = True,\
                 strideFactor= 4, embed_dim= 2, *args, **kwargs):
        super(tempoalNet_, self).__init__()
        self.nBands = nBands
        self.m = m
        self.scb = self.SCB(m, nChan, self.nBands, doWeightNorm = doWeightNorm)
        self.strideFactor = strideFactor
        self.temporalLayer = LogVarLayer_(dim = 3)
        self.multiHeadAttention = nn.ModuleList([nn.MultiheadAttention(embed_dim = embed_dim,\
            num_heads = 1) for i in range(m*nBands)])
        self.pos_embed = get_1d_sincos_pos_embed_from_grid(int(embed_dim), int(strideFactor), cls_token=False)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lastLayer = self.LastBlock(self.m*self.nBands*self.strideFactor*embed_dim, nClass, doWeightNorm = doWeightNorm)

    def making_suitable_pos_embedded_array(self, batch_size, spectral_dim):
            pos_embedded_used = np.expand_dims(self.pos_embed,0).repeat(batch_size, axis= 0)
            return torch.from_numpy(pos_embedded_used).to(self.device)
            
    def forward(self, x):
        x = self.scb(x)    
        x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3]/self.strideFactor)])
        x = self.temporalLayer(x)
        batch_size = x.size()[0]
        spectral_dim = x.size()[-1]
        pos_embedded_used = self.making_suitable_pos_embedded_array(batch_size, spectral_dim)
        print(x.size())
        for i, mh_attention in enumerate(self.multiHeadAttention):
            x_sliced = x[:,i,:,:] + pos_embedded_used[:,:,:]
            x_sliced = torch.squeeze(x[:,i,:,:],dim=1)
            x_sliced = torch.einsum('nhw->hnw',x_sliced)
            x_sliced, _ = mh_attention(x_sliced,x_sliced,x_sliced)
            x_sliced = torch.einsum('nhw->hnw',x_sliced)
            x[:,i,:,:] = x_sliced
        x = torch.flatten(x, start_dim= 1)
        x = self.lastLayer(x)
        return x

class tempo_layer(nn.Module):
    def __init__(self, dim):
        super(tempo_layer, self).__init__()
        self.dim = dim

    def forward(self, x):
        var = torch.log(torch.clamp(x.var(dim = self.dim, keepdim= True), 1e-6, 1e6))
        mean = torch.log(torch.clamp(x.mean(dim = self.dim, keepdim= True), 1e-6, 1e6))
        
        len_ = x.size()[-1]
        mean_ = x.mean(dim = self.dim, keepdim= True)

        for i in range(len_):
            x_rms += torch.unsqueeze((x[:,:,:,i] ** 2),dim=-1)

        x_rms = torch.sqrt(x_rms / len_) # 7.均方根值
        W = x_rms / mean_ # 10.波形指标 - 
        output = torch.cat((var, W),dim=-1)
        # print(output.size())
        return output

class tempoalNet_me(nn.Module):
    def SCB(self, m, nChan, nBands, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
                Conv2dWithConstraint(nBands, m*nBands, (nChan, 1), groups= nBands,
                                     max_norm = 2 , doWeightNorm = doWeightNorm,padding = 0),
                nn.BatchNorm2d(m*nBands),
                swish()
                )

    def LastBlock(self, inF, outF, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
                LinearWithConstraint(inF, outF, max_norm = 0.5, doWeightNorm = doWeightNorm, *args, **kwargs),
                nn.LogSoftmax(dim = 1))

    def __init__(self, nChan, nTime, nClass, nBands = 9, m = 32, doWeightNorm = True,\
                 strideFactor= 4, embed_dim= 2, *args, **kwargs):
        super(tempoalNet_me, self).__init__()
        self.nBands = nBands
        self.m = m
        self.scb = self.SCB(m, nChan, self.nBands, doWeightNorm = doWeightNorm)
        self.strideFactor = strideFactor
        self.data_tempo_len = int(self.strideFactor * 2 - 1)
        self.temporalLayer = tempo_layer(dim = 3)
        self.multiHeadAttention = nn.ModuleList([nn.MultiheadAttention(embed_dim = embed_dim,\
            num_heads = 2) for i in range(m*nBands)])
        self.pos_embed = get_1d_sincos_pos_embed_from_grid(int(embed_dim), int(self.data_tempo_len), cls_token=False)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lastLayer = self.LastBlock(self.m*self.nBands*self.data_tempo_len*embed_dim, nClass, doWeightNorm = doWeightNorm)

    def making_suitable_pos_embedded_array(self, batch_size, spectral_dim):
            pos_embedded_used = np.expand_dims(self.pos_embed,0).repeat(batch_size, axis= 0)
            return torch.from_numpy(pos_embedded_used).to(self.device)
            
    def forward(self, x):

        x = self.scb(x)
        tempal_size = int(x.size()[-1]/self.strideFactor/2)
        for i in range(int(self.strideFactor * 2 - 1)):
            if(i == 0):
                x_stacked = x[:,:,:,:int(tempal_size * 2)]
            else:
                x_stacked = torch.cat((x_stacked, x[:, :, :, int(i * tempal_size):\
                    int(i * tempal_size + (tempal_size*2))]),dim=2)

        x = self.temporalLayer(x_stacked)
        batch_size = x.size()[0]
        spectral_dim = x.size()[-1]
        pos_embedded_used = self.making_suitable_pos_embedded_array(batch_size, spectral_dim)

        for i, mh_attention in enumerate(self.multiHeadAttention):
            x_sliced = x[:,i,:,:] + pos_embedded_used
            x_sliced = torch.einsum('nhw->hnw', x_sliced)
            x_sliced = x_sliced.float()
            x_sliced, _ = mh_attention(x_sliced, x_sliced, x_sliced)
            x_sliced = torch.einsum('nhw->hnw', x_sliced)
            x[:,i,:,:] = x_sliced

        x = torch.flatten(x, start_dim= 1)
        x = self.lastLayer(x)
        return x

class TSceptionIJCNN(nn.Module):
    """
    Y. Ding et al., "TSception: A Deep Learning Framework for Emotion Detection Using EEG"
    2020 International Joint Conference on Neural Networks (IJCNN), Glasgow, UK, 2020, pp. 1-7,
    doi: 10.1109/IJCNN48605.2020.9206750.
    """
    def conv_block(self, in_chan, out_chan, kernel, step, pool):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=step, padding=0),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))

    def __init__(self, num_classes, input_size, sampling_rate, num_T, num_S, hidden, dropout_rate, frequ_bands_amount):
        # input_size: 1 x EEG channel x datapoint
        super(TSceptionIJCNN, self).__init__()
        self.inception_window = [0.5, 0.25, 0.125]
        self.frequ_bands_amount = frequ_bands_amount
        self.pool = 8
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = self.conv_block(frequ_bands_amount, num_T, (1, int(self.inception_window[0] * sampling_rate)), 1, self.pool)
        self.Tception2 = self.conv_block(frequ_bands_amount, num_T, (1, int(self.inception_window[1] * sampling_rate)), 1, self.pool)
        self.Tception3 = self.conv_block(frequ_bands_amount, num_T, (1, int(self.inception_window[2] * sampling_rate)), 1, self.pool)

        self.Sception1 = self.conv_block(num_T, num_S, (int(input_size[-2]), 1), 1, int(self.pool*0.25))
        self.Sception2 = self.conv_block(num_T, num_S, (int(input_size[-2] * 0.5), 1), (int(input_size[-2] * 0.5), 1),
                                         int(self.pool*0.25))
        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)

        size = self.get_size(input_size)
        self.fc = nn.Sequential(
            nn.Linear(size[1], hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        # x = torch.unsqueeze(x, 1)
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_ = z
        z = self.Sception2(out)
        out_ = torch.cat((out_, z), dim=2)
        out = self.BN_s(out_)
        # print(out.size())
        out = out.reshape(out.size()[0], -1)
        out = self.fc(out)
        return out

    def get_size(self, input_size):
        # here we use an array with the shape being
        # (1(mini-batch),1(convolutional channel),EEG channel,time data point)
        # to simulate the input data and get the output size
        data = torch.ones((1, self.frequ_bands_amount, input_size[-2], int(input_size[-1])))
        y = self.Tception1(data)
        out = y
        y = self.Tception2(data)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(data)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_final = z
        z = self.Sception2(out)
        out_final = torch.cat((out_final, z), dim=2)
        out = self.BN_s(out_final)
        out = out.view(out.size()[0], -1)
        return out.size()
