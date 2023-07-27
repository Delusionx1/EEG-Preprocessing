import torch
import torch.nn as nn
import sys
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
    
class swish(nn.Module):
    '''
    The swish layer: implements the swish activation function
    '''
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
    
class PowerLayer_me(nn.Module):
    '''
    The standard deviation layer: calculates the std of the data along given 'dim'
    '''

    def __init__(self, dim, length, step):
        super(PowerLayer_me, self).__init__()
        self.dim = dim
        self.pooling = nn.AvgPool2d(kernel_size=(1, length), stride=(1, step))

    def forward(self, x):
        return torch.log(self.pooling(x.pow(3)))

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
       
class FBSCNet(nn.Module):
    # just a FBCSP like structure : chan conv and then variance along the time axis
    '''
        FBNet with seperate variance for every 1s. 
        The data input is in a form of batch x 1 x chan x time x filterBand
    '''
    
    def temporal_learner(self, in_chan, out_chan, kernel, pool, pool_step_rate):
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=kernel, stride=(1, 3))
            # PowerLayer_me(dim=-1, length=pool, step=int(pool_step_rate*pool))
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
                LinearWithConstraint(inF, int(inF*0.25), max_norm = 0.5, doWeightNorm = doWeightNorm, *args, **kwargs),
                LinearWithConstraint(int(inF*0.25), outF, max_norm = 0.5, doWeightNorm = doWeightNorm, *args, **kwargs),
                nn.LogSoftmax(dim = 1))

    def __init__(self, nChan, nTime, nClass, nBands = 9, m = 48,
                 temporalLayer = 'LogVarLayer', strideFactor= 4,sampling_rate = 250, pool=16, pool_step_rate = 0.25, doWeightNorm = True, *args, **kwargs):
        super(FBSCNet, self).__init__()
        self.pool = pool
        self.nBands = nBands
        self.m = m
        self.strideFactor = strideFactor
        self.window = [0.33, 0.18, 0.04, 0.66]
        # create all the parrallel SCBc
        self.scb = self.SCB(m, nChan, self.nBands, doWeightNorm = doWeightNorm)

        self.temporalLayer_me_1 = self.temporal_learner(in_chan=m*nBands, out_chan=int(m*nBands*0.75),
                                               kernel=(1, int(self.window[0] * sampling_rate)),
                                               pool=self.pool, pool_step_rate=pool_step_rate)
        
        self.temporalLayer_me_2 = self.temporal_learner(in_chan=m*nBands, out_chan=int(m*nBands*0.75),
                                        kernel=(1, int(self.window[1] * sampling_rate)),
                                        pool=self.pool, pool_step_rate=pool_step_rate)
        
        self.temporalLayer_me_3 = self.temporal_learner(in_chan=m*nBands, out_chan=int(m*nBands*0.75),
                                               kernel=(1, int(self.window[2] * sampling_rate)),
                                               pool=self.pool, pool_step_rate=pool_step_rate)
        
        self.temporalLayer_me_4 = self.temporal_learner(in_chan=m*nBands, out_chan=int(m*nBands*0.75),
                                               kernel=(1, int(self.window[3] * sampling_rate)),
                                               pool=self.pool, pool_step_rate=pool_step_rate)
        # Formulate the temporal agreegator
        self.temporalLayer = LogVarLayer(dim = 3)

        # The final fully connected layer
        self.lastLayer = self.LastBlock(int(self.m*self.nBands*0.75 * self.strideFactor*4), nClass, doWeightNorm = doWeightNorm)

    def forward(self, x):
        x = self.scb(x)
        # print(x)
        x_1 = self.temporalLayer_me_1(x)
        # print(x_1.shape)
        x_1 = x_1.reshape([*x_1.shape[0:2], self.strideFactor, int(x_1.shape[3]/self.strideFactor)])
        # print(x_1.size())
        x_1 = self.temporalLayer(x_1)
        
        x_2 = self.temporalLayer_me_2(x)
        # print(x_2.shape)
        x_2 = x_2.reshape([*x_2.shape[0:2], self.strideFactor, int(x_2.shape[3]/self.strideFactor)])
        # print(x_2.size())
        x_2 = self.temporalLayer(x_2)
        
        x_3 = self.temporalLayer_me_3(x)
        # print(x_3.shape)
        x_3 = x_3.reshape([*x_3.shape[0:2], self.strideFactor, int(x_3.shape[3]/self.strideFactor)])
        x_3 = self.temporalLayer(x_3)

        x_4 = self.temporalLayer_me_4(x)
        # print(x_3.shape)
        x_4 = x_4.reshape([*x_4.shape[0:2], self.strideFactor, int(x_4.shape[3]/self.strideFactor)])
        x_4 = self.temporalLayer(x_4)

        # print(x_1.shape, x_2.shape, x_3.shape)
        x_all = torch.cat((x_1, x_2, x_3, x_4), dim = -1)
        x = torch.flatten(x_all, start_dim= 1)
        x = self.lastLayer(x)
        return x
    
# if __name__ == '__main__':
#         model = FBSCNet(nClass=4, nChan=22,nTime=800, nBands=9, m = 32, \
#                         sampling_rate = 250, dropout_rate = 0.5, pool=16, pool_step_rate = 0.25)
#         data = nn.Parameter(torch.rand(16, 9, 22, 800))
#         output = model(data)