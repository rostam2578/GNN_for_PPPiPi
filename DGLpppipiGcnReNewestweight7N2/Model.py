# Defining the network
print('\n\n\n\n', 'The Network ...', '\n\n')
from GraphDef import *
from DataLoad import *

import torch
from torch.nn import Linear
from dgl.nn import GraphConv
import dgl.function as fn
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import datetime
device = torch.device('cuda')

#########################################################
modelname = 'DGLpppipiGcnReNewestweight7N2'
print('Model name:', modelname)
class GCN(nn.Module):
    def __init__(self, in_feats, num_classes, batchsize):
        super(GCN, self).__init__()
        self.batchsizeini = batchsize
        #self.efeatur = nn.Parameter(torch.cat([torch.randn((36593 * 2, 1), requires_grad=True).to(device)] * batchsize))
        #self.efeatur = torch.ones((36593 * 2, 1), device=device)
        self.conv1 = GraphConv(in_feats, 256)
        self.conv2 = GraphConv(256, 128)
        self.conv3 = GraphConv(128, 64)
        self.conv4 = GraphConv(64, 32)
        self.conv5 = GraphConv(32, num_classes)
        #self.fc1 = nn.Linear(6796, 256)
        #self.fc2 = nn.Linear(256, 128)
        #self.fc3 = nn.Linear(128, 64)
        #self.fc4 = nn.Linear(64, 6796)

    def forward(self, g, in_fet):#, eweights):
        #weighttensorbatch = torch.cat([weighttensor] * self.batchsizeini)
        g.ndata['nfet'] = in_fet
        # Lets put all the graphs coming from a conv layer on top of 
        # each other as a node feature and then update all at once 
        # with message passing. An example is provided at the end of 
        # this file. We need to notice h is different and independent 
        # of g.ndata['h'].
        # Self loop can be done by either adding self edges or by
        # adding another edge feature which couples with the source 
        # node feature. We can simply add node features to the message 
        # obtained with or without a coupling.
        h = self.conv1(g, in_fet)
        g.ndata['h1'] = F.relu(h)
        g.update_all(fn.u_mul_e('h1', 'efet', 'm'), fn.sum('m', 'h1'))
        h1 = F.relu(g.ndata['h1'])
        #
        h2 = self.conv2(g, h1)
        g.ndata['h2'] = F.relu(h2)
        g.update_all(fn.u_mul_e('h2', 'efet', 'm'), fn.sum('m', 'h2'))
        h2 = F.relu(g.ndata['h2'])
        #
        h3 = self.conv3(g, h2)
        g.ndata['h3'] = F.relu(h3)
        g.update_all(fn.u_mul_e('h3', 'efet', 'm'), fn.sum('m', 'h3'))
        h3 = F.relu(g.ndata['h3'])
        #
        h4 = self.conv4(g, h3)
        h4 = F.relu(h4)
        #
        h5 = self.conv5(g, h4)
        #
        #h = torch.flatten(h, 1)
        #h = F.relu(self.fc1(h))
        #h = F.relu(self.fc2(h))
        #h = F.relu(self.fc3(h))
        #h = F.sigmoid(self.fc4(h))
        # tracking what happens at any step
        print('\n\n\ninput node feature: \ng.ndata[nfet]', g.ndata['nfet'], '\ng.ndata[nfet].shape', g.ndata['nfet'].shape, '\ng.ndata[nfet].sum', g.ndata['nfet'].sum())
        print('\n\n\ninput graph: \ng', g, \
            '\ng.edata[efet].shape', g.edata['efet'].shape, '\ng.edata[efet]', g.edata['efet'], '\ng.edata[efet].sum', g.edata['efet'].sum(), \
            '\ng.ndata[nfet].shape', g.ndata['nfet'].shape, '\ng.ndata[nfet]', g.ndata['nfet'], '\ng.ndata[nfet].sum', g.ndata['nfet'].sum())
        for name, param in net.named_parameters():
            if name == "conv1.weight":
                param0_0 = param.data[0]
                print("param0_0.shape", param0_0.shape)
                param0 = param.data[0, 0]
                param100 = param.data[0, 100]
                param200 = param.data[0, 200]
            if name == "conv2.weight":
                print('param.data[:, 0].shape', param.data[:, 0].shape)
                param0_2 = param.data[:, 0]
                param50_2 = param.data[:, 50]
                param100_2 = param.data[:, 100]
            if name == "conv2.bias":
                bias0 = param.data[0]
                bias50 = param.data[50]
                bias100 = param.data[100]
        print('\n\n\nh after the first convolutional layer: \n', h, '\nh.shape', h.shape, '\nh.sum', h.sum())
        print('\n\n\nh[:, 0].sum', h[:, 0].sum())
        print('\ng.ndata[nfet].sum() * conv1.weight[0]', g.ndata['nfet'].sum() * param0)
        print('\n\n\nh[100].sum', h[:, 100].sum())
        print('\ng.ndata[nfet].sum() * conv1.weight[100]', g.ndata['nfet'].sum() * param100)
        print('\n\n\nh[200].sum', h[:, 200].sum())
        print('\ng.ndata[nfet].sum() * conv1.weight[200]', g.ndata['nfet'].sum() * param200)
        #
        print('\n\n\nh1 after relu, the first updating, and another relu: \n', h1, '\nh.shape', h1.shape, '\nh.sum', h1.sum())
        #
        print('\n\n\nh2 after the second convolutional layer: \n', h2, '\nh2.shape', h2.shape, '\nh2.sum', h2.sum())
        print('\n\n\nh2[0].sum', h2[:, 0].sum())
        print('\n(h1.sum(axis=0) * param0_2).sum() + bias0', (h1.sum(axis=0) * param0_2).sum() + bias0)
        print('\n\n\nh2[100].sum', h2[:, 50].sum())
        print('\n(h1.sum(axis=0) * param50_2).sum() + bias50', (h1.sum(axis=0) * param50_2).sum() + bias50)
        print('\n\n\nh2[200].sum', h2[:, 100].sum())
        print('\n(h1.sum(axis=0) * param100_2).sum() + bias100', (h1.sum(axis=0) * param100_2).sum() + bias100)
        #
        print('\n\n\ng', g)
        print('\n\n\n output, \nh5', h5, '\nh5.shape', h5.shape, '\nh5.sum', h5.sum(), '\ng.edata[efet]', g.edata['efet'], '\ng.edata[efet].shape', g.edata['efet'].shape, '\ng.edata[efet].sum', g.edata['efet'].sum())
        return h5

#vars()[modelname] = GCN(1, 1).to(device)
#net = vars()[modelname]
net = GCN(1, 1, 1).to(device)
print("net", net)

#########################################################
for name, param in net.named_parameters():
    print(name, '\n', param.data.shape, '\n', param.requires_grad , '\n', param.data, '\n', param)

net = GCN(1, 1, 2).to(device)
for name, param in net.named_parameters():
    print(name, '\n', param.data.shape, '\n', param.requires_grad , '\n', param.data, '\n', param)

#########################################################
# one sample event
EvBTr = 20

fig = plt.figure(figsize=(40, 21))
plt.rcParams['font.size'] = '18'
ax1 = fig.add_subplot(321)
fig.colorbar(ax1.matshow(sitonsquare(traingnnpppipi[EvBTr]), aspect=2, vmin=0, vmax=1, extent=[0, 288, 0, 43], origin='lower')\
                ,fraction=0.014)
plt.xlabel('cell')
plt.ylabel('layer')
plt.xlim(0, 288)
plt.ylim(0, 43)
plt.title(f'event number {EvBTr} with noise \n color indicates time')

# Passing the sample data from the network before training
net = GCN(1, 1, 1).to(device)
result1 = net(dglgraph, TraTen[EvBTr].reshape(6796, 1))
print(f'\nPassing event {EvBTr} from the network before training', '\nresult1:', result1, '\nresult1.shape:', result1.shape, '\ninput:', traingnnpppipi[EvBTr])
ax2 = fig.add_subplot(322)
fig.colorbar(ax2.matshow(sitonsquare(result1.reshape(6796)), aspect=2, extent=[0, 288, 0, 43], origin='lower')\
                ,fraction=0.014)                
plt.xlabel('cell')
plt.ylabel('layer')
plt.xlim(0, 288)
plt.ylim(0, 43)
plt.title(f'event number {EvBTr} passed through network before training \n color indicates time')

#from torch.utils.data import DataLoader
#TraTenBat = DataLoader(TraTen, batch_size=2, shuffle=True)

# Passing two sample events from the network before training
batcheddglgraph = dgl.batch([dglgraph, dglgraph])
featwo = TraTen[EvBTr + 10: EvBTr + 12].reshape(2 * 6796, 1)
net = GCN(1, 1, 2).to(device)
result2 = net(batcheddglgraph, featwo)
print(f'\nPassing two random events from the network before training', '\nresult1:', result1, '\nresult1.shape:', result1.shape, '\ninput:', traingnnpppipi[EvBTr])


# the first event
ax3 = fig.add_subplot(323)
fig.colorbar(ax3.matshow(sitonsquare(traingnnpppipi[EvBTr + 10]), aspect=2, vmin=0, vmax=1, extent=[0, 288, 0, 43], origin='lower')\
                ,fraction=0.014)
plt.xlabel('cell')
plt.ylabel('layer')
plt.xlim(0, 288)
plt.ylim(0, 43)
plt.title(f'event number {EvBTr + 10} with noise \n color indicates time')

# pass from the net
ax4 = fig.add_subplot(324)
fig.colorbar(ax4.matshow(sitonsquare(result2.reshape(2, 6796)[0]), aspect=2, extent=[0, 288, 0, 43], origin='lower')\
                ,fraction=0.014)                
plt.xlabel('cell')
plt.ylabel('layer')
plt.xlim(0, 288)
plt.ylim(0, 43)
plt.title(f'event {EvBTr + 10} passed through network in a batch before training \n color indicates time')

# the second event
ax5 = fig.add_subplot(325)
fig.colorbar(ax5.matshow(sitonsquare(traingnnpppipi[EvBTr + 11]), aspect=2, vmin=0, vmax=1, extent=[0, 288, 0, 43], origin='lower')\
                ,fraction=0.014)
plt.xlabel('cell')
plt.ylabel('layer')
plt.xlim(0, 288)
plt.ylim(0, 43)
plt.title(f'event number {EvBTr + 11} with noise \n color indicates time')

# pass from the net
ax6 = fig.add_subplot(326)
fig.colorbar(ax6.matshow(sitonsquare(result2.reshape(2, 6796)[1]), aspect=2, extent=[0, 288, 0, 43], origin='lower')\
                ,fraction=0.014)                
plt.xlabel('cell')
plt.ylabel('layer')
plt.xlim(0, 288)
plt.ylim(0, 43)
plt.title(f'event {EvBTr + 11} passed through network in a batch before training \n color indicates time')

#########################################################
t = datetime.datetime.now()
plt.savefig(f'/hpcfs/bes/mlgpu/hoseinkk/MLTracking/otherparticles/pppipi/{modelname}/results/{t}\
    passing three random events {EvBTr, EvBTr + 10, EvBTr + 11} from network before training.png', bbox_inches='tight')

#########################################################
# message passing example for multi dimentional node feature:
#g = dgl.graph(([0, 0, 0, 1], [1, 2, 3, 2]))
#g.edata['efet'] = torch.tensor([0.7, 0.7, 0.7, 0.7])
#tens1 = torch.tensor(([0.5, 0.5, 0.5, 0.5], [0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]))
#g.ndata['h'] = torch.transpose(tens1, 0, 1)
#g.update_all(fn.u_mul_e('h', 'efet', 'm'), fn.sum('m', 'h1'))
#g = dgl.add_reverse_edges(g)
#g.edata['efet'] = torch.tensor([0.7] * 8)
#g.update_all(fn.u_mul_e('h', 'efet', 'm'), fn.sum('m', 'h2'))
