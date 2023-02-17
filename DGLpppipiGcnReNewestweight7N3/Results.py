print('\n\n\n\n', 'Results ...', '\n\n')
from Model import *

# Results
import matplotlib.pyplot as plt
import os
from os import path
import numpy as np
import datetime

#########################################################
SampEv = 75001
load_model = True
Thrd = 0.1
TesNum = 1000
TesStart = 75000
print("SampEv", SampEv, "load_model", load_model, "Thrd", Thrd, "TesNum", TesNum, "TesStart", TesStart)

log_dir = "/hpcfs/bes/mlgpu/hoseinkk/MLTracking/bhabha/bhabha-files/" + modelname #+ time.time()
checkpoint_dir = "/hpcfs/bes/mlgpu/hoseinkk/MLTracking/bhabha/bhabha-files/" + modelname

def checkpoint_load(loadedcheckpoint):
  print("=> loading checkpoint from", F"{checkpoint_dir}/saved_checkpoint.pth.tar")
  net.load_state_dict(loadedcheckpoint['state_dict'])
#  optimizer.load_state_dict(loadedcheckpoint['optimizer'])

if load_model:
  checkpoint_load(torch.load(F"{checkpoint_dir}/saved_checkpoint.pth.tar"))

#########################################################
# one sample event
fig = plt.figure(figsize=(30, 14))
ax1 = fig.add_subplot(321)
fig.colorbar(ax1.matshow(sitonsquare(traingnn80[SampEv]), aspect=3, vmin=0, vmax=1, extent=[0, 288, 0, 43], origin='lower')\
                ,fraction=0.014)
plt.xlabel('cell')
plt.ylabel('layer')
plt.xlim(0, 288)
plt.ylim(0, 43)
plt.title(f'event number {SampEv} with noise \n color indicates time')

ax2 = fig.add_subplot(322)
fig.colorbar(ax2.matshow(sitonsquare(trvalgnn80[SampEv]), aspect=3, vmin=0, vmax=1, extent=[0, 288, 0, 43], origin='lower')\
                ,fraction=0.014)
plt.xlabel('cell')
plt.ylabel('layer')
plt.xlim(0, 288)
plt.ylim(0, 43)
plt.title(f'event number {SampEv} with noise \n color indicates time')

# Passing the sample data from the network
result1 = net(dglgraph, TraTen[SampEv].reshape(6796, 1))
ax3 = fig.add_subplot(325)
fig.colorbar(ax3.matshow(sitonsquare(result1.reshape(6796)), aspect=3, extent=[0, 288, 0, 43], origin='lower')\
                ,fraction=0.014)                
plt.xlabel('cell')
plt.ylabel('layer')
plt.xlim(0, 288)
plt.ylim(0, 43)
plt.title(f'event number {SampEv} passed through network \
          \n Model is trained: {load_model}, color indicates time')

# Passing the sample data from the network with threshold
result2 = result1 > Thrd
ax4 = fig.add_subplot(323)
fig.colorbar(ax4.matshow(sitonsquare(result2.reshape(6796)), aspect=3, extent=[0, 288, 0, 43], origin='lower')\
                ,fraction=0.014)                
plt.xlabel('cell')
plt.ylabel('layer')
plt.xlim(0, 288)
plt.ylim(0, 43)
plt.title(f'event number {SampEv} passed through network \
          \n Model is trained: {load_model}, threshold is {Thrd}, color indicates time')

# Passing the sample data from the network with threshold and intersected with the evet itself.
result3 = (result2 & (TraTen[SampEv].reshape(6796, 1) > 0))
ax5 = fig.add_subplot(324)
fig.colorbar(ax5.matshow(sitonsquare(result3.reshape(6796)), aspect=3, extent=[0, 288, 0, 43], origin='lower')\
                ,fraction=0.014)                
plt.xlabel('cell')
plt.ylabel('layer')
plt.xlim(0, 288)
plt.ylim(0, 43)
plt.title(f'event number {SampEv} passed through network \
          \n Model is trained: {load_model}, threshold is {Thrd}, result is intersected with the inut, color indicates time')

t = datetime.datetime.now()
plt.savefig(f'/hpcfs/bes/mlgpu/hoseinkk/MLTracking/bhabha/GNN/{modelname}/results/{t}\
  GNN result for event number {SampEv} with threshold {Thrd} and model trained {load_model}.png', bbox_inches='tight')

#########################################################
#purity and efficiency for EvaNum test events
plnum = 50
starpoi = 0
endpoi = 1
stps = 0.02
purres = np.zeros(shape = plnum)
effres = np.zeros(shape = plnum)
purity = np.zeros(shape = (TesNum, plnum))
efficiency = np.zeros(shape=(TesNum, plnum))

for i in range(TesStart, TesStart + TesNum):
  result4 = net(dglgraph.to(device), TraTen[i].reshape(6796, 1).to(device))
  truevaluebatch = ((TrvTen[i]) >= 1).type(torch.float)#.to(device)
  gnnres = (torch.Tensor.cpu(result4).detach().numpy()).reshape(6796)
  for j in range(0, plnum):
     thrres = gnnres > (0 + j * stps)
     datnoi = torch.Tensor.cpu(TraTen[i].reshape(6796)).numpy() > 0
     datnon = torch.Tensor.cpu(TrvTen[i].reshape(6796)).numpy() > 0
     outres = thrres & datnoi 
     comres = outres & datnon
     purity[i - TesStart, j] = comres.sum() / outres.sum()
     efficiency[i - TesStart, j] = comres.sum() / datnon.sum()
purres = purity.mean(axis = 0)
effres = efficiency.mean(axis = 0)

plt.figure(figsize=(25, 20))
plt.rcParams['font.size'] = '18'
plt.scatter(np.arange(starpoi, endpoi, stps), effres, label='efficiency')
plt.scatter(np.arange(starpoi, endpoi, stps), purres, label='purity')
plt.legend()
plt.xlabel(f'threshold')
plt.xlim([starpoi, endpoi])
plt.savefig(f'/hpcfs/bes/mlgpu/hoseinkk/MLTracking/bhabha/GNN/{modelname}/results/{t}\
  Purity and efficiency.png', bbox_inches='tight')

'''
# for when the model is not trained
plnum = 100
purres = np.zeros(shape = plnum)
effres = np.zeros(shape = plnum)
purity = np.zeros(shape = (TesNum, plnum))
efficiency = np.zeros(shape=(TesNum, plnum))

for i in range(TesStart, TesStart + TesNum):
  result3 = net(dglgraph.to(device), TraTen[i].reshape(6796, 1).to(device))
  truevaluebatch = ((TrvTen[i]) >= 1).type(torch.float)#.to(device)
  gnnres = (torch.Tensor.cpu(result3).detach().numpy()).reshape(6796)
  for j in range(0, plnum):
     thrres = gnnres < (0 - 0.01 + j * 0.0001)
     datnoi = torch.Tensor.cpu(TraTen[i].reshape(6796)).numpy() > 0
     datnon = torch.Tensor.cpu(TrvTen[i].reshape(6796)).numpy() > 0
     outres = thrres & datnoi 
     comres = outres & datnon
     purity[i - TesStart, j] = comres.sum() / outres.sum()
     efficiency[i - TesStart, j] = comres.sum() / datnon.sum()
purres = purity.mean(axis = 0)
effres = efficiency.mean(axis = 0)

plt.figure(figsize=(25, 20))
plt.scatter(np.arange(-0.01, 0, 0.0001), purres, label='purity')
plt.scatter(np.arange(-0.01, 0, 0.0001), effres, label='efficiency')
plt.legend()
plt.xlabel('threshold')
plt.xlim([-0.01, 0])
plt.savefig(f'/hpcfs/bes/mlgpu/hoseinkk/MLTracking/bhabha/GNN/DGLBhabhaGcnReNewest3/results/Purity and efficiency if not trained-{t}.png', bbox_inches='tight')
'''