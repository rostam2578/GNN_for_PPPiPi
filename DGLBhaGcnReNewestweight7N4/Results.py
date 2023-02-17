from DataLoadBha import *
from ModelBha import *
import numpy as np
import os
from os import path 
device = torch.device('cuda')


#########################################################
# we need three values to call the correct checkpoint, TraEvN, EpochNum, and startmesh.
TraEvN = 9001 #7801 * 2
EpochNum = 6
load_model = True
hpmesh = []
for BatchSize in 5 * np.array([1, 3, 6, 10, 14, 19]): #[39, 117, 234, 390, 546, 741]
    for LrVal in np.array([0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]):
        for weight_decay_val in np.array([0, 0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]): #np.arange(0.001, 0.01, 0.009):
            hpmesh.append([BatchSize, LrVal, weight_decay_val])
hpmesh = np.array(hpmesh)
startmesh, endmesh = 284, 285 #305, 306
BatchSize, LrVal, weight_decay_val = hpmesh[startmesh:endmesh][0]

net = GCN(1, 1).to(device)
checkpoint_dir_path = "/hpcfs/bes/mlgpu/hoseinkk/MLTracking/otherparticles/pppipi/" + modelname 
def checkpoint_load(loadedcheckpoint):
  print("=> loading checkpoint from", F"{checkpoint_dir_path}/checkpoint_dir/{TraEvN}{EpochNum}{startmesh}saved_checkpoint.tar")
  net.load_state_dict(loadedcheckpoint['state_dict'])

checkpoint_load(torch.load(F"{checkpoint_dir_path}/checkpoint_dir/{TraEvN}{EpochNum}{startmesh}saved_checkpoint.tar"))
print("\n\n\nload_model", load_model, "\nTraEvN", TraEvN, "\nBatchSize", BatchSize, \
  "\nEpochNum", EpochNum, "\nLrVal", LrVal, '\nweight_decay', weight_decay_val, '\nstartmesh', startmesh, '\nendmesh', endmesh, '\n\n\n')

#########################################################
# Three sample events, passing from the network before training, after it, and with various thresholds

#########################################################
#plots
thr = [2e-1, 4e-1, 5e-1, 6e-1, 7e-1]
batcheddglgraph = dgl.batch(10 * [dglgraph])#.to('cpu')
print('\nnet', net, '\nbatcheddglgraph', batcheddglgraph, '\nTraTen', TraTen)
result1 = net(batcheddglgraph, TraTen[10000:10010].reshape(10 * 6796, 1))
net0 = GCN(1, 1).to(device)
result0 = net0(batcheddglgraph, TraTen[10000:10010].reshape(10 * 6796, 1))
print('\nresult1', result1, '\nresult1.shape', result1.shape, '\nresult0', result0, '\nresult0.shape', result0.shape)

def plotevent(event, posit, titletext):
  ax = fig.add_subplot(posit)
  fig.colorbar(ax.matshow(event, aspect=2, vmin=0, vmax=1, extent=[0, 288, 0, 43], origin='lower')\
                  ,fraction=0.014)
  plt.xlabel('cell')
  plt.ylabel('layer')
  plt.xlim(0, 288)
  plt.ylim(0, 43)
  plt.title(titletext)

# first event
fig = plt.figure(figsize=(40, 15))
plt.rcParams['font.size'] = '15'
EvBTr = 10003
plotevent(sitonsquare(traingnnpppipi[EvBTr]), 331, f'event number {EvBTr} with noise \n color indicates time')
plotevent(sitonsquare(trvalgnnpppipi[EvBTr]), 332, f'event number {EvBTr} without noise \n color indicates time')
plotevent(sitonsquare(result0.reshape(10, 6796)[EvBTr - 10000]), 333, f'event number {EvBTr} passed through network before training \n color indicates time')
# outcome
plotevent(sitonsquare(result1.reshape(10, 6796)[EvBTr - 10000]), 334, f'event number {EvBTr} passed through network after training \n color indicates time')
# with threshold
plotevent(sitonsquare((result1.reshape(10, 6796)[EvBTr - 10000] > thr[0]).cpu() & ((trvalgnnpppipi[EvBTr]) > 0)), 335, f'event number {EvBTr} passed through network after training \n threshold is {thr[0]}')
plotevent(sitonsquare((result1.reshape(10, 6796)[EvBTr - 10000] > thr[1]).cpu() & ((trvalgnnpppipi[EvBTr]) > 0)), 336, f'event number {EvBTr} passed through network after training \n threshold is {thr[1]}')
plotevent(sitonsquare((result1.reshape(10, 6796)[EvBTr - 10000] > thr[2]).cpu() & ((trvalgnnpppipi[EvBTr]) > 0)), 337, f'event number {EvBTr} passed through network after training \n threshold is {thr[2]}')
plotevent(sitonsquare((result1.reshape(10, 6796)[EvBTr - 10000] > thr[3]).cpu() & ((trvalgnnpppipi[EvBTr]) > 0)), 338, f'event number {EvBTr} passed through network after training \n threshold is {thr[3]}')
plotevent(sitonsquare((result1.reshape(10, 6796)[EvBTr - 10000] > thr[4]).cpu() & ((trvalgnnpppipi[EvBTr]) > 0)), 339, f'event number {EvBTr} passed through network after training \n threshold is {thr[4]}')

plt.savefig(f'/hpcfs/bes/mlgpu/hoseinkk/MLTracking/otherparticles/pppipi/{modelname}/results/{t}\
  Pass_event_{EvBTr}_before_and_after_training_with_{EpochNum}_epochs.png', bbox_inches='tight')
plt.show()

# second event
fig = plt.figure(figsize=(40, 15))
plt.rcParams['font.size'] = '15'
EvBTr = 10005
plotevent(sitonsquare(traingnnpppipi[EvBTr]), 331, f'event number {EvBTr} with noise \n color indicates time')
plotevent(sitonsquare(trvalgnnpppipi[EvBTr]), 332, f'event number {EvBTr} without noise \n color indicates time')
plotevent(sitonsquare(result0.reshape(10, 6796)[EvBTr - 10000]), 333, f'event number {EvBTr} passed through network before training \n color indicates time')
# outcome
plotevent(sitonsquare(result1.reshape(10, 6796)[EvBTr - 10000]), 334, f'event number {EvBTr} passed through network after training \n color indicates time')
# with threshold
plotevent(sitonsquare((result1.reshape(10, 6796)[EvBTr - 10000] > thr[0]).cpu() & ((trvalgnnpppipi[EvBTr]) > 0)), 335, f'event number {EvBTr} passed through network after training \n threshold is {thr[0]}')
plotevent(sitonsquare((result1.reshape(10, 6796)[EvBTr - 10000] > thr[1]).cpu() & ((trvalgnnpppipi[EvBTr]) > 0)), 336, f'event number {EvBTr} passed through network after training \n threshold is {thr[1]}')
plotevent(sitonsquare((result1.reshape(10, 6796)[EvBTr - 10000] > thr[2]).cpu() & ((trvalgnnpppipi[EvBTr]) > 0)), 337, f'event number {EvBTr} passed through network after training \n threshold is {thr[2]}')
plotevent(sitonsquare((result1.reshape(10, 6796)[EvBTr - 10000] > thr[3]).cpu() & ((trvalgnnpppipi[EvBTr]) > 0)), 338, f'event number {EvBTr} passed through network after training \n threshold is {thr[3]}')
plotevent(sitonsquare((result1.reshape(10, 6796)[EvBTr - 10000] > thr[4]).cpu() & ((trvalgnnpppipi[EvBTr]) > 0)), 339, f'event number {EvBTr} passed through network after training \n threshold is {thr[4]}')

plt.savefig(f'/hpcfs/bes/mlgpu/hoseinkk/MLTracking/otherparticles/pppipi/{modelname}/results/{t}\
  Pass_event_{EvBTr}_before_and_after_training_with_{EpochNum}_epochs.png', bbox_inches='tight')

# third event
fig = plt.figure(figsize=(40, 15))
plt.rcParams['font.size'] = '15'
EvBTr = 10007
plotevent(sitonsquare(traingnnpppipi[EvBTr]), 331, f'event number {EvBTr} with noise \n color indicates time')
plotevent(sitonsquare(trvalgnnpppipi[EvBTr]), 332, f'event number {EvBTr} without noise \n color indicates time')
plotevent(sitonsquare(result0.reshape(10, 6796)[EvBTr - 10000]), 333, f'event number {EvBTr} passed through network before training \n color indicates time')
# outcome
plotevent(sitonsquare(result1.reshape(10, 6796)[EvBTr - 10000]), 334, f'event number {EvBTr} passed through network after training \n color indicates time')
# with threshold
plotevent(sitonsquare((result1.reshape(10, 6796)[EvBTr - 10000] > thr[0]).cpu() & ((trvalgnnpppipi[EvBTr]) > 0)), 335, f'event number {EvBTr} passed through network after training \n threshold is {thr[0]}')
plotevent(sitonsquare((result1.reshape(10, 6796)[EvBTr - 10000] > thr[1]).cpu() & ((trvalgnnpppipi[EvBTr]) > 0)), 336, f'event number {EvBTr} passed through network after training \n threshold is {thr[1]}')
plotevent(sitonsquare((result1.reshape(10, 6796)[EvBTr - 10000] > thr[2]).cpu() & ((trvalgnnpppipi[EvBTr]) > 0)), 337, f'event number {EvBTr} passed through network after training \n threshold is {thr[2]}')
plotevent(sitonsquare((result1.reshape(10, 6796)[EvBTr - 10000] > thr[3]).cpu() & ((trvalgnnpppipi[EvBTr]) > 0)), 338, f'event number {EvBTr} passed through network after training \n threshold is {thr[3]}')
plotevent(sitonsquare((result1.reshape(10, 6796)[EvBTr - 10000] > thr[4]).cpu() & ((trvalgnnpppipi[EvBTr]) > 0)), 339, f'event number {EvBTr} passed through network after training \n threshold is {thr[4]}')

print(f'\nPassing event {EvBTr} from the network before training', 'input', TraTen[EvBTr], '\nresult1:', result1[EvBTr - 10000], '\nresult1.shape:', result1[EvBTr - 10000].shape)
print(f'\nPassing event {EvBTr} from the network after training', 'input', TraTen[EvBTr], '\nresult1:', result0[EvBTr - 10000], '\nresult1.shape:', result0[EvBTr - 10000].shape)
plt.savefig(f'/hpcfs/bes/mlgpu/hoseinkk/MLTracking/otherparticles/pppipi/{modelname}/results/{t}\
  Pass_event_{EvBTr}_before_and_after_training_with_{EpochNum}_epochs.png', bbox_inches='tight')
