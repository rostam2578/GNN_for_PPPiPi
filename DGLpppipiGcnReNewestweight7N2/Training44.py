print('\n\n\n\n', 'Training ...', '\n\n')
#from GraphDef import *
#from DataLoad import *
#from cProfile import label
from Model import *

# Training
import matplotlib.pyplot as plt
import os
from os import path
import datetime
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

log_dir = "/hpcfs/bes/mlgpu/hoseinkk/MLTracking/otherparticles/pppipi/" + modelname #+ time.time()
checkpoint_dir_path = "/hpcfs/bes/mlgpu/hoseinkk/MLTracking/otherparticles/pppipi/" + modelname 
if not os.path.exists(checkpoint_dir_path):
  os.mkdir(checkpoint_dir_path)
def checkpoint_save(state, epoch):
  print("=> saveing checkpoint at epoch", epoch)
  torch.save(state, F"{checkpoint_dir_path}/saved_checkpoint.pth.tar")
def checkpoint_load(loadedcheckpoint):
  print("=> loading checkpoint from", F"{checkpoint_dir_path}/saved_checkpoint.pth.tar")
  net.load_state_dict(loadedcheckpoint['state_dict'])
  #optimizer.load_state_dict(loadedcheckpoint['optimizer'])

TraEvN = 1998 #7801 * 2
EpochNum = 30
epoch_save = 5
pointnum = 100    # number of points for eff-pur plot
prlosbatch = 10   # print mean loss of every particular number of batches.
lossarray = np.zeros(shape=EpochNum)
evallossarray = np.zeros(shape=EpochNum)
load_model = False
if load_model:
  checkpoint_load(torch.load(F"{checkpoint_dir_path}/saved_checkpoint.pth.tar"))

# form a phase space to explore the hyperparameters in training.
hpmesh = []
for BatchSize in 5 * np.array([1, 3, 6, 10, 14, 19]): #[39, 117, 234, 390, 546, 741]
    for LrVal in np.array([0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]):
        for weight_decay_val in np.array([0, 0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]): #np.arange(0.001, 0.01, 0.009):
            hpmesh.append([BatchSize, LrVal, weight_decay_val])
hpmesh = np.array(hpmesh)
startmesh = 44
endmesh = 45

t1 = datetime.datetime.now()
for BatchSize, LrVal, weight_decay_val in hpmesh[startmesh:endmesh]:
  BatchSize = int(BatchSize)
  net = GCN(1, 1, BatchSize).to(device)
  print("\n\n\nload_model", load_model, "\nTraEvN", TraEvN, "\nBatchSize", BatchSize, \
  "\nEpochNum", EpochNum, "\nepoch_save", epoch_save, "\nLrVal", LrVal, '\nweight_decay', weight_decay_val, '\n\n\n')
  batcharray = np.arange(0 * TraEvN, 1 * TraEvN, BatchSize)
  evalbatchnum = int((TraEvN // BatchSize) * 0.06) + 1
  evalbatcharray = np.random.choice(batcharray, evalbatchnum)
  purity = np.zeros(shape = (1 * TraEvN, pointnum))
  efficiency = np.zeros(shape = (1 * TraEvN, pointnum))
  eval_purity = np.zeros(shape = (1 * TraEvN, pointnum))
  eval_efficiency = np.zeros(shape = (1 * TraEvN, pointnum))
  loss_function = torch.nn.CrossEntropyLoss() #CrossEntropyLoss() #BCELoss() #Define loss criterion.
  optimizer = torch.optim.Adam(net.parameters(), lr=LrVal, weight_decay=weight_decay_val)  # Define optimizer.
  print('\n\n\noptimizer.param_groups', optimizer.param_groups)
  optimizer.add_param_group({'params': dglgraph.edata['efet']})
  print('\n\n\noptimizer.param_groups', optimizer.param_groups)
  for epoch in range(EpochNum):
      ep_loss = 0
      b_loss = 0
      eval_loss = 0
      purefindex = 0
      # training
      for i in batcharray:
        optimizer.zero_grad()  # Clear gradients.
        graphlist = [dglgraph] * BatchSize
        batcheddglgraph = dgl.batch(graphlist)
        featbatch = TraTen[i : i + BatchSize].reshape(BatchSize * 6796, 1)
        outi = net(batcheddglgraph, featbatch).reshape(BatchSize, 6796)#.type(torch.LongTensor)  # Perform a single forward pass.
        truevaluebatch = (TrvTen[i : i + BatchSize] > 0).type(torch.float)#.to(device)
        loss = loss_function(outi, truevaluebatch)  # Compute the loss solely based on the training nodes.
        # loss
        if i not in evalbatcharray:
          loss.backward()   # Derive gradients.
          optimizer.step()  # Update parameters based on gradients.
          ep_loss += loss
          b_loss += loss
          if i % (prlosbatch * BatchSize) == 0:   #print the loss function after some batches.
            print('epoch:', epoch, 'batch', i / BatchSize, 'event:', i, 'loss:', b_loss/prlosbatch)
            b_loss = 0
        else:
          eval_loss += loss
        # purity & efficiency
        if epoch == EpochNum - 1:
          for j in range(0, pointnum):
            thrres = outi > (0 + j * 0.01)
            datnoi = featbatch.reshape  (BatchSize, 6796) > 0 #torch.Tensor.cpu(TraTen[i].reshape(6796)).numpy() > 0
            datnon = truevaluebatch > 0   #torch.Tensor.cpu(TrvTen[i].reshape(6796)).numpy() > 0
            outres = thrres & datnoi 
            comres = outres & datnon
            if i in evalbatcharray:
              eval_purity[i, j] = comres.sum() / outres.sum()
              eval_efficiency[i, j] = comres.sum() / datnon.sum()
            else:
              purity[i, j] = comres.sum() / outres.sum()
              efficiency[i, j] = comres.sum() / datnon.sum()

      lossarray[epoch] = ep_loss/(TraEvN//(BatchSize))
      print('time passed so far:\n', datetime.datetime.now() - t1)
      evallossarray[epoch] = eval_loss/evalbatchnum
      print('evaluation loss:', evallossarray[epoch])
      print('epoch:', epoch, 'mean loss:', lossarray[epoch])

      # saving
      if (epoch % epoch_save == 0) or epoch == EpochNum - 1:
        checkpoint = {'state_dict' : net.state_dict(), 'optimizer' : optimizer.state_dict()}
        checkpoint_save(checkpoint, epoch)
        print('checkpoint is saved at:', checkpoint_dir_path)
      writer.add_scalars(f'loss for {modelname}, Batchsize {BatchSize}, {TraEvN} training events, {evalbatchnum * BatchSize} evaluation events', \
        {f'training (Lr and weight_decay={LrVal},{weight_decay_val})': lossarray[epoch], f'evaluation (Lr and weight_decay={LrVal},{weight_decay_val})': evallossarray[epoch]}, epoch)

      for name, weight in net.named_parameters():
        if name != 'efeatur':
          writer.add_histogram(name, weight, epoch)
          writer.add_histogram(f'{name}.grad', weight.grad, epoch)
      
  purres = purity.sum(axis = 0) / (purity!=0).sum(axis = 0)
  effres = efficiency.sum(axis = 0) / (efficiency!=0).sum(axis = 0)
  eval_purres = eval_purity.sum(axis = 0) / (eval_purity!=0).sum(axis = 0)
  eval_effres = eval_efficiency.sum(axis = 0) / (eval_efficiency!=0).sum(axis = 0)
  best_thrd = np.where(abs(effres - purres) == np.amin(abs(effres - purres)))[0]
  best_eval_thrd = np.where(abs(eval_effres - eval_purres) == np.amin(abs(eval_effres - eval_purres)))[0]
  best_effpur = effres[best_thrd]
  if ((effres - purres) > 0).sum() in [0, len(batcharray) - len(evalbatcharray)]:
    best_effpur = 0  # to make sure the lines intersect
  best_eval_effpur = eval_effres[best_eval_thrd]
  if ((eval_effres - eval_purres) > 0).sum() in [0, len(evalbatcharray)]:
    best_eval_effpur = 0

  writer.add_hparams({'Lr': LrVal, 'weight_decay': weight_decay_val, 'BatchSize':BatchSize},\
             {'evaluation loss': evallossarray[EpochNum - 1], 'training loss': lossarray[EpochNum - 1], \
              'best_eval_effpur': best_eval_effpur, 'best_effpur': best_effpur})
  for j in range(pointnum):
    writer.add_scalars(f'eff-pur for Batchsize {BatchSize}', \
              {f'efficiency (Lr and weight_decay={LrVal},{weight_decay_val})':effres[j], f'purity (Lr and weight_decay={LrVal},{weight_decay_val})':purres[j], \
                f'eval_efficiency (Lr and weight_decay={LrVal},{weight_decay_val})':eval_effres[j], f'eval_purity (Lr and weight_decay={LrVal},{weight_decay_val})':eval_purres[j]}, j)   
#add_hparams(hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None)
#add_pr_curve(tag, labels, predictions, global_step=None, num_thresholds=127, weights=None, walltime=None)
writer.flush()

print('\n\n\ntraining loss:\n', lossarray, '\n\n\evaluation loss:\n', evallossarray)
plt.figure(figsize=(25, 20))
plt.rcParams['font.size'] = '18'
plt.scatter(np.arange(EpochNum), lossarray, label='training loss')
plt.scatter(np.arange(EpochNum), evallossarray, label='validation loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')  
plt.title(f'Loss function for the last {EpochNum} epochs, lr={LrVal}, weight_decay={weight_decay_val}')
t = datetime.datetime.now()
plt.savefig(f'/hpcfs/bes/mlgpu/hoseinkk/MLTracking/otherparticles/pppipi/{modelname}/results/{t}\
  Loss function for the last {EpochNum} epochs.png', bbox_inches='tight')

print('\n\n\neval_efficiency:\n', eval_effres, '\n\n\neval_purity:\n', eval_purres)
plt.figure(figsize=(25, 20))
plt.rcParams['font.size'] = '18'
plt.scatter(np.arange(pointnum) / 100, effres, label='efficiency') #/ 100
plt.scatter(np.arange(pointnum) / 100, purres, label='purity')
plt.scatter(np.arange(pointnum) / 100, eval_effres, label='eval_efficiency')
plt.scatter(np.arange(pointnum) / 100, eval_purres, label='eval_purity')
plt.legend()
plt.xlabel('threshold')
plt.ylabel('eff-pur')
plt.title(f'eff-pur for the last {EpochNum} epochs, lr={LrVal}, weight_decay={weight_decay_val}')
plt.savefig(f'/hpcfs/bes/mlgpu/hoseinkk/MLTracking/otherparticles/pppipi/{modelname}/results/{t}\
  eff-pur for the last {EpochNum} epochs.png', bbox_inches='tight')

print('\n\n\ntotal time:', datetime.datetime.now() - t1, f'hpmesh elements: {startmesh} to {endmesh}')
