print('\n\n\n\n\n', 'Loading data ...', '\n\n')

import cupy as cp
import numpy as np
import torch
import pandas as pd
import os
from numpy.lib import recfunctions as rfn
import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#########################################################
# Upload from local drive
traingnnpppipi = np.load('/hpcfs/bes/mlgpu/hoseinkk/MLTracking/bhabha/bhabha-files/traingnn80.npy')
trvalgnnpppipi = np.load('/hpcfs/bes/mlgpu/hoseinkk/MLTracking/bhabha/bhabha-files/trvalgnn80.npy')

print('\ntraining set shape', traingnnpppipi.shape, '\nsum', (traingnnpppipi > 0).sum())
print('\ntarget set shape', trvalgnnpppipi.shape, '\nsum', (trvalgnnpppipi > 0).sum())

#np.random.shuffle(traingnn80)
#np.random.shuffle(trvalgnn80)

TraTen = torch.tensor(traingnnpppipi, dtype=torch.float).to(device)
TrvTen = torch.tensor(trvalgnnpppipi, dtype=torch.float).to(device)

print('\nTraTen and TrvTen shape:\n', TraTen.shape, TrvTen.shape)

#########################################################
# Wires at each layer
wires = np.array([40, 44, 48, 56, 64, 72, 80, 80, 76, 76, 88, 88, 100, 100, 112, 112, 128, 128, 140, 140, \
                 160, 160, 160, 160, 176, 176, 176, 176, 208, 208, 208, 208, 240, 240, 240, 240, \
                  256, 256, 256, 256, 288, 288, 288])
#wiresum = cumsum(wires)
# The diameter of the wires are about 12 mm. First layer is in 71mm radius of the center.
# 1st tube radius: about 70mm to 170mm
# 1st gap: 170mm to 189mm, 2nd gap: 385mm to 392mm, 3rd gap: about 652mm to 658mm
# radius of the last layer: about 760mm 

# prepare events on rectangle to plot
sqevent = np.zeros(shape=(43, 288))
wiresum = np.zeros(shape = 44, dtype=int)
def sitonsquare(event):
    for i in range(43):
        wiresum[i + 1]= np.cumsum(wires)[i]
        w = int(wires[i] / 2)
        for j in range(-w, w):
            sqevent[i, j + 144] = event[(wiresum[i]) + j + w]
    return sqevent 

fig = plt.figure(figsize=(15, 33))
plt.rcParams['font.size'] = '13'
axnum = 1
for evt in (10, 100, 230, 54, 812):
    ax = fig.add_subplot(5, 2, axnum)
    fig.colorbar(ax.matshow(sitonsquare(traingnnpppipi[evt]), aspect=5, vmin=0, vmax=1, extent=[0, 288, 0, 43], origin='lower')\
                ,fraction=0.034)
    plt.title(f'\nevent number {evt} with noise \n color indicates time')
    plt.xlabel('cell')
    plt.ylabel('layer')
    plt.xlim(0, 288)
    plt.ylim(0, 43)
    ax = fig.add_subplot(5, 2, axnum + 1)
    fig.colorbar(ax.matshow(sitonsquare(trvalgnnpppipi[evt]), aspect=5, vmin=0, vmax=1, extent=[0, 288, 0, 43], origin='lower')\
                ,fraction=0.034)
    plt.xlabel('cell')
    plt.ylabel('layer')
    plt.xlim(0, 288)
    plt.ylim(0, 43)
    plt.title(f'\nevent number {evt} without noise \n color indicates time')
    axnum = axnum + 2

t = datetime.datetime.now()
plt.savefig(f'/hpcfs/bes/mlgpu/hoseinkk/MLTracking/otherparticles/pppipi/DGLBhaGcnReNewestweight7N4/results/{t}\
  Examples_from_data_set.png', bbox_inches='tight')