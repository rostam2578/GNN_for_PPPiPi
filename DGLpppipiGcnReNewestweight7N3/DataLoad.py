print('\n\n\n\n\n', 'Loading data ...', '\n\n')

import cupy as cp
import numpy as np
import torch
device = torch.device('cuda')

#########################################################
# Upload from local drive
traingnnpppipi = cp.load('/hpcfs/bes/mlgpu/hoseinkk/MLTracking/otherparticles/pppipi/files/traingnnpppipi.npy')
trvalgnnpppipi = cp.load('/hpcfs/bes/mlgpu/hoseinkk/MLTracking/otherparticles/pppipi/files/trvalgnnpppipi.npy')
print('shape', trvalgnnpppipi.shape, traingnnpppipi.shape)
print('sum', (trvalgnnpppipi > 0).sum(), (traingnnpppipi > 0).sum())

#np.random.shuffle(traingnn80)
#np.random.shuffle(trvalgnn80)

TraTen = torch.tensor(traingnnpppipi, dtype=torch.float).to(device)
TrvTen = torch.tensor(trvalgnnpppipi, dtype=torch.float).to(device)
print('shape', TraTen.shape, TraTen.shape)

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