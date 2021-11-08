#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal 
from matplotlib.ticker import NullFormatter
import sys
import Image 


mol = sys.argv[1]
geom = sys.argv[2]

image = False
exp_data = True
scaling_factor=0.965
broaden=2 # in cm-1, Gaussian, change in the code to L. 


xmin = {}; xmax = {}
eV2kcal = 23.0605

#Load data from geom:
IR = [] ; Int = []
for line in file(geom+'/input.log','r').readlines():
    if "Frequencies" in line:
        for r in range(2,5):
            IR.append(float(line.split()[r]))
    if "IR Int" in line:
        for r in range(3,6):
            Int.append(float(line.split()[r]))

data = np.zeros( (len(IR), 2) )
for n, ir, i  in zip(range(len(IR)), IR, Int): 
     data[n,0] = ir; data[n,1]=i

if exp_data == True:
  if mol == 'bgh1_na': 
      EXP = np.genfromtxt('/home/marianski/Desktop/Log/Sugars/trisacharide/experimental/'+mol+'.dat')
      EXP2 = None 
  elif mol == 'bgh2_H':
      EXP = np.genfromtxt('/home/marianski/Desktop/Log/Sugars/trisacharide/experimental/'+mol+'.dat')
      EXP2 = None
  elif mol == 'lewis_x_H':
      EXP = np.genfromtxt('/home/marianski/Desktop/Log/Sugars/trisacharide/experimental/'+mol+'.dat')
      EXP2 = None
  else:
      EXP = np.genfromtxt('/home/marianski/Desktop/Log/Sugars/trisacharide/experimental/'+mol+'_1.dat')
      EXP2 = np.genfromtxt('/home/marianski/Desktop/Log/Sugars/trisacharide/experimental/'+mol+'_2.dat')
  scale_exp=  1/np.amax(EXP[:,1])


def gaussian(X,x0,s):
    return np.exp(-0.5*((X-x0)/s)**2)

def lorentz(X,x0,s):
    return 1/(np.pi*s*(1+((X-x0)/s)**2))

Ytot = np.zeros((4001,))
X=np.linspace(0, 4000,4001)
Y=np.zeros((4001,))
for l in range(data.shape[0]):
    Y = data[l,1]*gaussian(X, data[l,0], broaden)
    Ytot += Y

fig = plt.figure(1, figsize=(16,4))

left, width = 0.02, 0.98
bottom, height = 0.1, 0.8
ax  = [left, bottom, width, height ]
ax  = plt.axes(ax)
xmin=800  ; xmax = 1800

#Calculate scaling factors: 
scale_t  =  1/np.amax(Ytot[xmin:xmax+100])


ax.tick_params(axis='both', which='both', bottom='on', top='off', labelbottom='on', right='off', left='off', labelleft='off')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.xaxis.set_tick_params(direction='out')
ax.yaxis.set_major_formatter(NullFormatter())
ax.set_ylim(0,2)

xticks = np.linspace(xmin,xmax,int((xmax-xmin)/100)+1)
ax.set_xticks(xticks[:-1])
ax.set_xticklabels([int(x) for x in xticks[:-1]], fontsize=20)
ax.set_xlim(xmin, xmax)
for t in xticks:
    ax.plot([t,t],[0,3], 'k--')

shift=1
Xsc = X*scaling_factor
Ysc = Ytot*scale_t
ir_theo = ax.plot(Xsc, -Ysc+shift, color='0.25', linewidth=2)
ax.fill_between(Xsc, np.linspace(shift, shift, len(Ysc)), -Ysc+1, color='0.5', alpha=0.5)
ax.plot([xmin,xmax], [shift, shift], 'k', lw=2)

#axHigh.text(1650, 1.0, 'R$_F$='+str(Pendry_high)+'\n$\Delta$='+str(Delta_high), size=14)

if exp_data == True:
  if EXP2 != None: 
      ax.plot(EXP2[:,0], EXP2[:,1]*scale_exp+shift, color='r', alpha=0.25, linewidth=2)
      ax.fill_between(EXP2[:,0], EXP2[:,1]*scale_exp+shift, np.linspace(shift,shift, len(EXP2[:,1])), color='r', alpha=0.25)
  ax.plot(EXP[:,0], EXP[:,1]*scale_exp+shift, color='r', alpha=0.5, linewidth=2)
  ax.fill_between(EXP[:,0], EXP[:,1]*scale_exp+shift, np.linspace(shift,shift, len(EXP[:,1])), color='r', alpha=0.5)
  

for l in range(data.shape[0]):
    ax.plot([scaling_factor*data[l,0], scaling_factor*data[l,0]], [shift, -data[l,1]*scale_t+shift], linewidth=2, color='0.25')

#axLow.set_title(title)
#fig.tight_layout()
plt.savefig('ir_'+str(geom)+'.png', dpi=300)
#plt.savefig('ir_'+str(geom)+'.pdf', dpi=300)

