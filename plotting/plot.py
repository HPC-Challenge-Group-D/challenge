#Use plotly to plot the exact solutions
from matplotlib import style
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import *
import numpy as np

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

plt.rcParams.update({'font.size': 16})



#DATA FOR PARALLEL PROGRAMMING#
#Dataset
#data_x = (1, 2, 4, 8, 16, 32, 64, 128)
#data_y = (659.5, 334.7, 169.0, 93.2, 52.3, 28.1, 14.6, 9.6)
#x_1 = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

#STRASSEN NON-PARALLEL
#y_1 = [0.00400e-3, 0.01e-3, 0.042e-3, 0.196e-3, 1.0e-3, 5.45e-3, 0.045, 0.323, 2.278, 16.08, 114.5]

#TRANSPOSE NON-PARALLEL
##y_1 = [0.283e-6, 1.109e-6, 4.605e-6, 0.021e-3, 0.0788e-3, 0.335e-3, 2.91e-3, 0.013, 0.0718, 0.421, 1.69, 9.06]


#STRASSEN Vectorized
#y_3_2 = [5e-6, 0.013e-3, 0.045e-3, 0.16e-3, 0.719e-3, 3.7e-3, 0.02566, 0.189, 1.42, 10.1, 71.03]

#STRASSEN PARALLEL
#y_3 = [0.016e-3, 0.029e-3, 0.08300e-3, 0.34700e-3, 1.765e-3, 8.61e-3, 0.047, 0.204, 1.08, 4.50, 21.8]

#STRASSEN FINAL
#y_4 = [0.019e-3, 0.036e-3, 0.09e-3, 0.292e-3, 1.21e-3, 6.35e-3, 0.029, 0.127, 0.788, 3.30, 15.87]

#TRANSPOSE PARALLEL
#y_3=[7.7e-6, 7.89e-6, 0.0117e-3, 0.042e-3, 0.153e-3, 0.419e-3, 0.742e-3, 1.67e-3, 6.51e-3, 0.034, 0.133, 0.74]

tcks = [1,2, 4,	8,	16,	32,	64,	128, 256]

data_x = [1,2,4,	8,	16,	32,	64,	96,	128,192]
data_y = [4123.69,	2293.73, 1167.00,	539.55,	276.82,	141.23,	74.28,	129.22,	96.25,	93.02]

efficiency_ticks = [2, 4,	8,	16,	32,	64,	128, 256]
eff_x = [2,4,	8,	16,	32,	64,	96,	128,192]
eff_y = [0.8989050959,	0.8833925961,	0.9553500792,	0.931023947,	0.9124327303,	0.8674343231,	0.3324239486,	0.3347093581,	0.2308856087]

gpu_x = [16,	32,	64,	128,	256,	512,	1024,	2048, 4096]
gpu_y = [0.00001623,	0.000014055,	0.00001444384615,	0.00001462792683,	0.00001431485356,	0.00002183947798,	0.00006364976096,	0.0004559446233, 0.00173978658]

#plt.loglog(x_1,y_1,'-o',label="Transpose seq")
#plt.plot(x_1,y_2,'-^',label="Strassen parallel")
#plt.loglog(x_1,y_3,'-d',label="Transpose par")
#plt.loglog(x_1,y_3_2,'-s',label="Strassen vec")
#plt.loglog(x_1,y_4,'-^',label="Strassen par + vec")
plt.loglog(gpu_x, gpu_y, "g-o", label="Runtime per iteration")
plt.legend(loc="best")
plt.xticks(ticks=gpu_x, labels=gpu_x)
plt.grid()
plt.xlabel("Gridsize $n$ x $n$")
plt.ylabel("$t/$iter $[s]$")
plt.tight_layout()
plt.savefig('plot.pdf')

plt.show()


#plt.bar(proc, balance_set)
#plt.xticks(ticks=proc,labels=proc)
#plt.xlabel("$rank$")

#plt.ylabel("$t [s]$")
#filename = "figures/" + "bar" + "." + "pdf"
#plt.savefig(filename, bbox_inches="tight")
#plt.show()

#=============================
#Parallel efficiency

#su = [data_y[0]/i for i in data_y[1:]]
#pe = [i/j for i,j in zip(su, data_x[1:])]


#plt.plot(data_x[1:],pe,'r-o')
#plt.xscale('log')
#plt.grid()
#plt.xticks(ticks=data_x[1:],labels=data_x[1:])
#plt.xlabel("$n_p$")
#plt.ylabel("$PE(n_p)$")

#filename = "figures/" + "pe" + "." + "pdf"
#plt.savefig(filename, bbox_inches="tight")
#plt.show()