#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
spnet.py: python3 version of Izhikevich's spnet.m, a spiking network with 
axonal conduction delays and STDP; you can find spnet.m at: 
https://www.izhikevich.org/publications/spnet.m
"""
import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(1)
M=200       # number of synapses per neuron
D=20        # maximal conduction delay
Ne=800      # excitatory neurons 
Ni=200      # inhibitory neurons              
N=Ne+Ni     # total neurons
a=np.concatenate((0.02*np.ones((Ne,1)), 0.1*np.ones((Ni,1))))
d=np.concatenate((8*np.ones((Ne,1)), 2*np.ones((Ni,1))))
sm=10       # maximal synaptic strength

# post=ceil([N*rand(Ne,M)Ne*rand(Ni,M)]) 
# Take special care not to have multiple connections between neurons
delays = np.empty((N,D), dtype=np.object)
delays[...] = [[[] for j in range(D)] for i in range(N)] 
post=np.zeros((N,M),dtype=np.int64)
for i in range(Ne):
    p=np.random.permutation(N)
    post[i,:]=p[:M]
    for j in range(M):
        delays[i, random.randint(0,D-1)].append(j)  # Assign random exc delays

for i in range(Ne, N):
    p=np.random.permutation(Ne)
    post[i,:]=p[:M]
    delays[i, 0]=np.arange(M)            # we just set all inh delays to 1 ms.

s=np.concatenate((6*np.ones((Ne,M)), -5*np.ones((Ni,M))))  # synaptic weights
sd=np.zeros((N,M))                                         # their derivatives

# Make links at postsynaptic targets to the presynaptic weights
pre = np.empty((N,1), dtype=np.object)
pre[...] = [[[] for j in range(1)] for i in range(N)] 
aux = np.empty((N,1), dtype=np.object)
aux[...] = [[[] for j in range(1)] for i in range(N)] 
for i in range(Ne):
    for j in range(D):
        for k in range(len(delays[i,j])):
            pre[post[i, delays[i,j][k]],0].append(N*(delays[i,j][k])+i)
            # take delay into account:
            aux[post[i, delays[i,j][k]],0].append(N*(D-j)+i) 

STDP = np.zeros((N,1001+D))
v = -65*np.ones((N,1))                  # initial values
u = 0.2*v                               # initial values
firings=np.array([[-D, -1]])            # spike timings

for sec in range(60*60*24):             # simulation of 1 day
    for t in range(1000):               # simulation of 1 sec
        I=np.zeros((N,1))
        I[random.randint(0,N-1)]=20     # random thalamic input 
        fired = np.where(v>=30)         # indices of fired neurons
        v[fired]=-65  
        u[fired]=u[fired]+d[fired]
        STDP[fired[0],t+D]=0.1
        for k in range(len(fired[0])):
            sd[np.unravel_index(pre[fired[0][k]][0],sd.shape,'F')]\
                +=STDP[np.unravel_index([x+N*t for x in aux[fired[0][k]][0]],\
                                              STDP.shape,'F')]
        
        new_firings = np.concatenate((t*np.ones((fired[0].shape[0],1),\
                                                dtype=np.int64),\
                                      fired[0].reshape((len(fired[0]),1))),\
                                      axis=1)
        firings=np.concatenate((firings, new_firings))
        k=firings.shape[0]
        while firings[k-1,0]>t-D:
            de=delays[firings[k-1,1],t-firings[k-1,0]]
            ind = post[firings[k-1,1],de]
            I[ind]+=np.reshape(s[firings[k-1,1], de],(ind.shape[0],1))
            sd[firings[k-1,1],de]-=1.2*STDP[ind,t+D] 
            k-=1
    
        v=v+0.5*((0.04*v+5)*v+140-u+I)    # for numerical 
        v=v+0.5*((0.04*v+5)*v+140-u+I)    # stability time 
        u=u+a*(0.2*v-u)                   # step is 0.5 ms
        STDP[:,t+D+1]=0.95*STDP[:,t+D]    # tau = 20 ms
  
    plt.plot(firings[:,0],firings[:,1],'.')
    plt.axis([0, 1000, 0, N])
    plt.title("Network's Spiking Activity")
    plt.xlabel("Time (ms)")
    plt.ylabel("Indices of Neurons")
    plt.show()
    
    STDP[:,:D]=STDP[:,1001:]
    ind = np.where(firings[:,0] > 1000-D)
    f1=np.reshape(firings[ind,0]-1000,(firings[ind,0].shape[1],1))
    f2=np.reshape(firings[ind,1],(firings[ind,1].shape[1],1))
    holdover_firings = np.concatenate((f1,f2), axis=1)
    firings=np.concatenate(([[-D, -1]], holdover_firings))
    s[:Ne,:]=np.maximum(0,np.minimum(sm,0.01+s[:Ne,:]+sd[:Ne,:]))
    sd=0.9*sd
