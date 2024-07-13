#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from cosmoLSTM import __path__

class evaluate():
    def __init__(self):
        self.model = keras.models.load_model('models/emulator.h5',compile=False)
        self.train_mins = np.loadtxt('models/train_mins.txt')
        self.train_maxs = np.loadtxt('models/train_maxs.txt')
        
    def __call__(self, parameters):

        if len(np.shape(parameters)) == 1:
            parameters = np.expand_dims(parameters, axis=0)
            
        unproc_f_s = parameters[:,0].copy() # f_star, controls the SFE
        unproc_V_c = parameters[:,1].copy() # minimum virial ciruclar velocity of star-forming halos
        unproc_f_X = parameters[:,2].copy() # efficiency of X-ray sources
        unproc_f_s = np.log10(unproc_f_s)
        unproc_V_c = np.log10(unproc_V_c)
        unproc_f_X[unproc_f_X == 0] = 1e-6 # set fX=0 values to 1e-6 before taking log10
        unproc_f_X = np.log10(unproc_f_X)
        parameters_log = np.empty(parameters.shape)
        parameters_log[:, 0] = unproc_f_s  # copy logfstar
        parameters_log[:, 1] = unproc_V_c  # copy logVc
        parameters_log[:, 2] = unproc_f_X  # copy logfX
        parameters_log[:, 3:] = parameters[:, 3:].copy()
        
        z_list = np.linspace(5, 50, 451) # list of redshifts for 21cmGEM signals; equiv to np.arange(5, 50.1, 0.1)
        vr = 1420.405751
        nu_list = [vr/(z+1) for z in z_list]
        nu_list = nu_list[::-1] # flip list of frequencies to be from high-z to low-z like done for the brightness temperatures
        nu_list_norm = nu_list/np.max(nu_list)
        N_proc = np.shape(parameters_log)[0] # number of parameter sets (i.e., signals) to process
        n = len(z_list) # number of redshift/frequency bins (i.e., signal resolution)
        p = np.shape(parameters)[1]+1 # number of input parameters for each LSTM cell/layer (i.e., # of physical params + 1 for z/nu step)
        proc_params_format = np.zeros((N_proc,n,p))
        proc_params = np.zeros_like(proc_params_format)
        
        for i in range(N_proc):
            for j in range(n):
                proc_params_format[i, j, 0:7] = parameters_log[i,:]
                proc_params_format[i, j, 7] = nu_list_norm[j]
        
        for i in range(p):
            x = proc_params_format[:,:,i]
            proc_params[:,:,i] = (x-self.train_mins[i])/(self.train_maxs[i]-self.train_mins[i])

        result = self.model(proc_params, training=False).numpy()
        evaluation = result.T[0] #evaluation = result[0].T[0]
        unproc_signals = evaluation.copy()
        unproc_signals = (evaluation*(self.train_maxs[-1]-self.train_mins[-1]))+self.train_mins[-1]
        unproc_signals = unproc_signals[-1::-1]
        return unproc_signals[:,0]


