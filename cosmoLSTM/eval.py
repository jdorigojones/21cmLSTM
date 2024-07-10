#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import h5py
import tensorflow as tf
from tensorflow import keras
import numpy as np
from cosmoLSTM import __path__
import cosmoLSTM.preprocess as pp

z_list = np.linspace(5, 50, 451) # list of redshifts for 21cmGEM signals; equiv to np.arange(5, 50.1, 0.1)
data = 'dataset_21cmLSTM.h5'
with h5py.File(data, "r") as f:
    print("Keys: %s" % f.keys())
    train_params = np.asarray(f['par_train'])[()]
    train_signals = np.asarray(f['signal_train'])[()]
f.close()

class evaluate():
    def __init__(self, emulator):
        self.emulator = emulator
        
    def __call__(self, parameters):

        if len(np.shape(unproc_params)) == 1:
            unproc_params = np.expand_dims(unproc_params, axis=0)
            
        unproc_f_s = unproc_params[:,0].copy() # f_star, controls the SFE
        unproc_V_c = unproc_params[:,1].copy() # minimum virial ciruclar velocity of star-forming halos
        unproc_f_X = unproc_params[:,2].copy() # efficiency of X-ray sources
        train_f_s = train_params[:,0].copy() # need to make copies because the lists are linked if input arrays are the same
        train_V_c = train_params[:,1].copy()
        train_f_X = train_params[:,2].copy()
        unproc_f_s = np.log10(unproc_f_s)
        unproc_V_c = np.log10(unproc_V_c)
        train_f_s = np.log10(train_f_s)
        train_V_c = np.log10(train_V_c)
        unproc_f_X[unproc_f_X == 0] = 1e-6 # set fX=0 values to 1e-6 before taking log10
        train_f_X[train_f_X == 0] = 1e-6
        unproc_f_X = np.log10(unproc_f_X)
        train_f_X = np.log10(train_f_X)
        unproc_params_log = np.empty(unproc_params.shape)
        unproc_params_log[:, 0] = unproc_f_s  # copy logfstar
        unproc_params_log[:, 1] = unproc_V_c  # copy logVc
        unproc_params_log[:, 2] = unproc_f_X  # copy logfX
        unproc_params_log[:, 3:] = unproc_params[:, 3:].copy()
        train_params_log = np.empty(train_params.shape)
        train_params_log[:, 0] = train_f_s
        train_params_log[:, 1] = train_V_c
        train_params_log[:, 2] = train_f_X
        train_params_log[:, 3:] = train_params[:, 3:].copy()
        z_list = np.linspace(5, 50, 451) # list of redshifts for 21cmGEM signals; equiv to np.arange(5, 50.1, 0.1)
        vr = 1420.405751
        nu_list = [vr/(z+1) for z in z_list]
        nu_list = nu_list[::-1] # flip list of frequencies to be from high-z to low-z like done for the brightness temperatures
        nu_list_norm = nu_list/np.max(nu_list)
        N_train = np.shape(train_params_log)[0] # number of parameter sets (i.e., signals) in training set
        N_proc = np.shape(unproc_params_log)[0] # number of parameter sets (i.e., signals) to process
        n = len(z_list) # number of redshift/frequency bins (i.e., signal resolution)
        p = np.shape(unproc_params)[1]+1 # number of input parameters for each LSTM cell/layer (i.e., # of physical params + 1 for z/nu step)
        train_params_format = np.zeros((N_train,n,p))
        proc_params_format = np.zeros((N_proc,n,p))
        proc_params = np.zeros_like(proc_params_format)
        for i in range(N_train):
            for j in range(n):
                train_params_format[i, j, 0:7] = train_params_log[i,:]
                train_params_format[i, j, 7] = nu_list_norm[j]
                
        for i in range(N_proc):
            for j in range(n):
                proc_params_format[i, j, 0:7] = unproc_params_log[i,:]
                proc_params_format[i, j, 7] = nu_list_norm[j]
        
        for i in range(p):
            x = proc_params_format[:,:,i]
            proc_params[:,:,i] = (x-np.min(train_params_format[:,:,i]))/(np.max(train_params_format[:,:,i])-np.min(train_params_format[:,:,i]))

        proc_emulated_signals = self.emulator.predict(proc_params)

        unproc_signals = proc_emulated_signals.copy()
        unproc_signals = (proc_emulated_signals*(np.max(train_signals)-np.min(train_signals)))+np.min(train_signals)
        emulated_signals = unproc_signals[:,::-1]
                
        if emulated_signals.shape[0] == 1:
            return emulated_signals[0, :]
        else:
            return emulated_signals


