#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
import gc
import os
from tensorflow import keras
from tensorflow.keras import backend as K
from Global21cmLSTM import __path__

#PATH = f"{os.environ.get('HOME')}/.Global21cmLSTM/"
PATH = '/projects/jodo2960/beam_weighted_foreground/'

class evaluate_foreground():
    def __init__(self, **kwargs):

        for key, values in kwargs.items():
            if key not in set(['model_dir', 'model']):
                raise KeyError("Unexpected keyword argument in evaluate()")

        self.model_dir = kwargs.pop('model_dir', PATH+'models/')
        if type(self.model_dir) is not str:
            raise TypeError("'model_dir' must be a sting.")
        elif self.model_dir.endswith('/') is False:
            raise KeyError("'model_dir' must end with '/'.")

        self.train_mins = np.loadtxt(self.model_dir+'train_mins_foreground_beam_meansub.npy')
        self.train_maxs = np.loadtxt(self.model_dir+'train_maxs_foreground_beam_meansub.npy')

        self.model = kwargs.pop('model', None)
        if self.model is None:
            self.model = keras.models.load_model(self.model_dir+'emulator_foreground_beam_meansub_21cmLSTM.h5',compile=False)

    def __call__(self, parameters):

        # preprocess input physical parameters (same as in preprocess_foreground.py)
        if len(np.shape(parameters)) == 1:
            parameters = np.expand_dims(parameters, axis=0) # if doing one signal at a time

        nu_list = np.linspace(6,50,176)
        vr = 1420.405751
        z_list = (vr/nu_list) - 1
        nu_list_norm = nu_list/np.max(nu_list) # list to be Min-Max normalized later
        N_proc = np.shape(parameters)[0] # number of signals (i.e., parameter sets) to process
        n = len(z_list) # number of frequency channels or redshift bins
        p = np.shape(parameters)[1]+1 # number of input parameters for each LSTM cell/layer (# of physical params plus one for frequency step)
        proc_params_format = np.zeros((N_proc,n,p))
        proc_params = np.zeros_like(proc_params_format)

        for i in range(N_proc):
            for j in range(n):
                proc_params_format[i, j, 0:18] = parameters[i,:]
                proc_params_format[i, j, 18] = nu_list_norm[j]

        for i in range(p):
            x = proc_params_format[:,:,i]
            proc_params[:,:,i] = (x-self.train_mins[i])/(self.train_maxs[i]-self.train_mins[i])

        result = self.model(proc_params, training=False).numpy() # evaluate trained instance of 21cmLSTM with processed parameters
        evaluation = result.T[0].T
        unproc_signals = evaluation.copy()
        unproc_signals = (evaluation*(self.train_maxs[-1]-self.train_mins[-1]))+self.train_mins[-1] # unpreprocess (i.e., denormalize) signals
        if unproc_signals.shape[0] == 1:
            return unproc_signals[0,:]
        else:
            return unproc_signals


