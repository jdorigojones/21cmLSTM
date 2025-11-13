#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
import gc
import os
from tensorflow import keras
from tensorflow.keras import backend as K
import Global21cmLSTM as Global21cmLSTM
from Global21cmLSTM import __path__

#PATH = f"{os.environ.get('HOME')}/.Global21cmLSTM/"
PATH = '/projects/jodo2960/beam_weighted_foreground/'

class evaluate_foreground():
    def __init__(self, **kwargs):

        for key, values in kwargs.items():
            if key not in set(['model_path', 'model']):
                raise KeyError("Unexpected keyword argument in evaluate()")

        self.model_path = kwargs.pop('model_path', PATH+'models/')
        if type(self.model_path) is not str:
            raise TypeError("'model_path' must be a sting.")
        elif self.model_path.endswith('/') is False:
            raise KeyError("'model_path' must end with '/'.")

        self.train_mins = np.load(self.model_path+'train_mins_foreground_beam_meansub.npy')
        self.train_maxs = np.load(self.model_path+'train_maxs_foreground_beam_meansub.npy')

        emulator_foreground = Global21cmLSTM.emulator_foreground.Emulate() # initialize 21cmLSTM to emulate 21cmGEM data
        emulator_foreground.load_model()

        self.model = kwargs.pop('model', None)
        if self.model is None:
            self.model = keras.models.load_model(self.model_path+'emulator_foreground_beam_meansub_21cmLSTM_long.h5')#,compile=False)

    def __call__(self, parameters):

        spectra_foreground_emulated = emulator_foreground.predict(parameters)
        return spectra_foreground_emulated
        
        # # preprocess input physical parameters (same as in preprocess_foreground.py)
        # if len(np.shape(parameters)) == 1:
        #     parameters = np.expand_dims(parameters, axis=0) # if doing one signal at a time

        # nu_list = np.linspace(6,50,176)
        # vr = 1420.405751
        # z_list = (vr/nu_list) - 1
        # nu_list_norm = nu_list/np.max(nu_list) # list to be Min-Max normalized later
        # N_proc = np.shape(parameters)[0] # number of signals (i.e., parameter sets) to process
        # n = len(z_list) # number of frequency channels or redshift bins
        # p = np.shape(parameters)[1]+1 # number of input parameters for each LSTM cell/layer (# of physical params plus one for frequency step)
        # proc_params_format = np.zeros((N_proc,n,p))
        # proc_params = np.zeros_like(proc_params_format)

        # for i in range(N_proc):
        #     for j in range(n):
        #         proc_params_format[i, j, 0:18] = parameters[i,:]
        #         proc_params_format[i, j, 18] = nu_list_norm[j]

        # for i in range(p):
        #     x = proc_params_format[:,:,i]
        #     proc_params[:,:,i] = (x-self.train_mins[i])/(self.train_maxs[i]-self.train_mins[i])
            
        # result = self.model(proc_params, training=False).numpy() # evaluate trained instance of 21cmLSTM with processed parameters
        # evaluation = result.T[0].T
        # unproc_signals = evaluation.copy()
        # unproc_signals = (evaluation*(self.train_maxs[-1]-self.train_mins[-1]))+self.train_mins[-1] # unpreprocess (i.e., denormalize) signals
        # unproc_signals = np.squeeze(unproc_signals, axis=2)
        # if unproc_signals.shape[0] == 1:
        #     return unproc_signals[0,:]
        # else:
        #     return unproc_signals


