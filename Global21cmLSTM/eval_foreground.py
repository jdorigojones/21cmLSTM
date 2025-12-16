#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
#import torch
#import gc
import os
from tensorflow import keras
from tensorflow.keras import backend as K
#import Global21cmLSTM as Global21cmLSTM
#import Global21cmLSTM.preprocess_foreground as pp
#from Global21cmLSTM import __path__

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)
#torch.set_default_dtype(torch.float64)

#PATH = f"{os.environ.get('HOME')}/.Global21cmLSTM/"

class evaluate_foreground():
    def __init__(self, **kwargs):
        for key, values in kwargs.items():
            if key not in set(['model_path', 'model']):
                raise KeyError("Unexpected keyword argument in evaluate()")

        # Default model path
        default_model_path = f"/projects/jodo2960/beam_weighted_foreground/models/emulator_foreground_beam_10regions_meansub_21cmLSTM_3layer.h5"
        model_path = kwargs.pop('model_path', default_model_path)

        # Load normalization data from the same directory as the model
        model_dir = os.path.dirname(model_path) + '/'
        self.train_mins = np.load(model_dir + 'train_mins_foreground_beam_meansub_10regions_LSTM.npy')
        self.train_maxs = np.load(model_dir + 'train_maxs_foreground_beam_meansub_10regions_LSTM.npy')

        self.model = kwargs.pop('model', None)
        if self.model is None:
            self.model = keras.models.load_model(model_path,compile=False)
            self.model.trainable = False
            # Warm up the model (first call is slow)
            dummy_params = np.zeros((1, 33))
            _ = self(dummy_params)
            print("Model loaded and warmed up!")
            
    def __call__(self, parameters):
        if len(np.shape(parameters)) == 1:
            parameters = np.expand_dims(parameters, axis=0) # if doing one signal at a time
            
        nu_list = np.linspace(5,50,180)
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
                proc_params_format[i, j, 0:33] = parameters[i,:]
                proc_params_format[i, j, 33] = nu_list_norm[j]
        
        for i in range(p):
            x = proc_params_format[:,:,i]
            proc_params[:,:,i] = (x-self.train_mins[i])/(self.train_maxs[i]-self.train_mins[i])

        parameters = 0
        result = self.model(proc_params, training=False).numpy() # evaluate trained instance of 21cmLSTM with processed parameters
        emulated_spectra = result.copy()
        emulated_spectra = (result*(self.train_maxs[-1]-self.train_mins[-1]))+self.train_mins[-1] # unpreprocess (i.e., denormalize) signals
        emulated_spectra = np.squeeze(emulated_spectra, axis=2)
        if emulated_spectra.shape[0] == 1:
            return emulated_spectra[0, :]
        else:
            return emulated_spectra

