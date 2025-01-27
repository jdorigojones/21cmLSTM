import numpy as np
import tensorflow as tf
import gc
import os
from tensorflow import keras
from tensorflow.keras import backend as K
from Global21cmLSTM import __path__

PATH = f"{os.environ.get('AUX_DIR', os.environ.get('HOME'))}/.Global21cmLSTM/"

class evaluate_21cmGEM():
    def __init__(self, **kwargs):
        
        for key, values in kwargs.items():
            if key not in set(['model_dir', 'model']):
                raise KeyError("Unexpected keyword argument in evaluate()")
                
        self.model_dir = kwargs.pop('model_dir', PATH+'models/')
        if type(self.model_dir) is not str:
            raise TypeError("'model_dir' must be a sting.")
        elif self.model_dir.endswith('/') is False:
            raise KeyError("'model_dir' must end with '/'.")
        
        self.train_mins = np.loadtxt(self.model_dir+'train_mins_21cmGEM.txt')
        self.train_maxs = np.loadtxt(self.model_dir+'train_maxs_21cmGEM.txt')
        
        self.model = kwargs.pop('model', None)
        if self.model is None:
            self.model = keras.models.load_model(self.model_dir+'emulator_21cmGEM.h5',compile=False)
        
    def __call__(self, parameters):

        # preprocess input physical parameters (same as in preprocess_21cmGEM.py)
        if len(np.shape(parameters)) == 1:
            parameters = np.expand_dims(parameters, axis=0) # if doing one signal at a time
            
        unproc_f_s = parameters[:,0].copy() # f_*, star formation efficiency
        unproc_V_c = parameters[:,1].copy() # V_c, minimum circular velocity of star-forming halos 
        unproc_f_X = parameters[:,2].copy() # f_X, X-ray efficiency of sources
        unproc_f_s = np.log10(unproc_f_s)
        unproc_V_c = np.log10(unproc_V_c)
        unproc_f_X[unproc_f_X == 0] = 1e-6 # for f_X, set zero values to 1e-6 before taking log_10
        unproc_f_X = np.log10(unproc_f_X)
        parameters_log = np.empty(parameters.shape)
        parameters_log[:,0] = unproc_f_s
        parameters_log[:,1] = unproc_V_c
        parameters_log[:,2] = unproc_f_X
        parameters_log[:,3:] = parameters[:,3:].copy()
        
        z_list = np.linspace(5, 50, 451) # list of redshifts for 21cmGEM signals; equiv to np.arange(5, 50.1, 0.1)
        vr = 1420.405751
        nu_list = [vr/(z+1) for z in z_list]
        nu_list = nu_list[::-1] # flip list of frequencies to be from high-z to low-z like done for the brightness temperatures
        nu_list_norm = nu_list/np.max(nu_list) # list to be Min-Max normalized later
        N_proc = np.shape(parameters_log)[0] # number of signals (i.e., parameter sets) to process
        n = len(z_list) # number of frequency channels or redshift bins
        p = np.shape(parameters)[1]+1 # number of input parameters for each LSTM cell/layer (# of physical params plus one for frequency step)
        proc_params_format = np.zeros((N_proc,n,p))
        proc_params = np.zeros_like(proc_params_format)
        
        for i in range(N_proc):
            for j in range(n):
                proc_params_format[i, j, 0:7] = parameters_log[i,:]
                proc_params_format[i, j, 7] = nu_list_norm[j]
        
        for i in range(p):
            x = proc_params_format[:,:,i]
            proc_params[:,:,i] = (x-self.train_mins[i])/(self.train_maxs[i]-self.train_mins[i])

        result = self.model(proc_params, training=False).numpy() # evaluate trained instance of 21cmLSTM with processed parameters
        evaluation = result.T[0].T
        unproc_signals = evaluation.copy()
        unproc_signals = (evaluation*(self.train_maxs[-1]-self.train_mins[-1]))+self.train_mins[-1] # unpreprocess (i.e., denormalize) signals
        unproc_signals = unproc_signals[:,::-1] # flip signals to be from high-z to low-z
        if unproc_signals.shape[0] == 1:
            return unproc_signals[0,:]
        else:
            return unproc_signals


