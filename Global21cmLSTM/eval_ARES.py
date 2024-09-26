import numpy as np
import tensorflow as tf
import gc
from tensorflow import keras
from tensorflow.keras import backend as K
from Global21cmLSTM import __path__

class evaluate_ARES():
    def __init__(self, **kwargs):
        
        for key, values in kwargs.items():
            if key not in set(['model_dir', 'model']):
                raise KeyError("Unexpected keyword argument in evaluate()")
                
        self.model_dir = kwargs.pop('model_dir', 'models/')
        if type(self.model_dir) is not str:
            raise TypeError("'model_dir' must be a sting.")
        elif self.model_dir.endswith('/') is False:
            raise KeyError("'model_dir' must end with '/'.")
        
        self.train_mins = np.loadtxt(self.model_dir+'train_mins_ARES.txt')
        self.train_maxs = np.loadtxt(self.model_dir+'train_maxs_ARES.txt')
        
        self.model = kwargs.pop('model', None)
        if self.model is None:
            self.model = keras.models.load_model(self.model_dir+'emulator_ARES.h5',compile=False)
        
    def __call__(self, parameters):

        # preprocess input physical parameters (same as in preprocess_ARES.py)
        if len(np.shape(parameters)) == 1:
            parameters = np.expand_dims(parameters, axis=0) # if doing one signal at a time

        unproc_c_X = parameters[:,0].copy() # c_X, normalization of X-ray luminosity-SFR relation
        unproc_T_min = parameters[:,2].copy() # T_min, minimum temperature of star-forming halos
        unproc_f_s = parameters[:,4].copy() # f_*,0, peak star formation efficiency 
        unproc_M_p = parameters[:,5].copy() # M_p, dark matter halo mass at f_*,0
        unproc_c_X = np.log10(unproc_c_X)
        unproc_T_min = np.log10(unproc_T_min)
        unproc_f_s = np.log10(unproc_f_s)
        unproc_M_p = np.log10(unproc_M_p)
        parameters_log = np.empty(parameters.shape)
        parameters_log[:,0] = unproc_c_X
        parameters_log[:,1] = parameters[:,1].copy()
        parameters_log[:,2] = unproc_T_min
        parameters_log[:,3] = parameters[:,3].copy()
        parameters_log[:,4] = unproc_f_s
        parameters_log[:,5] = unproc_M_p
        parameters_log[:,6] = parameters[:,6].copy()
        parameters_log[:,7] = parameters[:,7].copy()
        
        z_list = np.linspace(5.1, 49.9, 449) # list of redshifts for ARES signals; equiv to np.arange(5.1, 50, 0.1)
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
                proc_params_format[i, j, 0:8] = parameters_log[i,:]
                proc_params_format[i, j, 8] = nu_list_norm[j]
        
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


