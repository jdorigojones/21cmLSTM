#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def preproc_params(unproc_params: np.ndarray, train_params: np.ndarray) -> np.ndarray:
    """
    Preprocess a set of parameters the same way that the training set parameters are processed (Section 2.2 in DJ+24):
    (1) take log_10 of parameters that are uniform in log10-space (f_X ,V_c, and f_*);
    (2) flip signals to be from high-z to low-z;
    (3) apply Min-Max normalization to data and labels

    Parameters
    ----------
    unproc_params : np.ndarray
        Array of unprocessed parameters
    train_params : np.ndarray
        Array of parameters used for training

    Returns
    -------
    proc_params : np.ndarray
        Array of preprocessed parameters
    """
    if len(np.shape(unproc_params)) == 1:
        unproc_params = np.expand_dims(unproc_params, axis=0) # if doing one signal at a time

    # make copies of parameter arrays to log because the lists are linked if input arrays are the same
    unproc_f_s = unproc_params[:,0].copy() # f_*, star formation efficiency
    unproc_V_c = unproc_params[:,1].copy() # V_c, minimum circular velocity of star-forming halos 
    unproc_f_X = unproc_params[:,2].copy() # f_X, X-ray efficiency of sources
    train_f_s = train_params[:,0].copy()
    train_V_c = train_params[:,1].copy()
    train_f_X = train_params[:,2].copy()

    # take log_10 of the parameter arrays that are uniform in log10-space
    unproc_f_s = np.log10(unproc_f_s)
    unproc_V_c = np.log10(unproc_V_c)
    train_f_s = np.log10(train_f_s)
    train_V_c = np.log10(train_V_c)
    unproc_f_X[unproc_f_X == 0] = 1e-6 # for f_X, set zero values to 1e-6 before taking log_10 (also done for 21cmVAE)
    train_f_X[train_f_X == 0] = 1e-6
    unproc_f_X = np.log10(unproc_f_X)
    train_f_X = np.log10(train_f_X)

    # create new unprocessed and training parameter arrays with the logged parameters
    unproc_params_log = np.empty(unproc_params.shape)
    unproc_params_log[:, 0] = unproc_f_s
    unproc_params_log[:, 1] = unproc_V_c
    unproc_params_log[:, 2] = unproc_f_X
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
    nu_list_norm = nu_list/np.max(nu_list) # list to be Min-Max normalized later

    N_train = np.shape(train_params_log)[0] # number of signals (i.e., parameter sets) in training set
    N_proc = np.shape(unproc_params_log)[0] # number of signals (i.e., parameter sets) to process
    n = len(z_list) # number of frequency channels or redshift bins
    p = np.shape(unproc_params)[1]+1 # number of input parameters for each LSTM cell/layer (# of physical params plus one for frequency step)
    train_params_format = np.zeros((N_train,n,p))
    proc_params_format = np.zeros((N_proc,n,p))
    proc_params = np.zeros_like(proc_params_format)
    train_data_mins = []
    train_data_maxs = []
    
    for i in range(N_train):
        for j in range(n):
            train_params_format[i, j, 0:7] = train_params_log[i,:]
            train_params_format[i, j, 7] = nu_list_norm[j]
    
    for i in range(N_proc):
        for j in range(n):
            proc_params_format[i, j, 0:7] = unproc_params_log[i,:]
            proc_params_format[i, j, 7] = nu_list_norm[j]
            
    for i in range(p):
        train_min = np.min(train_params_format[:,:,i])
        train_max = np.max(train_params_format[:,:,i])
        train_data_mins.append(train_min)
        train_data_maxs.append(train_max)
        x = proc_params_format[:,:,i]
        proc_params[:,:,i] = (x-train_min)/(train_max-train_min)

    #np.savetxt('models/train_mins_21cmGEM.txt', train_data_mins)
    #np.savetxt('models/train_maxs_21cmGEM.txt', train_data_maxs)
    return proc_params

def preproc_signals(unproc_signals: np.ndarray, train_signals: np.ndarray) -> np.ndarray:
    """
    Preprocess a set of signals

    Parameters
    ----------
    unproc_signals : np.ndarray
        Array of brightness temperatures for unprocessed signals
    train_signals : np.ndarray
        Array of brightness temperatures for signals in training set

    Returns
    -------
    proc_signals : np.ndarray
        Array of brightness temperatures for preprocessed signals
    """
    train_signals_min = np.min(train_signals)
    train_signals_max = np.max(train_signals)
    # with open("models/train_mins_21cmGEM.txt", "a") as f:
    #    f.write(str(train_signals_min))
    # with open("models/train_maxs_21cmGEM.txt", "a") as file:
    #    file.write(str(train_signals_max))
    proc_signals = unproc_signals.copy()
    proc_signals = (unproc_signals - train_signals_min)/(train_signals_max-train_signals_min)  # global Min-Max normalization
    proc_signals = proc_signals[:,::-1] # flip signals to be from high-z to low-z
    return proc_signals

def unpreproc_signals(proc_signals: np.ndarray, train_signals: np.ndarray) -> np.ndarray:
    """
    Inverse of preproc_signals function

    Parameters
    ----------
    proc_signals : np.ndarray
        Array of brightness temperatures for preprocesed signals to unpreprocess
    train_signals : np.ndarray
        Array of brightness temperatures for signals in training set used for preprocessing

    Returns
    --------
    unproc_signals : np.ndarray
        Array of unprocessed signals
    """
    train_signals_min = np.min(train_signals)
    train_signals_max = np.max(train_signals)
    unproc_signals = proc_signals.copy()
    unproc_signals = (proc_signals*(train_signals_max-train_signals_min))+train_signals_min
    unproc_signals = unproc_signals[:,::-1] # flip signals to be from high-z to low-z
    return unproc_signals


