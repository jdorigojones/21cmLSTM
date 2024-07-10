#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

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
    proc_signals = unproc_signals.copy()
    proc_signals = (unproc_signals - np.min(train_signals))/(np.max(train_signals)-np.min(train_signals))  # global Min-Max normalization
    proc_signals = proc_signals[:,::-1] # flip signals to be from high-z to low-z
    return proc_signals

####################################################

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
    unproc_signals = proc_signals.copy()
    unproc_signals = (proc_signals*(np.max(train_signals)-np.min(train_signals)))+np.min(train_signals)
    unproc_signals = unproc_signals[:,::-1] # flip signals to be from high-z to low-z
    return unproc_signals

##################################################


def preproc_params(unproc_params: np.ndarray, train_params: np.ndarray) -> np.ndarray:
    """
    Preprocess a set of parameters the same way that the training set parameters are processed:
    (1) take log10 of the first three parameters;
    (2) flip signals to go from high-z to low-z;
    (3) apply normalization

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

    # create new unprocessed and training parameter arrays with the logged parameters
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

    return proc_params

