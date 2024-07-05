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
    proc_signals = np.zeros_like(unproc_signals)
    proc_signals = (unproc_signals - np.min(train_signals))/(np.max(train_signals)-np.min(train_signals))  # global Min-Max normalization
    proc_signals = proc_signals[:,::-1] # flip signals to be from high-z to low-z
    return proc_signal

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
    unproc_signals = np.zeros_like(proc_signals)
    unproc_signals = (proc_signals*(np.max(train_signals)-np.min(train_signals)))+np.min(train_signals)
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

    train_params[:,0] = np.log10(train_params[:,0])
    train_params[:,1] = np.log10(train_params[:,1])
    train_params[:,2][train_params[:,2] == 0] = 1e-6 # set fX=0 values to 1e-6 before taking log10
    train_params[:,2] = np.log10(train_params[:,2])
    unproc_params[:,0] = np.log10(unproc_params[:,0])
    unproc_params[:,1] = np.log10(unproc_params[:,1])
    unproc_params[:,2][unproc_params[:,2] == 0] = 1e-6 # set fX=0 values to 1e-6 before taking log10
    unproc_params[:,2] = np.log10(unproc_params[:,2])

    z_list = np.linspace(5, 50, 451) # list of redshifts for 21cmGEM signals; equiv to np.arange(5, 50.1, 0.1)
    vr = 1420.405751
    nu_list = [vr/(z+1) for z in z_list]
    nu_list = nu_list[::-1] # flip list of frequencies to be from high-z to low-z like done for the brightness temperatures
    nu_list_norm = nu_list/np.max(nu_list)

    N_train = np.shape(train_params)[0] # number of parameter sets (i.e., signals) in training set
    N = np.shape(unproc_params)[0] # number of parameter sets (i.e., signals) to process
    n = len(z_list) # number of redshift/frequency bins (i.e., signal resolution)
    p = np.shape(unproc_params)[1]+1 # number of input parameters for each LSTM cell/layer (i.e., # of physical params + 1 for z/nu step)
    train_params_format = np.zeros((N_train,n,p))
    params_format = np.zeros((N,n,p))
    proc_params = np.zeros_like(params_format)
    for i in range(N_train):
        for j in range(n):
            train_params_format[i, j, 0:7] = train_params[i,:]
            train_params_format[i, j, 7] = nu_list_norm[j]
    for i in range(N):
        for j in range(n):
            params_format[i, j, 0:7] = unproc_params[i,:]
            params_format[i, j, 7] = nu_list_norm[j]
    for i in range(p):
        x = params_format[:, :, i]
        proc_params[:,:,i] = (x - np.min(train_params_format[:, :, i]))/(np.max(train_params_format[:, :, i]) - np.min(train_params_format[:, :, i]))
    return proc_params

