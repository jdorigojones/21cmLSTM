#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import os

#PATH = f"{os.environ.get('HOME')}/.Global21cmLSTM/"
# PATH = '/projects/jodo2960/beam_weighted_foreground/'
# model_save_path = PATH+"models/emulator_foreground_beam_meansub.pth"
# train_mins_foreground_beam = np.load(PATH+"models/train_mins_foreground_beam_meansub.npy")
# train_maxs_foreground_beam = np.load(PATH+"models/train_maxs_foreground_beam_meansub.npy")

def preproc_params(unproc_params: np.ndarray, train_params: np.ndarray) -> np.ndarray:
    """
    Preprocess a set of parameters the same way that the training set parameters are processed (Section 2.2 in DJ+24):
    (1) apply Min-Max normalization to data and labels

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
        unproc_params = np.expand_dims(unproc_params, axis=0) # if doing one spectrum at a time

    nu_list = np.linspace(6,50,176) #np.linspace(5,50,180)
    vr = 1420.405751
    z_list = (vr/nu_list) - 1
    nu_list_norm = nu_list/np.max(nu_list) # frequency list to be Min-Max normalized

    N_train = np.shape(train_params)[0] # number of spectra (i.e., parameter sets) in training set
    N_proc = np.shape(unproc_params)[0] # number of spectra (i.e., parameter sets) to process
    n = len(z_list) # number of frequency channels, same as the number of redshift bins
    p = np.shape(unproc_params)[1]+1 # number of input parameters for each LSTM cell/layer (# of physical params plus one for frequency step)
    train_params_format = np.zeros((N_train,n,p))
    proc_params_format = np.zeros((N_proc,n,p))
    proc_params = np.zeros_like(proc_params_format)
    train_data_mins = []
    train_data_maxs = []

    for i in range(N_train):
        for j in range(n):
            train_params_format[i, j, 0:18] = train_params[i,:]
            train_params_format[i, j, 18] = nu_list_norm[j]

    for i in range(N_proc):
        for j in range(n):
            proc_params_format[i, j, 0:18] = unproc_params[i,:]
            proc_params_format[i, j, 18] = nu_list_norm[j]

    for i in range(p):
        train_min = np.min(train_params_format[:,:,i])
        train_max = np.max(train_params_format[:,:,i])
        train_data_mins.append(train_min)
        train_data_maxs.append(train_max)
        x = proc_params_format[:,:,i]
        proc_params[:,:,i] = (x-train_min)/(train_max-train_min)

    #np.savetxt(PATH+'models/train_mins_modelname.txt', train_data_mins) # these lines are for saving the min and max values of the training set 
    #np.savetxt(PATH+'models/train_maxs_modelname.txt', train_data_maxs) # used for spectrum denormalization; uncomment if using different data
    return proc_params

def preproc_spectra(unproc_spectra: np.ndarray, train_spectra: np.ndarray) -> np.ndarray:
    """
    Preprocess a set of spectra

    Parameters
    ----------
    unproc_spectra : np.ndarray
        Array of brightness temperatures for unprocessed spectra
    train_spectra : np.ndarray
        Array of brightness temperatures for spectra in training set

    Returns
    -------
    proc_spectra : np.ndarray
        Array of brightness temperatures for preprocessed spectra
    """
    train_spectra_min = np.min(train_spectra)
    train_spectra_max = np.max(train_spectra)
    # with open(PATH+"models/train_mins_modelname.txt", "a") as f: # uncomment to save if using different data set
    #    f.write(str(train_spectra_min))
    # with open(PATH+"models/train_maxs_modelname.txt", "a") as file: # uncomment to save if using different data set
    #    file.write(str(train_spectra_max))
    proc_spectra = unproc_spectra.copy()
    proc_spectra = (unproc_spectra - train_spectra_min)/(train_spectra_max-train_spectra_min)  # global Min-Max normalization
    return proc_spectra

def unpreproc_spectra(proc_spectra: np.ndarray, train_spectra: np.ndarray) -> np.ndarray:
    """
    Inverse of preproc_spectra function

    Parameters
    ----------
    proc_spectra : np.ndarray
        Array of brightness temperatures for preprocesed spectra to unpreprocess
    train_spectra : np.ndarray
        Array of brightness temperatures for spectra in training set used for preprocessing

    Returns
    --------
    unproc_spectra : np.ndarray
        Array of unprocessed spectra
    """
    train_spectra_min = np.min(train_spectra)
    train_spectra_max = np.max(train_spectra)
    unproc_spectra = proc_spectra.copy()
    unproc_spectra = (proc_spectra*(train_spectra_max-train_spectra_min))+train_spectra_min
    return unproc_spectra


