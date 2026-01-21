#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import h5py
import tensorflow as tf
import numpy as np
import os
import torch 
import torch.optim as optim
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from Global21cmLSTM import __path__
import Global21cmLSTM.preprocess_foreground as pp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

# Set default torch type to float64
torch.set_default_dtype(torch.float64)

#PATH = f"{os.environ.get('AUX_DIR', os.environ.get('HOME'))}/.Global21cmLSTM/"
PATH = '/projects/jodo2960/beam_weighted_foreground/'
#nu_list = np.linspace(5,50,180)
nu_list = np.linspace(6,50,176)
#nu_list = np.linspace(5,25,80)
vr = 1420.405751 # rest frequency of 21 cm line in MHz
z_list = (vr/nu_list) - 1
#z_list = np.linspace(27.40811504, 235.73429196, 176)
#with h5py.File(PATH + 'BWFG_Fatima_TS_490k_10region_split_meansub.h5', "r") as f:
with h5py.File(PATH + 'bw_training_set_500k_split_meansub.h5', "r") as f:
    print("Keys: %s" % f.keys())
    par_train = np.asarray(f['par_train'])[()]
    par_val = np.asarray(f['par_val'])[()]
    par_test = np.asarray(f['par_test'])[()]
    spectra_train = np.asarray(f['spectra_train'])[()]
    spectra_val = np.asarray(f['spectra_val'])[()]
    spectra_test = np.asarray(f['spectra_test'])[()]
f.close()

#spectra_train = spectra_train.copy()[:, :80]
#spectra_val = spectra_val.copy()[:, :80]
#spectra_test = spectra_test.copy()[:, :80]

#PATH = f"{os.environ.get('HOME')}/.Global21cmLSTM/"
PATH = '/projects/jodo2960/beam_weighted_foreground/'
#model_save_path = PATH+"models/emulator_foreground_beam_10regions_meansub_21cmLSTM_3layer_5to25MHz.h5"
#train_mins_foreground_beam = np.load(PATH+"models/train_mins_foreground_beam_meansub_10regions.npy")
#train_maxs_foreground_beam = np.load(PATH+"models/train_maxs_foreground_beam_meansub_10regions.npy")
model_save_path = PATH+"models/emulator_foreground_beam_meansub_21cmLSTM_3layer.h5"
train_mins_foreground_beam = np.load(PATH+"models/train_mins_foreground_beam_meansub.npy")
train_maxs_foreground_beam = np.load(PATH+"models/train_maxs_foreground_beam_meansub.npy")

def model(num_frequencies, num_params, dim_output, activation_func="tanh", name=None):
    """
    Generate a 21cmLSTM Keras Sequential model (i.e., linear stack of two LSTM layers and one Dense layer)

    Parameters
    ----------
    num_frequencies : int
        Dimensionality of the output space of each LSTM layer/cell (i.e., number of frequencies per spectrum)
    num_params : int
        Number of physical parameters plus one for the frequency step (e.g., 8 for 21cmGEM set; 9 for ARES set)
    dim_output : int
        Dimensionality of the fully-connected output layer of the model
    activation_func: str or instance of tf.keras.activations
        Activation function for LSTM cells. Default : tanh
    name : str or None
        Name of the model. Default : None

    Returns
    -------
    model : tf.keras.Model
        The generated model
    """
    model = Sequential([LSTM(units=num_frequencies, activation=activation_func, return_sequences=True, input_shape=(num_frequencies, num_params)),
                        LSTM(units=num_frequencies, activation=activation_func, return_sequences=True),
                        LSTM(units=num_frequencies, activation=activation_func, return_sequences=True),
                        Dense(units=dim_output)], name=name)
    return model

def frequency(z):
    """
    Convert redshift to frequency

    Parameters
    ----------
    z : float or np.ndarray
        The redshift or array of redshifts to convert

    Returns
    -------
    nu : float or np.ndarray
        The corresponding frequency or array of frequencies in MHz
    """
    nu = vr/(z+1)
    return nu

def redshift(nu):
    """
    Convert frequency to redshift

    Parameters
    ----------
    nu : float or np.ndarray
        The frequency or array of frequencies in MHz to convert

    Returns
    -------
    z : float or np.ndarray
        The corresponding redshift or array of redshifts
    """
    z = (vr/nu)-1
    return z

def error(true_spectrum, emulated_spectrum, relative=True, nu=None, nu_low=None, nu_high=None):
    """
    Compute the relative rms error (Eq. 3 in DJ+24) between the true and emulated spectra

    Parameters
    ----------
    true_spectrum : np.ndarray
        The true spectra created by the model on which the emulator is trained
        An array of brightness temperatures for different redshifts or frequencies
    emulated_spectrum : np.ndarray
        The emulated spectra. Must be same shape as true_spectrum
    relative : bool
        True to compute the rms error in relative (%) units. False for absolute (mK) units. Default : True
    nu : np.ndarray or None
        The array of frequencies corresponding to each spectrum
        Needed for computing the error in different frequency bands. Default : None.
    nu_low : float or None
        The lower bound of the frequency band to compute the error in
        Cannot be set without nu. Default : None
    nu_high : float or None
        The upper bound of the frequency bnd to compute the error in
        Cannot be set without nu. Default : None

    Returns
    -------
    err : float or np.ndarray
        The computed rms error for each input spectrum

    Raises
    ------
    ValueError : If nu is None and nu_low or nu_high are not None
    """
    if (nu_low or nu_high) and nu is None:
        raise ValueError("Cannot compute error because no frequency array is given.")
    if len(emulated_spectrum.shape) == 1:
        emulated_spectrum = np.expand_dims(emulated_spectrum, axis=0)
        true_spectrum = np.expand_dims(true_spectrum, axis=0)

    if nu_low and nu_high:
        nu_where = np.argwhere((nu >= nu_low) & (nu <= nu_high))[:, 0]
    elif nu_low:
        nu_where = np.argwhere(nu >= nu_low)
    elif nu_high:
        nu_where = np.argwhere(nu <= nu_high)

    if nu_low or nu_high:
        emulated_spectrum = emulated_spectrum[:, nu_where]
        true_spectrum = true_spectrum[:, nu_where]

    err = np.sqrt(np.mean((emulated_spectrum - true_spectrum)**2, axis=1))
    if relative:  # return the rms error as a fraction of the spectrum amplitude in the chosen frequency band
        err /= np.max(np.abs(true_spectrum), axis=1)
        err *= 100 # convert to per cent (%)
    return err

class Emulate:
    def __init__(
        self,
        par_train=par_train,
        par_val=par_val,
        par_test=par_test,
        spectrum_train=spectra_train,
        spectrum_val=spectra_val,
        spectrum_test=spectra_test,
        activation_func='tanh',
        redshifts=z_list,
        frequencies=None):
        """
        The emulator class for building, training, and using 21cmLSTM to emulate beam-weighted foreground spectr
        The default parameters are for training/testing 21cmLSTM on the 21cmGEM set described in Section 2.2 of DJ+24

        Parameters
        ----------
        par_train : np.ndarray
            Parameters in training set
        par_val : np.ndarray
            Parameters in validation set
        par_test : np.ndarray
            Parameters in test set
        spectrum_train : np.ndarray
            spectra in training set
        spectrum_val : np.ndarray
            spectra in validation set
        spectrum_test : np.ndarray
            spectra in test set
        activation_func: str or instance of tf.keras.activations
            Activation function for LSTM cells. Default : tanh
        redshifts : np.ndarray or None
            Array of redshifts corresponding to each spectrum
        frequencies : np.ndarray or None
            Array of frequencies corresponding to each spectrum

        Attributes
        ----------
        par_train : np.ndarray
            Parameters in training set
        par_val : np.ndarray
            Parameters in validation set
        par_test : np.ndarray
            Parameters in test set
        spectrum_train : np.ndarray
            spectra in training set
        spectrum_val : np.ndarray
            spectra in validation set
        spectrum_test : np.ndarray
            spectra in test set
        par_labels : list of str
            Names of the physical parameters
        emulator : tf.keras.Model
            The emulator model
        redshifts : np.ndarray or None
            Array of redshifts corresponding to each spectrum
        frequencies : np.ndarray or None
            Array of frequencies corresponding to each spectrum

        Methods
        -------
        load_model : load an existing instance of 21cmLSTM trained on 21cmGEM data
        train : train the emulator on 21cmGEM data
        predict : use the emulator to predict beam-weighted foreground spectra from input physical parameters
        test_error : compute the rms error of the emulator evaluated on the test set
        save : save the model class instance with all attributes
        """
        self.par_train = par_train
        self.par_val = par_val
        self.par_test = par_test
        self.spectrum_train = spectrum_train
        self.spectrum_val = spectrum_val
        self.spectrum_test = spectrum_test

        #self.par_labels = [r'$A_1$', r'$\beta_1$', r'$\gamma_1$', r'$A_2$', r'$\beta_2$', r'$\gamma_2$', r'$A_3$', r'$\beta_3$', r'$\gamma_3$',\
        #                   r'$A_4$', r'$\beta_4$', r'$\gamma_4$', r'$A_5$', r'$\beta_5$', r'$\gamma_5$', r'$A_6$', r'$\beta_6$', r'$\gamma_6$',\
        #                   r'$A_7$', r'$\beta_7$', r'$\gamma_7$', r'$A_8$', r'$\beta_8$', r'$\gamma_8$', r'$A_9$', r'$\beta_9$', r'$\gamma_9$',\
        #                   r'$A_10$', r'$\beta_10$', r'$\gamma_10$', r'L', r'$\epsilon_{top}$', r'$\epsilon_{bottom}$'] 
        self.par_labels = [r'$A_1$', r'$\beta_1$', r'$\gamma_1$', r'$A_2$', r'$\beta_2$', r'$\gamma_2$', r'$A_3$', r'$\beta_3$', r'$\gamma_3$',\
                           r'$A_4$', r'$\beta_4$', r'$\gamma_4$', r'$A_5$', r'$\beta_5$', r'$\gamma_5$',\
                           r'L', r'$\epsilon_{top}$', r'$\epsilon_{bottom}$'] 

        self.emulator = model(self.spectrum_train.shape[-1], self.par_train.shape[-1]+1, 1,
                              activation_func, name="emulator_foreground_beam_meansub_21cmLSTM_3layer") #emulator_foreground_beam_10regions_meansub_21cmLSTM_3layer_5to25MHz

        self.train_mins = train_mins_foreground_beam
        self.train_maxs = train_maxs_foreground_beam
        
        if frequencies is None:
            if redshifts is not None:
                frequencies = frequency(redshifts)
        elif redshifts is None:
            redshifts = redshift(frequencies)
        self.redshifts = redshifts
        self.frequencies = frequencies

    def load_model(self, model_path=PATH+"models/emulator_foreground_beam_meansub_21cmLSTM_3layer.h5"): #emulator_foreground_beam_10regions_meansub_21cmLSTM_3layer_5to25MH
        """
        Load a saved model instance of 21cmLSTM trained on the beam-weighted foreground spectra data set.

        Parameters
        ----------
        model_path : str
            The path to the saved model instance

        Raises
        ------
        IOError : if model_path does not point to a valid model instance
        """
        self.emulator = tf.keras.models.load_model(model_path)
        # print(f"Loading model from: {model_path}")
        # self.emulator = torch.load(model_path, weights_only=False)
        # self.emulator.to(device)

    def train(self, epochs, batch_size, callbacks=[], verbose=2, shuffle='True'):
        """
        Train an instance of 21cmLSTM on the beam-weighted foreground spectra data set

        Parameters
        ----------
        epochs : int
            Number of epochs to train for the given batch_size
        batch_size : int
            Number of spectra in each minibatch trained on
        callbacks : list of tf.keras.callbacks.Callback
            Callbacks to pass to the training loop. Default : []
        verbose : 0, 1, 2
            Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. Default : 2

        Returns
        -------
        train_loss : list of floats
           Training set loss values for each epoch
        val_loss : list of floats
           Validation set loss values for each epoch
        """
        X_train = pp.preproc_params(self.par_train, self.par_train)
        X_val = pp.preproc_params(self.par_val, self.par_train)
        y_train = pp.preproc_spectra(self.spectrum_train, self.spectrum_train)
        y_val = pp.preproc_spectra(self.spectrum_val, self.spectrum_train)

        hist = self.emulator.fit(
            x=X_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            validation_batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose, shuffle=shuffle)

        train_loss = hist.history["loss"]
        val_loss = hist.history["val_loss"]

        self.emulator.save(PATH+'models/emulator_foreground_beam_meansub_21cmLSTM_3layer.h5') # save the entire model in HDF5 format; emulator_foreground_beam_10regions_meansub_21cmLSTM_3layer_5to25MH.h5

        return train_loss, val_loss

    def predict(self, params):
        """
        Predict beam-weighted foreground spectra from input physical parameters using trained instance of 21cmLSTM

        Parameters
        ----------
        params : np.ndarray
            The values of the physical parameters in the order of par_labels. Input 2D array to predict a set of spectra

        Returns
        -------
        emulated_spectra : np.ndarray
           The predicted beam-weighted foreground spectra
        """
        proc_params = pp.preproc_params(params, self.par_train)
        proc_emulated_spectra = self.emulator.predict(proc_params)
        emulated_spectra = pp.unpreproc_spectra(proc_emulated_spectra, self.spectrum_train)
        emulated_spectra = np.squeeze(emulated_spectra, axis=2)
        if emulated_spectra.shape[0] == 1:
            return emulated_spectra[0, :]
        else:
            return emulated_spectra
    
        # if len(np.shape(params)) == 1:
        #     params = np.expand_dims(params, axis=0) # if doing one spectrum at a time

        # nu_list = np.linspace(6,50,176)
        # vr = 1420.405751
        # z_list = (vr/nu_list) - 1
        # nu_list_norm = nu_list/np.max(nu_list)

        # # include the same parameter preprocessing code performed earlier to transform the input
        # # physical parameters to normalized parameters that can be used to train 21cmKAN
        # N_proc = np.shape(params)[0] # number of spectra (i.e., parameter sets) to process
        # n = len(z_list) # number of frequency channels, same as the number of redshift bins
        # p = np.shape(params)[1]+1 # number of input parameters for each LSTM cell/layer (# of physical params plus one for frequency step)
        # proc_params_format = np.zeros((N_proc,n,p))
        # proc_params = np.zeros_like(proc_params_format)

        # for i in range(N_proc):
        #     for j in range(n):
        #         proc_params_format[i, j, 0:18] = params[i,:]
        #         proc_params_format[i, j, 18] = nu_list_norm[j]
        
        # for i in range(p):
        #     x = proc_params_format[:,:,i]
        #     proc_params[:,:,i] = (x-self.train_mins[i])/(self.train_maxs[i]-self.train_mins[i])

        # #proc_params_test = torch.from_numpy(proc_params)
        # #proc_params_test = proc_params_test.to(device)
        # proc_params_format = 0
        # params = 0

        # proc_spectra = self.emulator.predict(proc_params)
        # unproc_spectra = proc_spectra.copy()
        # unproc_spectra = (proc_spectra*(self.train_maxs[-1]-self.train_mins[-1]))+self.train_mins[-1] # denormalize spectra
        # unproc_spectra = np.squeeze(unproc_spectra, axis=2)
        # #unproc_spectra = unproc_spectra[:,::-1] # flip spectra to be from high-z to low-z
        # if unproc_spectra.shape[0] == 1:
        #     return unproc_spectra[0,:]
        # else:
        #     return unproc_spectra

    def test_error(self, relative=True, nu_low=None, nu_high=None):
        """
        Compute the rms error for the loaded trained instance of 21cmLSTM evaluated on the test set

        Parameters
        ----------
        relative : bool
            True to compute the rms error in relative (%) units. False for absolute (K) units. Default : True
        nu_low : float or None
            The lower bound of the frequency band to compute the error in
            Default : None
        nu_high : float or None
            The upper bound of the frequency band to compute the error in
            Default : None

        Returns
        -------
        err : np.ndarray
            The computed rms errors
        """
        err = error(
            self.spectrum_test,
            self.predict(self.par_test),
            relative=relative,
            nu=self.frequencies,
            nu_low=nu_low,
            nu_high=nu_high)
        return err


