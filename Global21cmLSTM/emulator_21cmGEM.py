#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import h5py
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from Global21cmLSTM import __path__
import Global21cmLSTM.preprocess_21cmGEM as pp

PATH = f"{os.environ.get('HOME')}/.Global21cmLSTM/"
z_list = np.linspace(5, 50, 451) # list of redshifts for 21cmGEM signals; equiv to np.arange(5, 50.1, 0.1)
with h5py.File(PATH + 'dataset_21cmGEM.h5', "r") as f:
    print("Keys: %s" % f.keys())
    par_train = np.asarray(f['par_train'])[()]
    par_val = np.asarray(f['par_val'])[()]
    par_test = np.asarray(f['par_test'])[()]
    signal_train = np.asarray(f['signal_train'])[()]
    signal_val = np.asarray(f['signal_val'])[()]
    signal_test = np.asarray(f['signal_test'])[()]
f.close()

def model(num_frequencies, num_params, dim_output, activation_func="tanh", name=None):
    """
    Generate a 21cmLSTM Keras Sequential model (i.e., linear stack of two LSTM layers and one Dense layer)

    Parameters
    ----------
    num_frequencies : int
        Dimensionality of the output space of each LSTM layer/cell (i.e., number of frequencies per signal)
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
                        Dense(units=dim_output)], name=name)
    return model

vr = 1420.4057517667  # rest frequency of 21 cm line in MHz

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

def error(true_signal, emulated_signal, relative=True, nu=None, nu_low=None, nu_high=None):
    """
    Compute the relative rms error (Eq. 3 in DJ+24) between the true and emulated signal(s)

    Parameters
    ----------
    true_signal : np.ndarray
        The true signal(s) created by the model on which the emulator is trained
        An array of brightness temperatures for different redshifts or frequencies
    emulated_signal : np.ndarray
        The emulated signal(s). Must be same shape as true_signal
    relative : bool
        True to compute the rms error in relative (%) units. False for absolute (mK) units. Default : True
    nu : np.ndarray or None
        The array of frequencies corresponding to each signal
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
        The computed rms error for each input signal

    Raises
    ------
    ValueError :
        If nu is None and nu_low or nu_high are not None.
    """
    if (nu_low or nu_high) and nu is None:
        raise ValueError("Cannot compute error in specified frequency band because no frequency array is given.")
    if len(emulated_signal.shape) == 1:
        emulated_signal = np.expand_dims(emulated_signal, axis=0)
        true_signal = np.expand_dims(true_signal, axis=0)

    if nu_low and nu_high:
        nu_where = np.argwhere((nu >= nu_low) & (nu <= nu_high))[:, 0]
    elif nu_low:
        nu_where = np.argwhere(nu >= nu_low)
    elif nu_high:
        nu_where = np.argwhere(nu <= nu_high)

    if nu_low or nu_high:
        emulated_signal = emulated_signal[:, nu_where]
        true_signal = true_signal[:, nu_where]

    err = np.sqrt(np.mean((emulated_signal - true_signal)**2, axis=1))
    if relative:  # return the rms error as a fraction of the signal amplitude in the chosen frequency band
        err /= np.max(np.abs(true_signal), axis=1)
        err *= 100 # convert to per cent (%)
    return err

class Emulate:
    def __init__(
        self,
        par_train=par_train,
        par_val=par_val,
        par_test=par_test,
        signal_train=signal_train,
        signal_val=signal_val,
        signal_test=signal_test,
        activation_func='tanh',
        redshifts=z_list,
        frequencies=None):
        """
        The emulator class for building, training, and using 21cmLSTM to emulate 21cmGEM signals 
        The default parameters are for training/testing 21cmLSTM on the 21cmGEM set described in Section 2.2 of DJ+24

        Parameters
        ----------
        par_train : np.ndarray
            Parameters in training set
        par_val : np.ndarray
            Parameters in validation set
        par_test : np.ndarray
            Parameters in test set
        signal_train : np.ndarray
            Signals in training set
        signal_val : np.ndarray
            Signals in validation set
        signal_test : np.ndarray
            Signals in test set
        activation_func: str or instance of tf.keras.activations
            Activation function for LSTM cells. Default : tanh
        redshifts : np.ndarray or None
            Array of redshifts corresponding to each signal
        frequencies : np.ndarray or None
            Array of frequencies corresponding to each signal

        Attributes
        ----------
        par_train : np.ndarray
            Parameters in training set
        par_val : np.ndarray
            Parameters in validation set
        par_test : np.ndarray
            Parameters in test set
        signal_train : np.ndarray
            Signals in training set
        signal_val : np.ndarray
            Signals in validation set
        signal_test : np.ndarray
            Signals in test set
        par_labels : list of str
            Names of the physical parameters
        emulator : tf.keras.Model
            The emulator model
        redshifts : np.ndarray or None
            Array of redshifts corresponding to each signal
        frequencies : np.ndarray or None
            Array of frequencies corresponding to each signal

        Methods
        -------
        load_model : load an existing instance of 21cmLSTM trained on 21cmGEM data
        train : train the emulator on 21cmGEM data
        predict : use the emulator to predict global signals from input physical parameters
        test_error : compute the rms error of the emulator evaluated on the test set
        save : save the model class instance with all attributes
        """
        self.par_train = par_train
        self.par_val = par_val
        self.par_test = par_test
        self.signal_train = signal_train
        self.signal_val = signal_val
        self.signal_test = signal_test

        self.par_labels = [r'$f_*$', r'$V_c$', r'$f_X$', r'$\tau$',
                           r'$\alpha$', r'$\nu_{\rm min}$', r'$R_{\rm mfp}$']

        self.emulator = model(self.signal_train.shape[-1], self.par_train.shape[-1]+1, 1,
                              activation_func, name="emulator_21cmGEM")

        if frequencies is None:
            if redshifts is not None:
                frequencies = frequency(redshifts)
        elif redshifts is None:
            redshifts = redshift(frequencies)
        self.redshifts = redshifts
        self.frequencies = frequencies

    def load_model(self, model_path=PATH+"models/emulator_21cmGEM.h5"):
        """
        Load a saved model instance of 21cmLSTM trained on the 21cmGEM data set.
        The instance of 21cmLSTM trained on 21cmGEM included in this repository is the one
        used to perform nested sampling analyses in DJ+24

        Parameters
        ----------
        model_path : str
            The path to the saved model.

        Raises
        ------
        IOError : if model_path does not point to a valid model.
        """
        self.emulator = tf.keras.models.load_model(model_path)

    def train(self, epochs, batch_size, callbacks=[], verbose=2, shuffle='True'):
        """
        Train an instance of 21cmLSTM on the 21cmGEM data set

        Parameters
        ----------
        epochs : int
            Number of epochs to train for the given batch_size
        batch_size : int
            Number of signals in each minibatch trained on
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
        y_train = pp.preproc_signals(self.signal_train, self.signal_train)
        y_val = pp.preproc_signals(self.signal_val, self.signal_train)
        
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

        self.emulator.save('models/emulator_21cmGEM.h5') # save the entire model in HDF5 format

        return train_loss, val_loss

    def predict(self, params):
        """
        Predict global 21 cm signal(s) from input 21cmGEM physical parameters using trained instance of 21cmLSTM

        Parameters
        ----------
        params : np.ndarray
            The values of the physical parameters in the order of par_labels. Input 2D array to predict a set of signals

        Returns
        -------
        emulated_signals : np.ndarray
           The predicted global 21 cm signal(s)
        """
        proc_params = pp.preproc_params(params, self.par_train)
        proc_emulated_signals = self.emulator.predict(proc_params)
        emulated_signals = pp.unpreproc_signals(proc_emulated_signals, self.signal_train)
        if emulated_signals.shape[0] == 1:
            return emulated_signals[0, :]
        else:
            return emulated_signals

    def test_error(self, relative=True, nu_low=None, nu_high=None):
        """
        Compute the rms error for the loaded trained instance of 21cmLSTM evaluated on the 1,704-signal 21cmGEM test set

        Parameters
        ----------
        relative : bool
            True to compute the rms error in relative (%) units. False for absolute (mK) units. Default : True
        nu_low : float or None
            The lower bound of the frequency band to compute the error in.
            Default : None.
        nu_high : float or None
            The upper bound of the frequency bnd to compute the error in.
            Default : None.

        Returns
        -------
        err : np.ndarray
            The computed rms errors
        """
        err = error(
            self.signal_test,
            self.predict(self.par_test),
            relative=relative,
            nu=self.frequencies,
            nu_low=nu_low,
            nu_high=nu_high)
        return err

