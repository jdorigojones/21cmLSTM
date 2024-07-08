import h5py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
from cosmoLSTM import __path__
import cosmoLSTM.preprocess as pp

PATH = __path__[0] + "/"
z_list = np.linspace(5, 50, 451) # list of redshifts for 21cmGEM signals; equiv to np.arange(5, 50.1, 0.1)
data = 'dataset_21cmLSTM.h5'
with h5py.File(data, "r") as f:
    print("Keys: %s" % f.keys())
    par_train = np.asarray(f['par_train'])[()]
    par_val = np.asarray(f['par_val'])[()]
    par_test = np.asarray(f['par_test'])[()]
    signal_train = np.asarray(f['signal_train'])[()]
    signal_val = np.asarray(f['signal_val'])[()]
    signal_test = np.asarray(f['signal_test'])[()]
f.close()

def model(dim_input, num_params, dim_output, activation_func="tanh", name=None):
    """
    Generate a 21cmLSTM keras Sequential model (i.e., linear stack of two LSTM layers and one Dense layer)

    Parameters
    ----------
    dim_input : int
        The number of units (i.e., dimensionality) of each LSTM layer/cell
    num_params : int
        Number of physical parameters plus one for the frequency step
    dim_output : int
        The dimensionality of the output layer of the model
    activation_func: str or instance of tf.keras.activations
        Activation function for LSTM cells. Default is tanh
    name : str or None
       Name of the model. Default : None

    Returns
    -------
    model : tf.keras.Model
        The generated model

    """
    model = Sequential([LSTM(units=dim_input, activation=activation_func, return_sequences=True, input_shape=(dim_input, num_params)),
                        LSTM(units=dim_input, activation=activation_func, return_sequences=True),
                        Dense(units=dim_output)], name=name)
    return model

vr = 1420.4057517667  # rest frequency of 21 cm line in MHz

def frequency(z):
    """
    Convert redshift to frequency.

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
    Convert frequency to redshfit.

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
    Compute the relative rms error (Eq. 3 in the paper) between the true and emulated signal(s).

    Parameters
    ----------
    true_signal : np.ndarray
        The true signal(s) created by the model on which the emulator is trained
        An array of dT_b for different redshifts or frequencies
    emulated_signal : np.ndarray
        The emulated signal(s). Must be same shape as true_signal.
    relative : bool
        Whether to compute the rms error in relative (%) or absolute (mK) units. Default : True.
    nu : np.ndarray or None
        The array of frequencies corresponding to each signal.
        Needed for computing the error in different frequency bands. Default : None.
    nu_low : float or None
        The lower bound of the frequency band to compute the error in.
        Cannot be set without nu. Default : None.
    nu_high : float or None
        The upper bound of the frequency bnd to compute the error in.
        Cannot be set without nu. Default : None.

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
        err *= 100 # convert to %
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
        frequencies=None,
    ):
        """
        The emulator class for building, training, and using 21cmLSTM
        The default parameters are for training/testing 21cmLSTM on the 21cmGEM set

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
            Activation function for LSTM cells. Default is tanh
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
            The names of the physical parameters
        emulator : tf.keras.Model
            The emulator
        redshifts : np.ndarray or None
            Array of redshifts corresponding to each signal
        frequencies : np.ndarray or None
            Array of frequencies corresponding to each signal

        Methods
        -------
        load_model : load an exsisting model
        train : train the emulator
        predict : use the emulator to predict global signals from input physical parameters
        test_error : compute the test set error of the emulator
        save : save the class instance with all attributes

        """

        self.par_train = par_train
        self.par_val = par_val
        self.par_test = par_test
        self.signal_train = signal_train
        self.signal_val = signal_val
        self.signal_test = signal_test

        self.par_labels = [
            "fstar",
            "Vc",
            "fx",
            "tau",
            "alpha",
            "nu_min",
            "Rmfp"]

        self.emulator = model(
            self.signal_train.shape[-1],
            self.par_train.shape[-1]+1,
            1,
            activation_func,
            name="emulator")

        if frequencies is None:
            if redshifts is not None:
                frequencies = frequency(redshifts)
        elif redshifts is None:
            redshifts = redshift(frequencies)
        self.redshifts = redshifts
        self.frequencies = frequencies

    def load_model(self, model_path=PATH+"models/emulator.h5"):
        """
        Load a saved model (i.e., trained instance of 21cmLSTM).
        The default is the path to the best trial described in the paper.

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
        Train an instance of the 21cmLSTM emulator of the global 21 cm signal

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
           Training set losses
        val_loss : list of floats
           Validation set losses

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

        # Save the entire model as a `.keras` zip archive.
        self.emulator.save('models/emulator.h5')

        return train_loss, val_loss

    def predict(self, params):
        """
        Predict global 21 cm signal(s) from input physical parameters using trained instance of 21cmLSTM

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
            return emulated_signals[:,:,0]

    def test_error(self, relative=True, nu_low=None, nu_high=None):
        """
        Compute the rms error for trained instance of 21cmLSTM evaluated on test set

        Parameters
        ----------
        relative : bool
            Whether to compute the error in % relative to the signal amplitude
            (True) or in mK (False). Default : True.
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

    def save(self):
        raise NotImplementedError("Not implemented yet.")

