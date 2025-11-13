#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
import gc
import os
from tensorflow import keras
from tensorflow.keras import backend as K
import Global21cmLSTM as Global21cmLSTM
import Global21cmLSTM.preprocess_foreground as pp
from Global21cmLSTM import __path__

#PATH = f"{os.environ.get('HOME')}/.Global21cmLSTM/"

class evaluate_foreground():
    def __init__(self, **kwargs):
        emulator_foreground = Global21cmLSTM.emulator_foreground.Emulate()
        self.par_train = emulator_foreground.par_train
        self.spectrum_train = emulator_foreground.spectrum_train
        PATH = '/projects/jodo2960/beam_weighted_foreground/models/'
        self.emulator = tf.keras.models.load_model(PATH+'emulator_foreground_beam_meansub_21cmLSTM_long.h5')

    def __call__(self, parameters):
        proc_params = pp.preproc_params(params, self.par_train)
        proc_emulated_spectra = self.emulator.predict(proc_params)
        emulated_spectra = pp.unpreproc_spectra(proc_emulated_spectra, self.spectrum_train)
        emulated_spectra = np.squeeze(emulated_spectra, axis=2)
        if emulated_spectra.shape[0] == 1:
            return emulated_spectra[0, :]
        else:
            return emulated_spectra

