#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cosmoLSTM as cosmoLSTM

emulator = cosmoLSTM.emulator.Emulate() # initialize emulator with 21cmGEM data sets
emulator.load_model()  # load pretrained instance of 21cmLSTM emulator. Use kwarg 'model_path' to load other models.

emulator.emulator.summary()

# print the input physical parameters
print(emulator.par_labels)

# define list of parameters. can also be shape (N_signals,7)
params = [0.0003, 4.2, 0, 0.055, 1.0, 0.1, 10]  # in the order "fstar", "Vc", "fx", "tau", "alpha", "nu_min", "Rmfp"

signal_emulated = emulator.predict(params)  # emulate the global signal

import matplotlib.pyplot as plt
frequencies = emulator.frequencies  # array of frequencies (dz=0.1, length=451)
plt.figure()
plt.plot(frequencies, signal_emulated)
plt.title('Predicted Global 21 signal')
plt.xlabel(r'$\nu$ (MHz)')
plt.ylabel(r'$\delta T_b$ (mK)')
plt.show()

import numpy as np
rel_error = emulator.test_error()
abs_error = emulator.test_error(relative=False, nu_low=50, nu_high=100)
# compute the relative rms error for this instance of 21cmLSTM evaluated on the same 21cmGEM test set used in the paper
# compute the absolute rms error in the frequency range 50-100 MHz
print('Mean relative rms error:', np.mean(rel_error), '%')
print('Mean absolute rms error for 50-100 MHz:', np.mean(abs_error), 'mK')
# the results should fall within the range found for 21cmLSTM when trained and tested on the 21cmGEM set for 20 trials (Fig. 2 in paper)

# Histogram of relative rms errors for this instance of 21cmLSTM, should look similar to top right panel of Fig. 1 in the paper
plt.hist(rel_error, bins=50)
plt.show()

