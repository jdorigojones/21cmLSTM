# 21cmLSTM

21cmLSTM is a long short-term memory (LSTM) recurrent neural network (RNN) emulator of the global 21 cm signal that takes advantage of the intrinsic spatio-temporal correlation between frequency/redshift bins in 21 cm data. Given an input of 7 (or 8) astrophysical parameters, 21cmLSTM outputs (or predicts) a realization of the global 21 cm signal across redshifts z=5-50. The average relative rms error of 21cmLSTM is 0.22% and the average evaluation time is only ~0.6 ms when trained and tested on the “21cmGEM” data set described in [Cohen et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.4845C/abstract) and also used in the original papers for [globalemu](https://ui.adsabs.harvard.edu/abs/2021MNRAS.508.2923B/abstract) and [21cmVAE](https://ui.adsabs.harvard.edu/abs/2022ApJ...930...79B/abstract) ([GitHub](https://github.com/christianhbye/21cmVAE)). These advantages of memory-based emulation of the global 21 cm signal will be useful for computationally expensive parameter estimation of upcoming measurements. The emulator is described in detail in Dorigo Jones et al. 2024.

21cmLSTM is free to use on the MIT open source license. Here we provide the full code for training and running the emulator, including data preprocessing, network training, and model evaluation, to facilitate others in retraining 21cmLSTM on other data sets and models. The sample notebooks are for training 21cmLSTM on the 21cmGEM data set used in the paper, and for evaluating the best trial of 21cmLSTM discussed in the paper.

Please feel free to email me at johnny.dorigojones@colorado.edu about any questions, comments, improvements, or contributions (or open an issue, make pull request). Please reference Dorigo Jones et al. 2024 and provide a link to this GitHub repository if you utilize this work or emulator in any way.

# Set-up
Dependencies: python>=3.6, tensorflow>=2.5, h5py, jupyter, matplotlib, numpy, tqdm

It is recommended to install 21cmLSTM in a virtual environment with all required dependencies:
```
git clone https://github.com/johnnydorigojones/21cmLSTM.git
cd 21cmLSTM
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

# Contributions
Authors: Johnny Dorigo Jones and Shah Bahauddin
