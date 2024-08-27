# 21cmLSTM

21cmLSTM (Dorigo Jones et al. 2024) is a long short-term memory (LSTM) recurrent neural network (RNN) emulator of the global 21 cm signal that takes advantage of the intrinsic spatio-temporal correlation between frequency/redshift bins in 21 cm data. Given an input of 7 (or 8) astrophysical parameters, 21cmLSTM outputs (i.e., predicts) a realization of the global 21 cm signal across redshifts z=5-50. The average relative rms error of 21cmLSTM is 0.22% when trained and tested on the same data sets as previous emulators [globalemu](https://ui.adsabs.harvard.edu/abs/2021MNRAS.508.2923B/abstract) ([GitHub](https://github.com/htjb/globalemu)) and [21cmVAE](https://ui.adsabs.harvard.edu/abs/2022ApJ...930...79B/abstract) ([GitHub](https://github.com/christianhbye/21cmVAE)) (i.e., the “21cmGEM” data set described in [Cohen et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.4845C/abstract)). The average speed of 21cmLSTM is 46 ms when emulating one signal at a time and using the computational resources stated in Section 2.3 of Dorigo Jones et al. 2024. The unique ability of LSTM RNNs to learn patterns in temporal data is why 21cmLSTM achieves unprecedented emulation accuracy: the emulator exploits the intrinsic correlation between adjacent frequency bins (i.e., autocorrelation) in the 21 cm signal. 21cmLSTM can sufficiently exploit and constrain even outstandingly optimisic mesurements of the signal (Section 4 of DJ+24), showing that 21cmLSTM is useful for computationally expensive parameter estimation performed by upcoming 21 cm measurements, which may require jointly fitting various complementary data sets or summary statistics (e.g., Dorigo Jones et al. 2023).

21cmLSTM is free to use on the MIT open source license. This GitHub contains the full code for training and running the emulator, including data preprocessing, network training, and model evaluation, to facilitate the community in retraining 21cmLSTM on other data sets and models. The sample notebooks are for training 21cmLSTM on the 21cmGEM data set used in the paper, and for evaluating the best trial of 21cmLSTM discussed in the paper.

Please feel free to email me at johnny.dorigojones@colorado.edu about any questions, comments, improvements, or contributions (or open an issue or make a pull request). Please reference Dorigo Jones et al. 2024 and provide a link to this GitHub repository if you utilize this work or emulator in any way, and Dorigo Jones et al. 2023 if you perform joint fits using 21cmLSTM.

# Set-up
We recommend to pip install 21cmLSTM within a virtual environment containing the following dependencies: python>=3.6, tensorflow>=2.5, h5py, jupyter, matplotlib, numpy

```
git clone https://github.com/jdorigojones/21cmLSTM
cd 21cmLSTM
python -m pip install .
```

# Contributions
Authors: Johnny Dorigo Jones and Shah Bahauddin

Additional contributions from: David Rapetti, Christian H. Bye (provided 21cmGEM data sets)
