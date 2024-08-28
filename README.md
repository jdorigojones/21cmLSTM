# 21cmLSTM

### Background
21cmLSTM (Dorigo Jones et al. 2024, or DJ+24) is a long short-term memory (LSTM) recurrent neural network (RNN) emulator of the global 21 cm signal. 21cmLSTM has unprecedented emulation accuracy because of the unique ability of LSTM RNNs to learn patterns in sequential data, allowing the emulator to leverage the intrinsic correlation between adjacent frequency bins (i.e., autocorrelation) in the 21 cm signal. Given input astrophysical parameters, 21cmLSTM outputs, or predicts, a realization of the global 21 cm signal across redshifts z=5-50 (spanning the EoR to the Dark Ages).

### Performance
The average relative rms error of 21cmLSTM is 0.22% when trained and tested on the same data as used for previous emulators globalemu ([Bevins et al. 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.508.2923B/abstract), [GitHub](https://github.com/htjb/globalemu)) and 21cmVAE ([Bye et al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJ...930...79B/abstract), [GitHub](https://github.com/christianhbye/21cmVAE)), which was originally created for the 21cmGEM emulator ([Cohen et al. 2020](https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.4845C/abstract)). The average speed of 21cmLSTM is 46 ms when emulating one signal at a time and using GPU acceleration (see computational resources in Section 2.3 of DJ+24). In Section 4 of DJ+23, we used 21cmLSTM to fit mock data of the global 21 cm signal and showed that it can sufficiently exploit even outstandingly optimisic mesurements of the signal and obtain unbiased Bayesian posterior constraints. 21cmLSTM is thus useful for computationally expensive multi-parameter inference of upcoming 21 cm experiments, which may require jointly fitting various complementary data sets or summary statistics (e.g., Dorigo Jones et al. 2023; Breitman et al. 2024).

### GitHub description and papers to reference
21cmLSTM is free to use on the MIT open source license. This GitHub contains the full code for training and running the emulator, including data preprocessing, network training, and model evaluation, to facilitate the community in retraining 21cmLSTM on other data sets and models. The sample notebooks are for training 21cmLSTM on the 21cmGEM data set used in DJ+24 (as well as for other emulators, see above), and for evaluating a representative trial of 21cmLSTM (i.e., the same one used for multinest parameter estimation analyses in DJ+24).

Please feel free to contact me at johnny.dorigojones@colorado.edu about comments, questions, or contributions (can also open an issue or make a pull request). Please reference Dorigo Jones et al. 2024 and provide a link to this GitHub repository if you utilize this work or emulator in any way, and Dorigo Jones et al. 2023 if you perform joint fits using 21cmLSTM.

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
