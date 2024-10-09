# 21cmLSTM

### Background
21cmLSTM (Dorigo Jones et al. 2024, or DJ+24) is a long short-term memory (LSTM) recurrent neural network (RNN) emulator of the global 21 cm signal that has unprecedented emulation accuracy compared to previous, fully connected neural network emulators. 21cmLSTM owes its accuracy to the unique ability of LSTM RNNs to learn patterns in sequential data; in other words, 21cmLSTM leverages the intrinsic correlation between neighboring frequency channels (i.e., autocorrelation) in the 21 cm signal. Given input astrophysical parameters, 21cmLSTM outputs, or predicts, a realization of the global 21 cm signal across redshifts z=5-50 (spanning the EoR to the Dark Ages).

### Performance
The average relative rms error of 21cmLSTM is 0.22% when trained and tested on the same simulated set of signals as used for previous emulators globalemu ([Bevins et al. 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.508.2923B/abstract), [GitHub](https://github.com/htjb/globalemu)) and 21cmVAE ([Bye et al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJ...930...79B/abstract), [GitHub](https://github.com/christianhbye/21cmVAE)); this data set was originally created for the 21cmGEM emulator ([Cohen et al. 2020](https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.4845C/abstract)) and the exact version that we used for 21cmLSTM was uploaded to Zenodo for 21cmVAE: https://doi.org/10.5281/zenodo.5084113. The average speed of 21cmLSTM is 46 ms when emulating one signal at a time and using GPU acceleration (see computational resources in Section 2.3 of DJ+24). In Section 4 of DJ+23, we used 21cmLSTM to fit mock data of the global 21 cm signal and showed that it can sufficiently exploit even outstandingly optimisic mesurements of the signal (5 mK noise) and obtain unbiased Bayesian posterior constraints. 21cmLSTM is thus useful for computationally expensive multi-parameter inference of upcoming 21 cm experiments, which may require jointly fitting various complementary data sets or summary statistics (e.g., [Dorigo Jones et al. 2023](https://ui.adsabs.harvard.edu/abs/2023ApJ...959...49D/abstract); [Breitman et al. 2024](https://ui.adsabs.harvard.edu/abs/2024MNRAS.527.9833B/abstract)).

### GitHub description
21cmLSTM is free to use on the MIT open source license. This GitHub contains the full code for training and running the emulator, including data preprocessing, network training, and model evaluation, to facilitate the community in retraining 21cmLSTM on other data sets and models. The sample notebooks are for training 21cmLSTM on the 21cmGEM data set used in DJ+24 (as well as for other emulators, see above), and also for evaluating 21cmLSTM trained on either the 21cmGEM or ARES model data sets. The instance of 21cmLSTM trained on 21cmGEM included in this repository is the same one used for Bayesian analyses in DJ+24 and has test set mean rms error of 0.20%, and the instance of 21cmLSTM trained on ARES is a representative trial with mean emulation error of 0.38%.

### Contact; papers to reference
Please reach out to me at johnny.dorigojones@colorado.edu about any comments, questions, or contributions (can also open an issue or make a pull request). Please reference Dorigo Jones et al. 2024 and provide a link to this GitHub repository if you utilize this work or emulator in any way, and [Dorigo Jones et al. 2023](https://ui.adsabs.harvard.edu/abs/2023ApJ...959...49D/abstract) regarding posterior emulation bias or if you perform joint fits using 21cmLSTM.

# Set-up
We recommend to pip install 21cmLSTM within a virtual environment containing the following dependencies: python>=3.6, tensorflow>=2.5, h5py, jupyter, matplotlib, numpy

```
git clone https://github.com/jdorigojones/21cmLSTM
cd 21cmLSTM
python -m pip install .
```

# Contributions
Authors: Johnny Dorigo Jones and Shah Bahauddin

Additional contributions from: Jordan Mirocha, David Rapetti, Christian H. Bye
