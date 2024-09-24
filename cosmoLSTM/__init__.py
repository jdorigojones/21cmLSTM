__version__ = "1.0"
__author__ = "Johnny Dorigo Jones"

from pathlib import Path

HERE = __file__[: -len("__init__.py")]
if not Path(HERE + "dataset_21cmGEM_VAE.h5").exists():
    import requests

    print("Downloading data set for 21cmGEM (Cohen et al. 2020) used in Dorigo Jones et al. 2024 and also Bye et al. 2022 and Bevins et al. 2021.")
    r = requests.get(
        "https://zenodo.org/record/5084114/files/dataset_21cmVAE.h5?download=1"
    )
    with open(HERE + "dataset_21cmGEM_VAE.h5", "wb") as f:
        f.write(r.content)

from cosmoLSTM import emulator
from cosmoLSTM import preprocess
