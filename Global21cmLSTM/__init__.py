__version__ = "1.0"
__author__ = "Johnny Dorigo Jones"

from pathlib import Path

HERE = __file__[: -len("__init__.py")]
if not Path(HERE + "dataset_21cmGEM.h5").exists():
    import requests
    print("Downloading data set for 21cmGEM (Cohen et al. 2020) used in Dorigo Jones et al. 2024 and also Bye et al. 2022 and Bevins et al. 2021.")
    r = requests.get("https://zenodo.org/record/5084114/files/dataset_21cmVAE.h5?download=1")
    with open(HERE + "dataset_21cmGEM.h5", "wb") as f:
        f.write(r.content)

if not Path(HERE + "dataset_ARES.h5").exists():
    import requests
    print("Downloading data set for ARES used and generated in Dorigo Jones et al. 2024.")
    r = requests.get("https://zenodo.org/record/13840725/files/dataset_ARES.h5?download=1")
    with open(HERE + "dataset_ARES.h5", "wb") as f:
        f.write(r.content)

from Global21cmLSTM import emulator_21cmGEM
from Global21cmLSTM import emulator_ARES
from Global21cmLSTM import preprocess_21cmGEM
from Global21cmLSTM import preprocess_ARES
