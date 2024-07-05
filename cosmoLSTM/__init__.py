__version__ = "1.0"
__author__ = "Johnny Dorigo Jones"

from pathlib import Path

HERE = __file__[: -len("__init__.py")]
if not Path(HERE + "dataset_21cmLSTM.h5").exists():
    import requests

    print("Downloading 21cmGEM data set (see Cohen et al. 2020).")
    r = requests.get(
        "https://zenodo.org/record/5084114/files/dataset_21cmVAE.h5?download=1"
    )
    with open(HERE + "dataset_21cmLSTM.h5", "wb") as f:
        f.write(r.content)

from cosmoLSTM import emulator
from cosmoLSTM import preprocess
