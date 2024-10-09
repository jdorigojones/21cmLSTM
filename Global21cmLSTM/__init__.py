#!/usr/bin/env python
# coding: utf-8

# In[ ]:


__version__ = "1.0"
__author__ = "Johnny Dorigo Jones"

import requests
import os
from pathlib import Path
from Global21cmLSTM import __path__

from Global21cmLSTM import emulator_21cmGEM
from Global21cmLSTM import emulator_ARES
from Global21cmLSTM import preprocess_21cmGEM
from Global21cmLSTM import preprocess_ARES

