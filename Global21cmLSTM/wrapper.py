'''
Name: 21cmLSTM/Global21cmLSTM/wrapper.py
Author: Johnny Dorigo Jones
Original: May 2022, Edited: November 2025 to include classes for predicting both beam-weighted foreground
Description: A model class which wraps around the 21cmLSTM module for signal model evaluations
'''
from __future__ import division
import time
import os
import numpy as np
from pylinex import LoadableModel
import Global21cmLSTM as Global21cmLSTM
from Global21cmLSTM.eval_foreground import evaluate_foreground

try:
	# this runs with no issues in python 2 but raises error in python 3
	basestring
except:
	# this try/except allows for python 2/3 compatible string type checking
	basestring = str

#PATH = f"{os.environ.get('AUX_DIR', os.environ.get('HOME'))}/.Global21cmLSTM/"
PATH = '/projects/jodo2960/beam_weighted_foreground/'
model_save_path_foreground = PATH+"models/emulator_foreground_beam_meansub_21cmLSTM_3layer.h5" #emulator_foreground_beam_10regions_meansub_21cmLSTM_3layer_5to25MHz.h5

class predict_foreground(LoadableModel):
	def __init__(self, parameters, model_path=model_save_path_foreground):
		'''
  		parameters: np.ndarray
    			list of parameters to accept as input
       		model_path : str
	 		The path to the saved 21cmLSTM model instance
    			Default: '/projects/jodo2960/beam_weighted_foreground/models/emulator_foreground_beam_meansub_21cmLSTM_3layer.h5'
       		'''
		self.parameters = parameters
		self.model_path = model_path
	
	@property
	def parameters(self):
		"""
  		Property storing an array of parameters for this model
    		"""
		return self._parameters
	
	@parameters.setter
	def parameters(self, value):
		"""
  		Setter for the array of parameters for this model
    		value: array of parameters to give to the evaluate_on_foreground.__call__() function
      		"""
		self._parameters = [element for element in value]
	
	@property
	def neural_network_predictor(self):
		if not hasattr(self, '_neural_network_predictor'):
			self._neural_network_predictor = evaluate_foreground(model_path=self.model_path)
		return self._neural_network_predictor
	
	def __call__(self, parameters):
		'''
  		parameters: np.ndarray
    			list of parameters to accept as input
       		'''
		signal = self.neural_network_predictor(parameters)
		return signal
