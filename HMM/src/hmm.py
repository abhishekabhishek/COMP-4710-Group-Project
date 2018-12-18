# -*- coding: utf-8 -*-
#!/home/abhi/anaconda3/envs/dev1/bin/python

"""Implementation of the Hidden Markov Model ( HMM ) using hmmlearn."""

# Implementation of the Hidden Markov Model for a regresssive model for the agent

# Python 3 standard library modules
from os import path
import time

# Libraries and module documentations
# numpy - https://docs.scipy.org/doc/
# hmmlearn - https://hmmlearn.readthedocs.io/en/stable/

import numpy as np
from hmmlearn import hmm
from sklearn.externals import joblib

# Class for the HMM implementation
class HMM:
  
  # Initializer for the HMM module
  def __init__(self):
    
    """Initialize the class variables to be used for fitting the observed dataset."""
    
    # # Documentation - https://hmmlearn.readthedocs.io/en/latest/api.html#gaussianhmm
    self.__gaussian_hmm = hmm.GaussianHMM()
    
    # Documentation - https://hmmlearn.readthedocs.io/en/latest/api.html#gmmhmm
    self.__gaussian_mixture_hmm = hmm.GMMHMM()
    
    # Documentation - https://hmmlearn.readthedocs.io/en/latest/api.html#multinomialhmm
    self.__multinomial_hmm = hmm.MultinomialHMM()
    
    # Initialize the model array for modularly training and test the models
    self.__model_array = [self.__gaussian_hmm, self.__gaussian_mixture_hmm, self.__multinomial_hmm]
    
  # Estimate the model parameters for the specified model
  def fit_data(self, model_index, input_dataset, num_components_list, 
               n_iterations, model_save_dir, covariance_class, conv_tol,
               input_shape):
    
    """Estimate the parameters for the specified model using the input dataset."""
    model_to_estimate = self.__model_array[model_index]
    print('Estimating the parameters for the model with type: ' + str(type(model_to_estimate)))
    
    # Array to hold the multiple models trained using the training data
    trained_models_array_viterbi = []
    trained_models_array_map = []
    
    # Fit for the HMM with Gaussian emmissions for the observed states
    if(type(model_to_estimate) is hmm.GaussianHMM):
      
      # Iterate over the num_components list to generate parameters using several
      # different number of hidden states
      
      # Train models using the viterbi algorithm
      for num_components in num_components_list:
        model_to_estimate = hmm.GaussianHMM(n_components=num_components,
                                            covariance_type=covariance_class,
                                            algorithm='viterbi', 
                                            random_state=np.random.RandomState(seed=1),
                                            n_iter=n_iterations,
                                            verbose=True,params='stmc',
                                            init_params='',tol=conv_tol)
        
        try:
          save_path = str(model_save_dir + 'gaussian_hmm_viterbi_' + covariance_class 
                          + '_' + str(num_components) + '_' + str(n_iterations) + 
                          '_(' + str(input_shape[0]) + ',' + str(input_shape[1]) +').pkl')
          
          # Check if such model has already been trained
          if path.exists(save_path):
            
            print('A model has already been trained with the following'
                  ' parameters. Loading model from path:' +
                  save_path)
            
            trained_models_array_viterbi.append(joblib.load(save_path))
          else:
            print('Training model with parameters: ' + str(model_to_estimate))
            print('Note : The verbose output has the following format :'
                  ,'Column 1: Iteration, Column 2: ?, Column 3: Log probability score')
            
            start = time.time()
            model_to_estimate.fit(input_dataset)
            print('Time elapsed for training: %d seconds' %(int(time.time()-start)))
            
            trained_models_array_viterbi.append(model_to_estimate)
            joblib.dump(model_to_estimate, save_path)
            
        except ValueError:
          print('ValueError: Unable to train the model using %d components with Viterbi algorithm.' + 
                ' Trying other num_components listed.' %(num_components) )
          
          pass
      
      # Train models using the MAP algorithm
      '''
      for num_components in num_components_list:
        model_to_estimate = hmm.GaussianHMM(n_components=num_components,
                                            covariance_type=covariance_class,
                                            algorithm='map',
                                            random_state=np.random.RandomState(seed=1),
                                            n_iter=n_iterations,
                                            verbose=True,params='stmc',
                                            init_params='',tol=conv_tol)
        
        try:
          save_path = str(model_save_dir + 'gaussian_hmm_map_' + covariance_class 
                          + '_' + str(num_components) + '_' + str(n_iterations) + 
                          '_(' + str(input_shape[0]) + ',' + str(input_shape[1]) +').pkl')
          
          # Check if such model has already been trained
          if path.exists(save_path):
            
            print('A model has already been trained with the following'
                  ' parameters. Loading model from path:' +
                  save_path)
            
            trained_models_array_map.append(joblib.load(save_path))
          else:
            print('Training model with parameters: ' + str(model_to_estimate))
            print('Note : The verbose output has the following format :'
                  ,'Column 1: Iteration, Column 2: ?, Column 3: Log probability score')
            start = time.time()
            model_to_estimate.fit(input_dataset)
            print('Time elapsed for training: %d seconds' %(int(time.time()-start)))
            trained_models_array_map.append(model_to_estimate)
            joblib.dump(model_to_estimate, save_path)
            
        except ValueError:
          print('ValueError: Unable to train the model using %d components with MAP algorithm.' + 
                ' Trying other num_components listed.' %(num_components) )
          
          pass
    '''
    # Fit for the HMM with Gaussian mixture emmissions for the observed states
    if(type(model_to_estimate) is hmm.GMMHMM):
      pass

    # Fit for the HMM with Gaussian mixture emmissions for the observed states
    if(type(model_to_estimate) is hmm.MultinomialHMM):
      pass
      
    # Return the arrays contraining the trained models
    return trained_models_array_viterbi, trained_models_array_map
    
  # Method for calculating the log probability of a given sample
  def score_observation(trained_model, observation):
    return trained_model.score(observation)
    