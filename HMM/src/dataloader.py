# -*- coding: utf-8 -*-

"""Python module to create the input datasets for the HMM

This module contains the data formatting and pre-processing tools for
formatting and preparing both training and testing data to use to fit 
the model to the observed data and then test for anomalies."""

# Library and module documentations :
# pandas - https://pandas.pydata.org/pandas-docs/stable/
# numpy - https://docs.scipy.org/doc/
# csv - https://docs.python.org/3.6/library/csv.html
# os.path - https://docs.python.org/3.6/library/os.path.html
# time - https://docs.python.org/3.6/library/time.html
# datetime - https://docs.python.org/3.6/library/datetime.html

# Standard python library modules
from os import path
import time
from datetime import datetime
import copy

# Data formatting and Pre-processing modules
import pandas as pd
import numpy as np

class DataLoader:
  
  #----------------------------------------------------------------------------
  # load_formatted()
  # PURPOSE : Load the formatted datasets from the saved csv files and contruct
  #           the dataframes 
  # PARAMETERS : Path to the csv file to read
  # RETURNS : Training and Testing dataframes
  # OUTPUT : Time taken to read in the dataframes, Preview of the dataframes 
  #          constructed from the csv files
  #----------------------------------------------------------------------------
  @staticmethod
  def load_formatted(train_file_path, test_file_path):
    train_df_local = pd.read_csv(train_file_path, sep=',', 
                                 dtype={'timestamp':np.float32, 'sensor_id':str})
    test_df_local = pd.read_csv(test_file_path, sep=',', 
                                 dtype={'timestamp':np.float32, 'sensor_id':str})
    
    print('Dimensions of the training datafame: ' + str(train_df_local.shape))
    print('Dimensions of the test datafame: ' + str(test_df_local.shape))
    
    
    return train_df_local, test_df_local
  
  #----------------------------------------------------------------------------
  # load_raw()
  # PURPOSE : Load the raw datasets from the saved csv files and contruct
  #           the dataframes 
  # PARAMETERS : Path to the csv file to read
  # RETURNS : Training and Testing dataframes
  # OUTPUT : Time taken to construct the dataframes, Preview of the dataframes 
  #          constructed from the csv files, Svaes the dataframes constructed
  #          on the disk
  #----------------------------------------------------------------------------
  @staticmethod
  def load_raw(train_file_path, train_save_file_path, test_file_path, 
               test_save_file_path, col_dict):
    
    start_time = time.time()
    
    if not path.exists(train_file_path):
      print("Error. Path %s does not exist." % train_file_path)
      raise BaseException
      
    if not path.exists(test_file_path):
      print("Error. Path %s does not exist." % test_file_path)
      raise BaseException
    
    # Get the details of the columns to be read from the csv file
    col_ids = list(col_dict.keys())
    col_indices = list(col_dict.values())
    
    col_indices_test = [i+1 for i in col_indices]
    
    train_df_raw = pd.read_csv(train_file_path, sep=',', usecols=col_indices,
                              engine='python', names=col_ids, skiprows=1)
    test_df_raw = pd.read_csv(test_file_path, sep=',', usecols=col_indices_test,
                              engine='python', names=col_ids, skiprows=1)
    
    train_df_fmt = DataLoader.fmt_df(train_df_raw)
    test_df_fmt = DataLoader.fmt_df(test_df_raw)
      
    train_df_fmt.to_csv(train_save_file_path)
    test_df_fmt.to_csv(test_save_file_path)
    
    end_time = time.time()
    print('Time elapsed to read and format the training and testing datasets: %f seconds' 
          %(end_time-start_time))
    
    return train_df_fmt, test_df_fmt
  
  #----------------------------------------------------------------------------
  # fmt_df()
  # PURPOSE : Format the dataframe i.e. construct the timestamp values from the
  #           string date and time values in the dataframe
  # PARAMETERS : Dataframe to format
  # RETURNS : Formatted dataframe
  # OUTPUT : Time taken to format the dataframes, Preview of the dataframes 
  #          constructed from the csv files.
  #----------------------------------------------------------------------------
  @staticmethod
  def fmt_df(input_df):
    
    # Initialize the array for storing POSIX timestamps
    timestamps = []
    
    # Extract the series from the dataframe
    date = input_df['date'].values
    time_collected = input_df['time'].values
    
    # Transform the datetime into POSIX datetime
    for i in range(len(date)):
    
      # Collect the logged time value
      timestamp_logged = date[i]+'T'+time_collected[i]
    
      # If the time collected is not in the right format, reformat it
   
      # If the timestamp is missing milliseconds, add zeros to maintain formatting consistency
      if '.' not in timestamp_logged:
          timestamp_logged = timestamp_logged + '.000000'
        
      # If the timestamp is missing some of the digits in the milliseconds, add zeros to maintain formatting 
      # consistency
      if(len(timestamp_logged[timestamp_logged.find('.')+1:len(timestamp_logged)]) < 6):
          timestamp_logged = timestamp_logged[0:timestamp_logged.find('.')]
          timestamp_logged = timestamp_logged + '.000000'
    
      timestamps.append(timestamp_logged[:-3]+'Z')
    
      try:
          timestamps[i] = datetime.strptime(timestamps[i], '%Y-%m-%dT%H:%M:%S.%fZ')
          timestamps[i] = time.mktime(timestamps[i].timetuple())
      except ValueError:
        pass
      
    input_df['timestamp'] = timestamps
    input_df.dropna(axis=0, how='any', inplace=True)
    
    filter1 = input_df['sensor_state'] == 'ON'
    filter2 = input_df['sensor_state'] == 'OPEN'

    formatted_df = input_df.where(filter1 | filter2).dropna().reset_index()
    formatted_df.drop(columns=['index','date', 'time', 'sensor_state'], inplace=True)
    
    return formatted_df
  
  # Get all of the unique sensors given in the dataset
  @staticmethod
  def get_unique_sensors(sensor_id, model_path, test):
    
    # List to hold the unique door and motion sensor ids
    unique_motion_sensors = []
    unique_door_sensors = []
      
    if test:
      unique_sensors_path = model_path[:model_path.rfind('/')]+'/unique_test_sensors.csv'
    else:
      unique_sensors_path = model_path[:model_path.rfind('/')]+'/unique_train_sensors.csv'
    
    if path.exists(unique_sensors_path):
      unique_sensors_df = pd.read_csv(unique_sensors_path, sep=',')
      unique_sensors = np.sort(unique_sensors_df['sensor_id'].values)
      
      for sensor_id in unique_sensors:
        if sensor_id[0] is 'D':
          unique_door_sensors.append(sensor_id)
        
        if sensor_id[0] is 'M':
          unique_motion_sensors.append(sensor_id)
      
    else:
      # Performing analysis of the dataframe
      for i in range(len(sensor_id)):
      
        if(len(sensor_id[i]) > 0):
        
          if sensor_id[i] not in unique_motion_sensors and sensor_id[i][0] is 'M':
              unique_motion_sensors.append(sensor_id[i])
            
          if sensor_id[i] not in unique_door_sensors and sensor_id[i][0] is 'D':
              unique_door_sensors.append(sensor_id[i])
            
      unique_sensors = []
      unique_sensors.extend(unique_door_sensors)
      unique_sensors.extend(unique_motion_sensors)
      sensor_id_dataframe = pd.DataFrame.from_dict({'sensor_id':unique_sensors})
      sensor_id_dataframe.to_csv(unique_sensors_path)

    return unique_motion_sensors, unique_door_sensors

  # Build the dictionaries to map the sensor values to real valued numbbers
  @staticmethod
  def build_maps(unique_motion_sensors, unique_door_sensors):
    # First we need to map the sensor ID to integers in the dataframe

    # Create a new sensor map with a more intuitive mapping with the constraint mentioned above
    transition_sensor_map = {}
    inverse_transition_sensor_map = {}

    # If the spatial sensor is a door sensor, first sort the door sensor ID array
    unique_door_sensors = pd.Series(unique_door_sensors)
    unique_door_sensors.sort_values(ascending=True, inplace=True)
    unique_door_sensors = unique_door_sensors.values

    for i in range(len(unique_door_sensors)):
      transition_sensor_map[unique_door_sensors[i]] = i
      inverse_transition_sensor_map[i] = unique_door_sensors[i]
    
    # If the spatial sensor is a motion sensor
    for i in range(len(unique_motion_sensors)):
      if unique_motion_sensors[i][0] == 'M':
        transition_sensor_map[unique_motion_sensors[i]] = int(unique_motion_sensors[i][len(unique_motion_sensors[i])-2:len(unique_motion_sensors[i])])-1+len(unique_door_sensors)
        inverse_transition_sensor_map[int(unique_motion_sensors[i][len(unique_motion_sensors[i])-2:len(unique_motion_sensors[i])])-1+len(unique_door_sensors)] = unique_motion_sensors[i]
        
    return transition_sensor_map, inverse_transition_sensor_map
  
  # Build the dictionaries to map the test sensor values to real values
  def build_test_maps(training_map, inverse_training_map, 
                      test_motion_sensors, test_door_sensors):
    
    # Get the list of sensors already in the training dataset
    training_ids = list(training_map.keys())
    
    # Get the list of sensors not in the training dataset
    test_ids = []
    
    # Motion sensors in the test data not in training dataset
    for sensor_id in test_motion_sensors:
      if sensor_id not in training_ids:
        test_ids.append(sensor_id)
        
    # Door sensors in the test data not in the training dataset
    for sensor_id in test_door_sensors:
      if sensor_id not in training_ids:
        test_ids.append(sensor_id)
        
    # Initialize the maps for the test dataset
    test_map = copy.deepcopy(training_map)
    inverse_test_map = copy.deepcopy(inverse_training_map)

    # Add the remaining ids to the maps
    for i in range(len(test_ids)):
      test_map[test_ids[i]] = len(training_ids) + i
      inverse_test_map[len(training_ids) + i] = test_ids[i]
      
    return test_map, inverse_test_map
    
    
      