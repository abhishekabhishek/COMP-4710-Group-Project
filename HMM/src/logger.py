# -*- coding: utf-8 -*-
#!/home/abhi/anaconda3/envs/dev1/bin/python

# System library for handling the I/O for this class
import sys

# Logging mechanism for generating the logs within the script
class Logger(object):
  
  #----------------------------------------------------------------------------
  # __init__() 
  # PURPOSE : Initializer for the Logger class
  # PARAMETERS : File name to log the output of the program
  # RETURNS : None
  # OUTPUT : None 
  #----------------------------------------------------------------------------
  def __init__(self, filename):
    self.terminal = sys.stdout
    self.log = open(filename, "w")
      
  #----------------------------------------------------------------------------
  # write() 
  # PURPOSE : Output buffer for the Logging class
  # PARAMETERS : String to write to the log file for the program
  # RETURNS : None
  # OUTPUT : Writes the String passed as parameter to the log file 
  #----------------------------------------------------------------------------
  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)
  
  #----------------------------------------------------------------------------
  # flush() 
  # PURPOSE : Allow the system library to handle the buffer for printing to the
  #           log file
  # PARAMETERS : None
  # RETURNS : None
  # OUTPUT : Passes buffer handling to the standard Python 3.6 library
  #----------------------------------------------------------------------------
  def flush(self):
    pass
