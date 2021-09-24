import torch
import numpy as np


class TimeSeriesLoader():
    def __init__(self, task, root = 'data/'):
        """ Initiate data loading for each task
        """
    
        if task == 'forecasting':
            # Returns load and renewable energy forecasting data
        
        elif task == 'classification':
            # Returns event detection, classification and localization data
            
        elif task == 'generation':
            # Returns PMU stream data
            
        else:
            raise Exception
            
        
    def load():
        