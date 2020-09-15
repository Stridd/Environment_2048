from abc import ABC
import numpy as np
import pandas as pd
import os
import glob
from datetime import datetime 
from Enums import Optimizers
from Parameters import Parameters
import torch.optim as optim

class Utility(ABC):

    @staticmethod
    def clean_folder(path):
        folder_path = path + '\\' + '*'
        old_files = glob.glob(folder_path)
        for old_file in old_files:
            os.remove(old_file)

    @staticmethod
    def make_folder_if_not_exist(path):
        if not os.path.isdir(path):
            os.mkdir(path)

    @staticmethod
    def get_time_of_experiment():
        time_of_experiment = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        return time_of_experiment

   

    @staticmethod
    def get_optimizer_for_parameters(parameters):
        optimizer = None
        if Parameters.optimizer == Optimizers.ADAM:
            optimizer = optim.Adam(parameters, lr = Parameters.lr)
        elif Parameters.optimizer == Optimizers.RMSPROP:
            optimizer = optim.RMSprop(parameters, lr = Parameters.lr, momentum = Parameters.momentum)
        elif Parameters.optimizer == Optimizers.SGD:
            optimizer = optim.SGD(parameters, momentum = Parameters.momentum)

        return optimizer