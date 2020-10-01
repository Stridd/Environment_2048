from abc import ABC
import os
import glob
import cProfile, pstats, io
from pstats import SortKey

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

    @staticmethod
    def profile_function(function):

        time_of_profiling = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')

        pr = cProfile.Profile()
        pr.enable()

        function()

        pr.disable()
        s = io.StringIO()
        sortby = SortKey.TIME
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)

        current_directory = os.path.dirname(__file__)

        profiler_folder = current_directory + '\\' + Parameters.profiles_folder_name

        Utility.make_folder_if_not_exist(profiler_folder)

        profile_file_name = profiler_folder + '\\' + time_of_profiling + '.stats'

        ps.dump_stats(profile_file_name)