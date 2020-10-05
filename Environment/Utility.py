from abc import ABC
import os
import glob
import cProfile, pstats, io
from pstats import SortKey

from datetime import datetime 
from Enums import Optimizers,WeightInit
from Parameters import Parameters
import torch.optim as optim
import torch

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
    def get_optimizer_map():
        optimizers = {}
        optimizers[Optimizers.ADAM]     = Utility.get_adam
        optimizers[Optimizers.RMSPROP]  = Utility.get_RMSPROP
        optimizers[Optimizers.SGD]      = Utility.get_SGD
        optimizers[Optimizers.ADAGRAD]  = Utility.get_ADAGRAD
        optimizers[Optimizers.ADADELTA] = Utility.get_ADADELTA
        return optimizers

    @staticmethod
    def get_adam(params):
        return optim.Adam(params, lr = Parameters.lr)
    
    @staticmethod
    def get_RMSPROP(params):
        return optim.RMSprop(params, lr = Parameters.lr, momentum = Parameters.momentum)

    @staticmethod
    def get_SGD(params):
        return optim.SGD(params, lr = Parameters.lr, momentum=Parameters.momentum)
    
    @staticmethod
    def get_ADAGRAD(params):
        return optim.Adagrad(params, lr = Parameters.lr)

    @staticmethod
    def get_ADADELTA(params):
        return optim.Adadelta(params, r = Parameters.lr)

    @staticmethod
    def get_optimizer_for_parameters(parameters):
        optimizer = None
        optimizer_map = Utility.get_optimizer_map()
        optimizer = optimizer_map[Parameters.optimizer](parameters)

        del optimizer_map

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

    @staticmethod
    def get_weight_initialization_map():

        weight_initialization = {}
        weight_initialization[WeightInit.EYE]               = torch.nn.init.eye_
        weight_initialization[WeightInit.UNIFORM]           = torch.nn.init.uniform_
        weight_initialization[WeightInit.NORMAL]            = torch.nn.init.normal_
        weight_initialization[WeightInit.XAVIER_NORMAL]     = torch.nn.init.xavier_normal_
        weight_initialization[WeightInit.XAVIER_UNIFORM]    = torch.nn.init.xavier_uniform_
        weight_initialization[WeightInit.KAIMING_NORMAL]    = torch.nn.init.kaiming_normal_
        weight_initialization[WeightInit.KAIMING_UNIFORM]   = torch.nn.init.kaiming_uniform_

        return weight_initialization

    @staticmethod
    def get_initialization_function(nn_layer):

        if type(nn_layer) == torch.nn.Linear:
            weight_init_map = Utility.get_weight_initialization_map()
            weight_init_function = weight_init_map[Parameters.weight_init]
            weight_init_function(nn_layer.weight)
            nn_layer.bias.data.fill_(0.01)