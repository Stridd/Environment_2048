from timeit import default_timer as timer

from agents.reinforce.reinforce_agent import Reinforce_Agent
#from agents.DDQN.DDQN import DDQN
from utilities import Utility


import numpy as np 
#Source: https://www.quora.com/How-can-you-convert-milliseconds-in-a-format-with-days-hours-minutes-and-seconds-in-Python

def pretty_print_time(seconds):

    minutes, seconds = divmod(seconds, 60) 
    hours, minutes = divmod(minutes, 60) 	
    days, hours = divmod(hours, 24)  
    print('Execution time: | Hours:{:.0f} | Minutes:{:.0f} | Seconds:{:.0f} |'.format(hours, minutes, seconds)) 

def create_and_train_reinforce_agent():

    agent = Reinforce_Agent()
    start = timer()
    agent.train()
    end = timer()
    pretty_print_time(end-start)
    agent.plot_statistics_to_files()

def create_and_train_DDQN_agent():
    agent = DDQN()
    start = timer()
    agent.train()
    end = timer()
    pretty_print_time(end-start)

if __name__ == '__main__':
    create_and_train_reinforce_agent()
    #create_and_train_DDQN_agent()
    #np.set_printoptions(suppress=True)
    #with open(,'r') as f:
    #text = np.loadtxt(r'D:\Projects\1. Environment_2048\Environment\logs\20-10-2020_06-35-18\experiment_data.txt',delimiter=',', dtype = np.float32)
    #print(text)
