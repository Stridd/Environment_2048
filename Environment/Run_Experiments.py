from timeit import default_timer as timer

from agents.reinforce.reinforce_agent import Reinforce_Agent

from utilities import Utility
#import sys

#Source: https://www.quora.com/How-can-you-convert-milliseconds-in-a-format-with-days-hours-minutes-and-seconds-in-Python

def pretty_print_time(seconds):

    minutes, seconds = divmod(seconds, 60) 
    hours, minutes = divmod(minutes, 60) 	
    days, hours = divmod(hours, 24)  
    print('Execution time: | Hours:{:.0f} | Minutes:{:.0f} | Seconds:{:.0f} |'.format(hours, minutes, seconds)) 

def create_and_train_reinforce_agent():

    agent = Reinforce_Agent()
    start = timer()
    agent.learn()
    end = timer()
    pretty_print_time(end-start)
    agent.plot_statistics_to_files()

if __name__ == '__main__':
    create_and_train_reinforce_agent()
    