from Reinforce_Agent import Reinforce_Agent
from timeit import default_timer as timer
from Utility import Utility
import pandas as pd


def create_and_train_reinforce_agent():
    agent = Reinforce_Agent()
    start = timer()
    agent.learn()
    end = timer()
    print('Execution time: {}'.format(end - start))
    #agent.plot_statistics_to_files()

if __name__ == '__main__':
    create_and_train_reinforce_agent()
    '''
    data = [ [[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]], [[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]], [[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]], [[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]]]
    pd = pd.DataFrame(columns = ['State'])
    pd.at[1,'State'] = ""
    for i in range(len(data)):
        for row in data[i]: 
            pd.at[i+1,'State'] += str(row)
            pd.at[i+1,'State'] += '\n'
    pd.to_csv(r'D:\Projects\1. Environment_2048\Environment\test.csv')
    '''
