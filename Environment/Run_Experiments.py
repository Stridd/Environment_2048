from Reinforce_Agent import Reinforce_Agent
import cProfile, pstats, io
from pstats import SortKey

def create_and_train_reinforce_agent():
    agent = Reinforce_Agent()
    agent.learn()
    agent.plot_statistics_to_files()

if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()
    create_and_train_reinforce_agent()
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.dump_stats(r'D:\Projects\1. Environment_2048\Environment\profiler.stats')

    out_stream = open(r'D:\Projects\1. Environment_2048\Environment\log.txt', 'w')
    ps = pstats.Stats(r'D:\Projects\1. Environment_2048\Environment\profiler.stats', stream=out_stream)
    ps.strip_dirs().sort_stats('cumulative').print_stats()