import coordinator
import os, time
import cPickle as pickle
import numpy as np
import pandas as pd


class massunpack(object):

    def __init__(self, path, rand = -1):
        self.rand = rand
        self.path = path
        self.files = os.listdir(path)
        self.gameDict = {}
        self.frames = []
    def run(self, size = 400):
        self.runtime = time.time()
        if self.rand == -1:
            self.list = ['0021500492.json']
        else:
            if self.rand:
                self.list = list(np.random.choice(self.files, size))
            else:
                self.list = list(self.files[:size])
        for game in self.list:
            print 'Start unpacking {}'.format(game[:-5])
            t = time.time()
            coo = coordinator.Coordination(game, self.path)
            coo.run()
            self.gameDict[game[:-5]] = coo.rowDict
            try:
                self.frames.append(coo.gameFrame)
            except Exception:
                pass
            self.FinalFrame = pd.concat(self.frames, ignore_index = True)
            print "{} has finished unpacking in {} seconds.".format(coo.pbpID, str(time.time() - t))
        pickle.dump((self.gameDict, self.FinalFrame), open('gameinfo.pkl', 'wb'))
        self.run = time.time()
        self.rt = int(self.run - self.runtime)

        print 'Job took {} hours, {} minutes, and {} seconds.'.format(self.rt/3600, (self.rt/60)%60, self.rt%3600)
    def write(self):
        self.FinalFrame.to_csv('big_ass_frame.csv')

if __name__ == '__main__':
    # r = time.time()
    path = './data/games/'
    start = massunpack(path, rand = 1)
    start.run()
    # t = time.time()
    # rt = int(t - r)
    # print 'Job took {} hours, {} minutes, and {} seconds.'.format(rt/3600, (rt%60)/60, rt%3600)
    start.write()
