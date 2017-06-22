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
        self.Coos = {}
        self.runtime = time.time()
        self.wins = 0
        self.losses = 0
        self.check =[]
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
            try:
                coo.run()
                self.gameDict[game[:-5]] = coo.rowDict
                print 'Shape of {} with a dict size of {}.'.format(coo.gameFrame.shape, len(coo.rowDict))
                self.frames.append(coo.gameFrame)
                self.wins +=1
                if not self.wins%100:
                    print '{} games run.'.format(self.wins)
                # coo.gameFrame.to_csv(self.pbpID +'.csv')
                print "{} has finished unpacking in {} seconds.".format(coo.pbpID, str(time.time() - t))
            except Exception:
                print "There was an error unpacking {}".format(coo.pbpID)
                self.losses+=1
                self.check.append(coo.pbpID)
                pass
            # pickle.dump(coo, open('/media/nymy2/Data/Pickles/' + game + '.pkl', 'wb'))

            # self.Coos[game] = coo
            # print "{} has finished unpacking in {} seconds.".format(coo.pbpID, str(time.time() - t))
        self.FinalFrame = pd.concat(self.frames, ignore_index = True)

        self.run = time.time()
        self.rt = int(self.run - self.runtime)

        # print 'Job took {} hours, {} minutes, and {} seconds.'.format(self.rt/3600, (self.rt/60)%60, self.rt%3600)
    def write(self):

        self.FinalFrame.to_csv('big_ass_frame.csv')
        pickle.dump(self.gameDict, open('gameinfo.pkl', 'wb'))

if __name__ == '__main__':
    # r = time.time()
    path = './data/games/'
    start = massunpack(path, rand = 1)
    start.run(400)
    # t = time.time()
    # rt = int(t - r)
    # print 'Job took {} hours, {} minutes, and {} seconds.'.format(rt/3600, (rt%60)/60, rt%3600)
    start.write()
