import json, os
import pandas as pd
import numpy as np

class sport_digest(object):

    def __init__(self, path, gameID):
        self.ID = gameID
        self.keep_col = ['GameClock', 'Quarter', 'ShotClock', 'Position',
       'Clock', 'BallX', 'BallY', 'BallZ', 'NearRim', 'RimStart', 'LowStart', 'HighStart']
        with open(path + self.ID) as json_data:
            self.data = json.load(json_data)

    def unpack(self):
        events = self.data['events']
        moments = []
        cols1 = ['Quarter', 'EvID', 'GameClock', 'ShotClock', 'NoClue', 'Position']
        for event in events:
            moments += event['moments']
        Moments = pd.DataFrame(moments, columns=cols1)
        Moments['Clock'] = Moments['GameClock'].astype(int)
        Moments['Quarter'] = Moments['Quarter'].astype(int)

        ballMom = []
        for moment in Moments['Position']:
            ballMom.append(moment[0])

        cols2 = ['TeamID', 'PlayerID', 'BallX', 'BallY', 'BallZ']
        ballFrame = pd.DataFrame(ballMom, columns=cols2)

        ballFrame['TeamID'] = ballFrame['TeamID'].astype(str)
        ballFrame['PlayerID'] = ballFrame['PlayerID'].astype(str)

        self.Positions = pd.concat([Moments, ballFrame], axis = 1)
        self.Positions.drop_duplicates(subset=['Quarter', 'GameClock', 'ShotClock','BallX','BallY','BallZ'], inplace=True)
        self.Positions = self.Positions.sort_values(by=['Quarter', 'GameClock'], ascending=[True, False])
        self.Positions = self.Positions[self.Positions['TeamID'] == '-1']
        self.Positions.reset_index(inplace=True, drop=True)
        return self.Positions

    def feat_gen(self):
        self.Positions['Lower'] = (self.Positions['BallZ'].shift() > self.Positions['BallZ']).astype(int)
        newpos = self.Positions[['BallX', 'BallY']].values - np.array([47, 0])
        newpos = np.absolute(newpos)
        hdist = newpos - np.array([41.65, 25])
        hdist = np.sqrt((hdist ** 2).sum(axis = 1))
        self.Positions['NearRim'] = (hdist < 1).astype(int)

        self.Positions['NearRim'] += (self.Positions['BallZ'] > 9.9).astype(int)
        self.Positions['NearRim'] = (self.Positions['NearRim'] > 1).astype(int)
        self.Positions['RimStart'] = ((self.Positions['NearRim'] == 1) & (self.Positions['NearRim'].shift() != 1)).astype(int)  * self.Positions['Clock']
        self.Positions['LowStart'] = ((self.Positions['Lower'] == 1) & (self.Positions['Lower'].shift() != 1)).astype(int)  * self.Positions['Clock']
        self.Positions['HighStart'] = ((self.Positions['Lower'] == 0) & (self.Positions['Lower'].shift() != 0)).astype(int)  * self.Positions['Clock']

        self.Positions = self.Positions[self.keep_col]
        self.Positions.reset_index(inplace = True, drop = True)
        return self.Positions

    def run(self):
        self.unpack()
        self.feat_gen()

if __name__ == '__main__':
    import time
    t = time.time()
    path = './'
    gameID = '0021500492.json'
    sportvu = sport_digest(path, gameID)
    sportvu.run()
    print sportvu.Positions.info()
    print 'Runtime: ' + str(time.time() - t) + ' seconds'
