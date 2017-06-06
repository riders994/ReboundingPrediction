import json, os
import pandas as pd
import numpy as np

class sport_digest(object):

    def __init__(self, path, gameID):
        self.ID = gameID
        self.keep_col = ['Quarter', 'ShotClock', 'Position',
       'rClock', 'Ballx', 'Bally', 'Height', 'Close', 'NearRim', 'RimStart', 'LowStart', 'HighStart']
        with open(path + self.ID) as json_data:
            self.data = json.load(json_data)

    def unpack(self):
        e = self.data['events']
        E = []
        for event in e:
            E += event['moments']
        self.events = np.array(E)

        self.Events = pd.DataFrame(self.events, columns = ['Quarter', 'EvID', 'GameClock', 'ShotClock', 'NoClue', 'Position'])
        self.Events['rClock'] = self.Events['GameClock'].astype(int)
        self.Events['Quarter'] = self.Events['Quarter'].astype(int)

        self.ball = np.array([event[0] for event in self.Events['Position']])
        self.balldf = pd.DataFrame(self.ball, index = self.Events.index, columns=['Teamid','Playerid','Ballx','Bally','Height'])
        self.balldf.Teamid = self.balldf.Teamid.astype(int).astype(str)
        self.balldf.Playerid = self.balldf.Playerid.astype(int).astype(str)
        self.balldf.Ballx = np.absolute(self.balldf['Ballx'].values - 47)
        self.Positions = pd.concat([self.Events, self.balldf], axis = 1)

        self.Positions.drop_duplicates(subset=['Quarter', 'GameClock', 'ShotClock','Ballx','Bally','Height'], inplace=True)

        self.Positions = self.Positions.sort_values(by=['Quarter', 'GameClock'], ascending=[True, False])
        self.Positions.reset_index(inplace=True, drop=True)
        return self.Positions

    def feat_gen(self):
        heights = self.Positions['Height'].values
        pre = heights[:-1]
        post = heights[1:]
        lower = np.concatenate((np.array([False]), (pre > post)))
        self.Positions['Lower'] = lower.astype(int)
        self.Positions['High'] = (self.Positions['Height'] > 10).astype(int)

        self.ball = self.Positions[['Ballx', 'Bally']].values

        hoop = np.absolute(self.ball - np.array([41.65, 25]))
        hoop = (hoop ** 2).sum(axis = 1)
        close = hoop < 1
        
        self.Positions['Close'] = close.astype(int)

        self.Positions['NearRim'] = ((self.Positions['Close'] + self.Positions['High']) == 2).astype(int)
        self.Positions['RimStart'] = ((self.Positions['NearRim'] == 1) & (self.Positions['NearRim'].shift() != 1)).astype(int)  * self.Positions['rClock']
        self.Positions['LowStart'] = ((self.Positions['Lower'] == 1) & (self.Positions['Lower'].shift() != 1)).astype(int)  * self.Positions['rClock']
        self.Positions['HighStart'] = ((self.Positions['Lower'] == 0) & (self.Positions['Lower'].shift() != 0)).astype(int)  * self.Positions['rClock']
        self.Positions = self.Positions[self.keep_col]
        self.Positions.reset_index(inplace = True, drop = True)
        return self.Positions

    def run(self):
        self.unpack()
        self.feat_gen()

if __name__ == '__main__':
    import time
    t = time.time()
    path = './data/games/'
    gameID = '0021500492.json'
    sportvu = sport_digest(path, gameID)
    sportvu.run()
    print sportvu.Positions.head(10)
    print 'Runtime: ' + str(time.time() - t) + ' seconds'
