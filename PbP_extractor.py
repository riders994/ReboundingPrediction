import pandas as pd
import datetime as dt
from math import *
import numpy as np
import requests, json, os

pd.options.mode.chained_assignment = None

def clocker(clock):
    m, s = clock.split(':')
    return 60 * int(m) + int(s)

class playbyplay(object):

    def __init__(self, gameID):
        self.ID = gameID
        self.g = 1

    def get(self):
        headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:53.0) Gecko/20100101 Firefox/53.0'}
        params = {'EndPeriod': 10, 'GameID': self.ID, 'StartPeriod': 1}
        r = requests.get("http://stats.nba.com/stats/playbyplayv2", params=params, headers=headers)
        self.data = r.json()
        self.g = 0
        return self.data

    def unpack(self):
        if self.g:
            self.get()
            self.g = 0
        plays = self.data['resultSets']
        plays = plays[0]
        cols = plays['headers']
        self.playbyplay = pd.DataFrame(plays['rowSet'], columns= cols)
        keepers = ['EVENTMSGTYPE', 'PERIOD', 'PCTIMESTRING', 'HOMEDESCRIPTION', 'PLAYER1_ID', 'PLAYER1_NAME', 'PLAYER1_TEAM_ID', 'PLAYER1_TEAM_ABBREVIATION', 'PLAYER2_ID', 'PLAYER2_NAME', 'PLAYER2_TEAM_ID', 'PLAYER2_TEAM_ABBREVIATION', 'PLAYER3_ID', 'PLAYER3_NAME', 'PLAYER3_TEAM_ID', 'PLAYER3_TEAM_ABBREVIATION']

        strCols = ['PCTIMESTRING', 'HOMEDESCRIPTION', 'PLAYER1_ID', 'PLAYER1_NAME', 'PLAYER1_TEAM_ID', 'PLAYER1_TEAM_ABBREVIATION', 'PLAYER2_ID', 'PLAYER2_NAME', 'PLAYER2_TEAM_ID', 'PLAYER2_TEAM_ABBREVIATION', 'PLAYER3_ID', 'PLAYER3_NAME', 'PLAYER3_TEAM_ID', 'PLAYER3_TEAM_ABBREVIATION']

        intCols = ['EVENTMSGTYPE', 'PERIOD']
        self.playbyplay = self.playbyplay[keepers]

        for col in strCols:
            self.playbyplay[col] = self.playbyplay[col].astype(str)

        for col in intCols:
            self.playbyplay[col] = self.playbyplay[col].astype(int)

        shots = self.playbyplay.index[self.playbyplay['EVENTMSGTYPE'] == 2]
        rebs = shots + 1

        self.playbyplay['GameClock'] = self.playbyplay['PCTIMESTRING'].apply(clocker)
        self.playbyplay.sort_values(by = ['PERIOD', 'GameClock'], ascending = [True, False], inplace = True)
        self.playbyplay['HomeShot'] = self.playbyplay['HOMEDESCRIPTION'].str.find('MISS') > -1
        self.playbyplay['HomeShot'].fillna(value=False, inplace=True)
        self.playbyplay['HomeReb'] = self.playbyplay['HOMEDESCRIPTION'].str.find('REBOUND') > -1
        self.playbyplay['HomeReb'].fillna(value=False, inplace=True)
        self.Shots = self.playbyplay.iloc[shots,:]
        self.Rebs = self.playbyplay.iloc[rebs,:]
        self.Shots.reset_index(inplace = True, drop = True)
        self.Rebs.reset_index(inplace = True, drop = True)
        return

    def isolate(self):
        self.paired = pd.DataFrame({'Period': self.Shots.PERIOD,\
        'Clock': self.Shots.GameClock,\
        'ShootPlayerName': self.Shots.PLAYER1_NAME, 'ShootPlayerID': self.Shots.PLAYER1_ID,\
        'ShootTeamID': self.Shots.PLAYER1_TEAM_ID, 'ShootTeamName': self.Shots.PLAYER1_TEAM_ABBREVIATION,\
        'RebPlayerName': self.Rebs.PLAYER1_NAME, 'RebPlayerID': self.Rebs.PLAYER1_ID, \
        'RebTeamID': self.Rebs.PLAYER1_TEAM_ID, 'RebTeamName': self.Rebs.PLAYER1_TEAM_ABBREVIATION, \
        'GameID': np.repeat(self.ID, self.Shots.shape[0])})
        self.paired['ShootTeamID'] = self.paired['ShootTeamID'].str[:-2]
        self.paired['RebTeamID'] = self.paired['RebTeamID'].str[:-2]
        self.paired.reset_index(inplace = True, drop = True)
        return self.paired

    def run(self):
        self.unpack()
        self.isolate()

if __name__ == '__main__':
    import time
    t = time.time()
    pbp = playbyplay('0021500492')
    pbp.run()
    print pbp.paired.info()
    print 'Runtime: ' + str(time.time() - t) + ' seconds'
