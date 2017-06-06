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

    def get(self):
        headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:53.0) Gecko/20100101 Firefox/53.0'}
        params = {'EndPeriod': 10, 'GameID': self.ID, 'StartPeriod': 1}
        r = requests.get("http://stats.nba.com/stats/playbyplayv2", params=params, headers=headers)
        self.data = r.json()
        return self.data

    def unpack(self):
        self.get()
        plays = self.data['resultSets']
        plays = plays[0]
        cols = plays['headers']
        self.playbyplay = pd.DataFrame(plays['rowSet'], columns= cols)
        self.playbyplay['PCTIMESTRING'] = self.playbyplay['PCTIMESTRING'].astype(str)
        self.playbyplay['PLAYER1_TEAM_ID'] = self.playbyplay['PLAYER1_TEAM_ID'].astype(str)
        self.playbyplay['GameClock'] = self.playbyplay['PCTIMESTRING'].apply(clocker)
        self.shotsandrebs = self.playbyplay[(self.playbyplay['EVENTMSGTYPE'] == 2) | (self.playbyplay['EVENTMSGTYPE'] == 4)]
        self.shotsandrebs.reset_index(inplace=True, drop = True)
        self.shotsandrebs['ShotInd'] = (self.shotsandrebs['EVENTMSGTYPE'].shift() == 2).astype(int) * (self.shotsandrebs.index - 1)
        self.shotsandrebs['HomeShot'] = self.shotsandrebs['HOMEDESCRIPTION'].str.find('MISS') > -1
        self.shotsandrebs['HomeShot'].fillna(value=0, inplace=True)
        self.shotsandrebs['HomeReb'] = self.shotsandrebs['HOMEDESCRIPTION'].str.find('REBOUND') > -1
        self.shotsandrebs['HomeReb'].fillna(value=0, inplace=True)
        self.shotsandrebs['TeamR1'] = self.shotsandrebs['HOMEDESCRIPTION'].str.find('Rebound')
        self.shotsandrebs['TeamR2'] = self.shotsandrebs['VISITORDESCRIPTION'].str.find('Rebound')
        self.shotsandrebs['TeamR'] = (self.shotsandrebs['TeamR1'] > -1) | (self.shotsandrebs['TeamR2'] > -1)
        return self.shotsandrebs

    def isolate(self):
        self.Rebs = self.shotsandrebs[self.shotsandrebs['EVENTMSGTYPE'] == 4]
        self.Rebs = self.Rebs[self.Rebs['TeamR'] == False]
        self.Shots = self.shotsandrebs.iloc[self.Rebs['ShotInd']]
        self.Shots.reset_index(inplace=True, drop=True)
        self.Rebs.reset_index(inplace=True, drop=True)
        self.paired = pd.DataFrame({'Period': self.Shots.PERIOD,\
        'Clock': self.Shots.GameClock,\
        'ShootPlayerName': self.Shots.PLAYER1_NAME, 'ShootPlayerID': self.Shots.PLAYER1_ID,\
        'ShootTeamID': self.Shots.PLAYER1_TEAM_ID, 'ShootTeamName': self.Shots.PLAYER1_TEAM_NICKNAME,\
        'RebPlayerName': self.Rebs.PLAYER1_NAME, 'RebPlayerID': self.Rebs.PLAYER1_ID})
        self.paired['ShootTeamID'] = self.paired['ShootTeamID'].str[:-2]
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
    print pbp.paired
    print 'Runtime: ' + str(time.time() - t) + ' seconds'
