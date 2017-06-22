import pandas as pd
import datetime as dt
from math import *
import numpy as np
import requests, json, os

pd.options.mode.chained_assignment = None

def clocker(clock):
    """
    Turns clock (HH:MM or MM:SS format) into time in units.
    """
    m, s = clock.split(':')
    return 60 * int(m) + int(s)


class playbyplay(object):
    """
    This class is used to send a json query to fetch the correct Play By Play data
    from the NBA server.

    Inputs:
    gameID, string. Self explanatory. Sourced from filename of json corresponding to
                    SportVU data when run in main pipeline.
    """
    def __init__(self, gameID):
        """
        Stores ID, and notes that data has not been called yet.
        """
        self.ID = gameID
        self.g = 1

    def get(self):
        """
        Queries STATS server for game data.
        """
        headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:53.0) Gecko/20100101 Firefox/53.0'}
        params = {'EndPeriod': 10, 'GameID': self.ID, 'StartPeriod': 1}
        r = requests.get("http://stats.nba.com/stats/playbyplayv2", params=params, headers=headers)
        self.data = r.json()
        self.g = 0 #Signifier. Separated out for debugging purposes.
        return self.data

    def unpack(self):
        """
        Unpacks json data received from STATS to extract the full play by play
        and put it in a convertible format.
        """
        """
        #1.0 Grabs data if not acquired already.
        """

        if self.g:
            self.get()
            self.g = 0

        """
        #1.1 Creates DataFrame.
        """
        plays = self.data['resultSets']
        plays = plays[0]
        cols = plays['headers']
        self.playbyplay = pd.DataFrame(plays['rowSet'], columns= cols)

        """
        #1.2 Sets columns to keep, and columns to reformat.
        """
        keepers = ['EVENTMSGTYPE', 'PERIOD', 'PCTIMESTRING', 'HOMEDESCRIPTION', \
        'PLAYER1_ID', 'PLAYER1_NAME', 'PLAYER1_TEAM_ID', 'PLAYER1_TEAM_ABBREVIATION', \
        'PLAYER2_ID', 'PLAYER2_NAME', 'PLAYER2_TEAM_ID', 'PLAYER2_TEAM_ABBREVIATION', \
        'PLAYER3_ID', 'PLAYER3_NAME', 'PLAYER3_TEAM_ID', 'PLAYER3_TEAM_ABBREVIATION']
        strCols = ['PCTIMESTRING', 'HOMEDESCRIPTION', 'PLAYER1_ID', 'PLAYER1_NAME', \
        'PLAYER1_TEAM_ID', 'PLAYER1_TEAM_ABBREVIATION', 'PLAYER2_ID', 'PLAYER2_NAME', \
        'PLAYER2_TEAM_ID', 'PLAYER2_TEAM_ABBREVIATION', 'PLAYER3_ID', 'PLAYER3_NAME', \
        'PLAYER3_TEAM_ID', 'PLAYER3_TEAM_ABBREVIATION']
        intCols = ['EVENTMSGTYPE', 'PERIOD']

        """
        #1.3 Sets frame to correct columns, and converts columns to correct datatypes.
        """
        self.playbyplay = self.playbyplay[keepers]
        for col in strCols:
            self.playbyplay[col] = self.playbyplay[col].astype(str)
        for col in intCols:
            self.playbyplay[col] = self.playbyplay[col].astype(int)

        """
        #2.0 The Class now identifies needed plays, adds column for game clock, and
             separates out the events that are shots and rebounds.
        """
        """
        #2.1 Identifies rows where shots occur, and the appropriate rebound for
             each shot.
        """
        shots = self.playbyplay.index[self.playbyplay['EVENTMSGTYPE'] == 2]
        rebs = shots + 1

        """
        #2.2 Adds clock column and orders.
        """
        self.playbyplay['GameClock'] = self.playbyplay['PCTIMESTRING'].apply(clocker)
        self.playbyplay.sort_values(by = ['PERIOD', 'GameClock'], ascending = [True, False], inplace = True)

        """
        #2.3 Separates the correct rows into new frames.
        """
        self.Shots = self.playbyplay.iloc[shots,:]
        self.Rebs = self.playbyplay.iloc[rebs,:]
        self.Shots.reset_index(inplace = True, drop = True)
        self.Rebs.reset_index(inplace = True, drop = True)
        return

    def pairer(self):
        """
        #3.0 Creates a DataFrame with the information from a shot paired with the
        information from a rebound.
        """
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
        self.pairer()

if __name__ == '__main__':
    import time
    t = time.time()
    pbp = playbyplay('0021500492')
    pbp.run()
    print pbp.paired.info()
    print 'Runtime: ' + str(time.time() - t) + ' seconds'
