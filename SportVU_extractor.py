import json, os
import pandas as pd
import numpy as np

class sport_digest(object):
    """
    This class is used to take a SportVU json and convert it. It extracts all of
    the positional data (organized by the ball's position), and creates a big
    DataFrame that can be paired with the play by play data.

    Inputs:
    gameID, string. Self explanatory. Pulled from filename in main pipeline.
    path, string. Path to json file to be unpacked.
    """

    def __init__(self, path, gameID):
        """
        Stores ID, reads json, and sets up final columns.
        """
        self.ID = gameID
        self.keep_col = ['GameClock', 'Quarter', 'ShotClock', 'Position',
       'Clock', 'BallX', 'BallY', 'BallZ', 'NearRim', 'RimStart', 'LowStart', 'HighStart']
        with open(path + self.ID) as json_data:
            self.data = json.load(json_data)

    def unpack(self):
        """
        Unpacks the json file and formats it to the initial DataFrame.
        """
        """
        #1.0 Grabs relevant data.
        """
        events = self.data['events']
        moments = []
        """
        #2.0 Constructing first DataFrame of each 'moment'.
             Position column retains list of all player positions at that moment.
        """
        cols1 = ['Quarter', 'EvID', 'GameClock', 'ShotClock', 'NoClue', 'Position']
        for i, event in enumerate(events):
            moments += event['moments']
            if i == 0:
                # Used later to totally order on court position
                # Done on a per-game basis due to roster volatility
                vis = event['visitor']
                home = event['home']
        Moments = pd.DataFrame(moments, columns=cols1)
        Moments['Clock'] = Moments['GameClock'].astype(int)
        Moments['Quarter'] = Moments['Quarter'].astype(int)

        """
        #3.0 Construct DataFrame of ball positions to attach to Moments.
             Ideally will find a way to reduce size.
        """
        ballMom = []
        for moment in Moments['Position']:
            ballMom.append(moment[0])

        cols2 = ['TeamID', 'PlayerID', 'BallX', 'BallY', 'BallZ']
        ballFrame = pd.DataFrame(ballMom, columns=cols2)
        ballFrame['TeamID'] = ballFrame['TeamID'].astype(str)
        ballFrame['PlayerID'] = ballFrame['PlayerID'].astype(str)

        """
        #4.0 Creates dictionary for player positions to be used when coordinating.
        """
        posDict = {'G': 1.0,'G-F':5.0/3, 'F-G':7.0/3 , 'F': 3.0, 'F-C': 11.0/3,'C-F':13.0/3, 'C': 5.0}
        self.playerDict = {}
        for player in (home['players'] + vis['players']):
            self.playerDict[str(player['playerid'])] = posDict[player['position']]

        """
        #5.0 Creates final frame of Positions to be used in coordination script.
        """
        self.Positions = pd.concat([Moments, ballFrame], axis = 1)
        self.Positions.drop_duplicates(subset=['Quarter', 'GameClock', 'ShotClock','BallX','BallY','BallZ'], inplace=True)
        self.Positions = self.Positions.sort_values(by=['Quarter', 'GameClock'], ascending=[True, False])
        self.Positions = self.Positions[self.Positions['TeamID'] == '-1']
        self.Positions.reset_index(inplace=True, drop=True)
        return self.Positions

    def feat_gen(self):
        """
        Generates some base features that are used when coordinating.
        """
        """
        #1.0 Identifies when the ball is approaching the rim, starting or
             starting to rise. This is used to identify when shtos and rebounds
             actually start.
        """
        """
        #2.0 Identify when ball is lower than in previous position.
        """
        self.Positions['Lower'] = (self.Positions['BallZ'].shift() > self.Positions['BallZ']).astype(int)
        """
        #3.0 Changes the balls dimensions to half court. Not stored in final
             DataFrame because ball x,y,z is not used.
        """
        newpos = self.Positions[['BallX', 'BallY']].values - np.array([47, 0])
        newpos = np.absolute(newpos)
        hdist = newpos - np.array([41.65, 25]) #Location of hoop
        hdist = np.sqrt((hdist ** 2).sum(axis = 1))
        self.Positions['NearRim'] = (hdist < 1).astype(int) #is the ball within a foot of the hoop?
        self.Positions['NearRim'] += (self.Positions['BallZ'] > 9.9).astype(int) #is it also above (or almost) the rim?
        self.Positions['NearRim'] = (self.Positions['NearRim'] > 1).astype(int)
        """
        #4.0 Creates a time signature for when ball starts to be near rim,
             starts going up, or starts going down.
        """
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
