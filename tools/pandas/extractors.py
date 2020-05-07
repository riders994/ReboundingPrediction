import os
import json
import requests
import numpy as np
import pandas as pd
from nba_api.stats.endpoints.playbyplayv2 import PlayByPlayV2

POSITION_MAP = {'G': 1.0, 'G-F': 5.0 / 3, 'F-G': 7.0 / 3, 'F': 3.0, 'F-C': 11.0 / 3, 'C-F': 13.0 / 3, 'C': 5.0}


def clocker(clock):
    """
    Turns clock (HH:MM or MM:SS format) into time in units.
    """
    m, s = clock.split(':')
    return 60 * int(m) + int(s)


class SportVu:
    data = None
    keep_col = [
        'GameClock', 'Quarter', 'ShotClock', 'Position', 'Clock', 'BallX', 'BallY', 'BallZ', 'NearRim',
        'RimStart', 'LowStart', 'HighStart'
    ]
    playerDict = dict()
    results = dict()
    current = None

    def __init__(self, path='./'):
        self.path = path

    def load(self, game_id):
        with open(os.path.join(self.path, game_id + '.json')) as json_data:
            self.data = json.load(json_data)
        self.current = game_id

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
        moment_cols = ['Quarter', 'EvID', 'GameClock', 'ShotClock', 'NoClue', 'Position']
        for i, event in enumerate(events):
            moments += event['moments']
            if i == 0:
                # Used later to totally order on court position
                # Done on a per-game basis due to roster volatility
                vis = event['visitor']
                home = event['home']
        moment_df = pd.DataFrame(moments, columns=moment_cols)
        moment_df['Clock'] = moment_df['GameClock'].astype(int)
        moment_df['Quarter'] = moment_df['Quarter'].astype(int)
        """
        #3.0 Construct DataFrame of ball positions to attach to Moments.
             Ideally will find a way to reduce size.
        """
        ball_moment = []
        for moment in moment_df['Position']:
            ball_moment.append(moment[0])
        ball_cols = ['TeamID', 'PlayerID', 'BallX', 'BallY', 'BallZ']
        balldf = pd.DataFrame(ball_moment, columns=ball_cols)
        balldf['TeamID'] = balldf['TeamID'].astype(str)
        balldf['PlayerID'] = balldf['PlayerID'].astype(str)

        """
        #4.0 Creates dictionary for player positions to be used when coordinating.
        """
        for player in (home['players'] + vis['players']):
            self.playerDict[str(player['playerid'])] = POSITION_MAP[player['position']]

        """
        #5.0 Creates final frame of Positions to be used in coordination script.
        """
        positions = pd.concat([moment_df, balldf], axis=1)
        positions.drop_duplicates(subset=['Quarter', 'GameClock', 'ShotClock', 'BallX', 'BallY', 'BallZ'], inplace=True)
        positions = positions.sort_values(by=['Quarter', 'GameClock'], ascending=[True, False])
        positions = positions[positions['TeamID'] == '-1']
        positions.reset_index(inplace=True, drop=True)
        self.results.update({self.current: self.feat_gen(positions)})

    def feat_gen(self, position_df):
        """
        Generates some base features that are used when coordinating.
        """
        """
        Identifies when the ball is approaching the rim, starting or
        starting to rise. This is used to identify when shtos and rebounds
        actually start.
        """
        """
        #1.0 Identify when ball is lower than in previous position.
        """
        position_df['Lower'] = (position_df['BallZ'].shift() > position_df['BallZ']).astype(int)
        """
        #2.0 Changes the balls dimensions to half court. Not stored in final
             DataFrame because ball x,y,z is not used.
        """
        newpos = position_df[['BallX', 'BallY']].values - np.array([47, 0])
        newpos = np.absolute(newpos)
        hdist = newpos - np.array([41.65, 25])  # Location of hoop
        hdist = np.sqrt((hdist ** 2).sum(axis=1))
        position_df['NearRim'] = (hdist < 1).astype(int)  # is the ball within a foot of the hoop?
        position_df['NearRim'] += (position_df['BallZ'] > 9.9).astype(int)  # is it also above (or almost) the rim?
        position_df['NearRim'] = (position_df['NearRim'] > 1).astype(int)
        """
        #3.0 Creates a time signature for when ball starts to be near rim,
             starts going up, or starts going down.
        """
        position_df['RimStart'] = ((position_df['NearRim'] == 1) & (position_df['NearRim'].shift() != 1)).astype(int) *\
                                  position_df['Clock']
        position_df['LowStart'] = ((position_df['Lower'] == 1) & (position_df['Lower'].shift() != 1)).astype(int) *\
                                  position_df['Clock']
        position_df['HighStart'] = ((position_df['Lower'] == 0) & (position_df['Lower'].shift() != 0)).astype(int) * \
                                   position_df['Clock']

        position_df = position_df[self.keep_col]
        position_df.reset_index(inplace=True, drop=True)
        return position_df

    def run(self, game_id=None):
        if game_id:
            self.load(game_id)
        if not self.data:
            raise ValueError
        self.unpack()


class PlayByPlay:

    rcv = False
    results = dict()

    @staticmethod
    def pairer(shots, rebs, game_id):
        """
        #3.0 Creates a DataFrame with the information from a shot paired with the
        information from a rebound.
        """
        paired = pd.DataFrame({
            'Period': shots.PERIOD,
            'Clock': shots.GameClock,
            'ShootPlayerName': shots.PLAYER1_NAME,
            'ShootPlayerID': shots.PLAYER1_ID,
            'ShootTeamID': shots.PLAYER1_TEAM_ID,
            'ShootTeamName': shots.PLAYER1_TEAM_ABBREVIATION,
            'RebPlayerName': rebs.PLAYER1_NAME,
            'RebPlayerID': rebs.PLAYER1_ID,
            'RebTeamID': rebs.PLAYER1_TEAM_ID,
            'RebTeamName': rebs.PLAYER1_TEAM_ABBREVIATION,
            'GameID': np.repeat(game_id, shots.shape[0])
        })
        paired['ShootTeamID'] = paired['ShootTeamID'].str[:-2]
        paired['RebTeamID'] = paired['RebTeamID'].str[:-2]
        paired.reset_index(inplace=True, drop=True)
        return paired

    def unpack(self, playbyplay, game_id):
        """
        Unpacks json data received from STATS to extract the full play by play
        and put it in a convertible format.
        """
        """
        1.0 Set up DFs
        """

        """
        #1.1 Sets columns to keep, and columns to reformat.
        """
        keepers = [
            'EVENTMSGTYPE', 'PERIOD', 'PCTIMESTRING', 'HOMEDESCRIPTION', 'PLAYER1_ID', 'PLAYER1_NAME', 'PLAYER1_TEAM_ID'
            , 'PLAYER1_TEAM_ABBREVIATION', 'PLAYER2_ID', 'PLAYER2_NAME', 'PLAYER2_TEAM_ID', 'PLAYER2_TEAM_ABBREVIATION',
            'PLAYER3_ID', 'PLAYER3_NAME', 'PLAYER3_TEAM_ID', 'PLAYER3_TEAM_ABBREVIATION'
        ]
        str_cols = [
            'PCTIMESTRING', 'HOMEDESCRIPTION', 'PLAYER1_ID', 'PLAYER1_NAME', 'PLAYER1_TEAM_ID',
            'PLAYER1_TEAM_ABBREVIATION', 'PLAYER2_ID', 'PLAYER2_NAME', 'PLAYER2_TEAM_ID', 'PLAYER2_TEAM_ABBREVIATION',
            'PLAYER3_ID', 'PLAYER3_NAME', 'PLAYER3_TEAM_ID', 'PLAYER3_TEAM_ABBREVIATION'
        ]
        int_cols = ['EVENTMSGTYPE', 'PERIOD']

        """
        #1.2 Sets frame to correct columns, and converts columns to correct datatypes.
        """
        playbyplay = playbyplay[keepers]
        for col in str_cols:
            playbyplay[col] = playbyplay[col].astype(str)
        for col in int_cols:
            playbyplay[col] = playbyplay[col].astype(int)

        """
        #2.0 The Class now identifies needed plays, adds column for game clock, and
             separates out the events that are shots and rebounds.
        """
        """
        #2.1 Identifies rows where shots occur, and the appropriate rebound for
             each shot.
        """
        shots = playbyplay.index[playbyplay['EVENTMSGTYPE'] == 2]
        rebs = shots + 1

        """
        #2.2 Adds clock column and orders.
        """
        playbyplay['GameClock'] = playbyplay['PCTIMESTRING'].apply(clocker)
        playbyplay.sort_values(by=['PERIOD', 'GameClock'], ascending=[True, False], inplace=True)

        """
        #2.3 Separates the correct rows into new frames.
        """
        shots = playbyplay.iloc[shots, :]
        rebs = playbyplay.iloc[rebs, :]
        shots.reset_index(inplace=True, drop=True)
        rebs.reset_index(inplace=True, drop=True)
        paired = self.pairer(shots, rebs, game_id)
        return paired

    @staticmethod
    def _get_pbp(game_id):
        """
        Queries STATS server for game data.
        """
        if not game_id:
            raise ValueError
        params = {'game_id': game_id, 'start_period': 1, 'end_period': 10}
        pbp = PlayByPlayV2(**params)
        return pbp.play_by_play.get_data_frame()

    def run(self, game_id):
        pbp = self._get_pbp(game_id)
        self.results.update({game_id: self.unpack(pbp, game_id)})



if __name__ == '__main__':
    import time

    t = time.time()
    gameID = '0021500492'
    sportvu = SportVu()
    sportvu.load(gameID)
    sportvu.run()
    print(sportvu.results[gameID].info())
    print('Runtime: ' + str(time.time() - t) + ' seconds')
