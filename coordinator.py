import pandas as pd
import numpy as np
import PbP_extractor
from collections import Counter
from SportVU_extractor import sport_digest
import time

def findRim(df, t, q):
    """
    Based on time signature, finds the point where ball starts to approach rim.
    """
    selection = df[(df['Quarter'] == q) & (df['RimStart'] >= t)]
    try:
        return selection.iloc[-1]
    except Exception:
        return []

def rimErr(df, t, q):
    """
    If the previous function does not work properly, due to various aspects of
    the game, returns a row that can be used as proxy.
    """
    selection = df[(df['Quarter'] == q) & (df['Clock'] >= t)]
    try:
        return selection.iloc[-1]
    except Exception:
        return []

def findShot(df, srs):
    """
    Based on the row returned by previous functions, uses the time signature to
    find the shot's row. Based on the time signature of when the ball started
    rising.
    """
    selection = df[(df['Quarter'] == srs['Quarter']) & (df['HighStart'] >= srs['Clock']) & (df['Loc'] < srs['Loc'])]
    if selection.shape[0]:
        try:
            return selection.iloc[-1]
        except Exception:
            return []
    else:
        return []


def features(df):
    """
    More complicated feature engineering, last step in the processing of each shot.
    """
    """
    #1.0 Sets offense and defnse, then sorts.
    """
    df['Off'] = df['TeamID'] == df['ShootTeamID']
    df.sort_values(by = 'Off', inplace = True)
    """
    #2.0 Separates numeric columns for #math operations.
    """
    numcols = ['pre_PlayerX', 'pre_PlayerY', 'pre_HDist', 'pre_Angle',
               'pos_PlayerX', 'pos_PlayerY', 'pos_HDist', 'pos_Angle', 'pre_GameClock', 'pos_GameClock']
    """
    #2.1 Separates out shooter specifically for comparison columns.
    """
    shooter = df[df['PlayerID'] == df['ShootPlayerID']][numcols].values
    if shooter.shape[0]: #removes erroneously identified shots
        """
        #3.0 Computes various numeric stats including how they move, their
             cosine similarity to the shooter, and also how they are boxing out.
        """
        nums = df[numcols].values
        closer = ((df['pre_HDist'] < df['pos_HDist']).astype(int) - 0.5) * 2
        move = nums[:,[0,1]] - nums[:,[4,5]]
        df['MoveV'] = np.sqrt((move ** 2).sum(axis = 1)) * closer
        df['pre_CosSim'] = np.cos(nums[:,3] - shooter[:,3])
        df['pos_CosSim'] = np.cos(nums[:,7] - shooter[:,7])
        df['DT'] = df['pre_GameClock'] - df['pos_GameClock']
        arrpre = df[['pre_PlayerX','pre_PlayerY']].values
        arrpos = df[['pos_PlayerX','pos_PlayerY']].values
        df['pre_Box'] = boxgen(arrpre)
        df['pos_Box'] = boxgen(arrpos)
        df['IsShoot'] = (df['PlayerID'] == df['ShootPlayerID']).astype(int)
        df['Rebounder'] = df['PlayerID'] == df['RebPlayerID']
        if df['Rebounder'].sum(): #filters shots with no rebounder
            return df
        else:
            return []
    else:
        return []
def boxgen(arr):
    """
    Uses a single iteration of K-means to identify who is boxing out whom.
    Basically, uses one team as sample centroids, and then looks at the number
    of elements in a centroid to determine how many players one player is boxing
    out. Ideally will also add column that indicates how many are closer to the
    hoop.
    """
    bdist = np.sqrt(((arr - np.array([41.65,25])) ** 2).sum(axis = 1)).reshape(10,1)
    arr = np.concatenate((arr, bdist), axis = 1)
    arr1 = arr[:5, :]
    arr2 = arr[5:,:]
    dists = np.array([np.sqrt(((arr1[:,:2] - row)**2).sum(axis = 1)) for row in arr2[:,:2]])
    d = np.argmin(dists, axis = 1)
    o = np.argmin(dists, axis = 0)
    dbox = [np.sum(o == i) for i in xrange(5)]
    obox = [np.sum(d == i) for i in xrange(5)]
    boxes = np.array(dbox + obox)
    return boxes

class Coordination(object):

    def __init__(self, gameID, path):
        self.jsonID = gameID
        self.pbpID = gameID[:-5] #strips '.json'
        self.path = path

    def playbyplay(self):
        self.pbpEX = PbP_extractor.playbyplay(self.pbpID)
        self.pbpEX.run()

    def sportvu(self):
        self.sportEX = sport_digest(self.path, self.jsonID)
        self.sportEX.run()

    def rimshots(self):
        """
        Uses clock from Play by Play to identify when the ball hits the rim, and
        then uses that as a reference point to search back and find when the
        shot started.
        """
        paired = self.pbpEX.paired
        Positions = self.sportEX.Positions
        Positions['Loc'] = Positions.index #isolate for consistency, avoids problems with .iloc and indexing
        rims = {}
        prev = set() #avoids repeats
        for i, row in paired.iterrows():
            t = row['Clock']
            q = row['Period']
            r = findRim(Positions, t, q)
            if len(r): #makes sure shot is findable, due to problems with game clock
                    l = r['Loc'].astype(str)
                    if l not in prev:
                        rims[str(i)] = r
                        prev.update([l])
                    else:
                        r = rimErr(Positions, t, q)
                        rims[str(i)] = r

        shots = {}
        prev = set()
        for key, value in rims.iteritems():
            r = findShot(Positions, value)
            if len(r): #same as above, game log will sometimes be incomplete.
                l = r['Loc'].astype(str)
                if l not in prev:
                    shots[key] = r
                    prev.update([l])

        pnum = shots.keys()
        pnum = np.array(pnum).astype(int)
        pnum.sort()
        pnum = np.array(pnum).astype(str)
        rebs = [rims[shot] for shot in pnum]
        ups = [shots[shot] for shot in pnum]
        ShotDF = pd.DataFrame(ups, index = pnum)
        RebDF = pd.DataFrame(rebs, index = pnum)
        keep_col = ['GameClock', 'Quarter', 'Position']
        ShotDF = ShotDF[keep_col]
        """
        Creates separate pre/post labeling
        """
        ShotDF.columns = ['pre_' + col for col in keep_col]
        RebDF = RebDF[keep_col]
        RebDF.columns = ['pos_' + col for col in keep_col]
        RebDF.head()
        pnum = np.array(pnum).astype(int)
        paired = paired.iloc[list(pnum),:]
        snr = ShotDF.join(RebDF)
        snr.reset_index(inplace=True, drop = True)
        self.PSPaired = paired.join(snr)
        arr = self.PSPaired[['GameID', 'RebPlayerID', 'ShootPlayerID','Period', 'Clock']].values
        a = arr[:,0]
        for i in xrange(1,5):
            a += arr[:,i].astype(str)
        self.PSPaired['IDNum'] = a
        self.PSPaired.drop_duplicates(subset = 'IDNum', inplace = True)
        self.PSPaired.dropna(inplace=True)

        self.PSPaired.reset_index(inplace = True, drop = True)

    def obConstructor(self):
        self.rowDict = {}
        hoop = np.array([41.65, 25])
        cols = ['TeamID', 'PlayerID', 'PlayerX', 'PlayerY', 'PlayerZ']
        tostr = cols[:2]
        rowcols = ['Clock', 'Period', 'RebPlayerID', 'RebPlayerName', 'ShootPlayerID', \
                   'ShootPlayerName', 'ShootTeamID', 'ShootTeamName', \
                   'pre_GameClock', 'pos_GameClock', 'IDNum']
        df = self.PSPaired
        pDict = self.sportEX.playerDict

        df['Loc'] = df.index

        for i, row in df.iterrows():
            dfs = [pd.DataFrame(row['pre_Position'], columns=cols), pd.DataFrame(row['pos_Position'], columns=cols)]
            for frame in dfs:
                frame['PlayerX'] = np.absolute(frame['PlayerX'].values - 47)
                diff = (frame[['PlayerX', 'PlayerY']].values - hoop)
                frame['HDist'] = np.sqrt((diff ** 2).sum(axis = 1))
                frame['Angle'] = np.arctan2(diff[:,0], diff[:,1])
                frame.pop('PlayerZ')
                for col in tostr:
                    frame[col] = frame[col].astype(str)

            ucols = frame.columns.values[2:]
            cols2 = [tostr + ['pre_' + col for col in ucols], tostr + ['pos_' + col for col in ucols]]
            dfs[0].columns = cols2[0]
            dfs[1].columns = cols2[1]
            dfs[1].pop('TeamID')
            new = dfs[0].set_index('PlayerID').join(dfs[1].set_index('PlayerID'))
            new.reset_index(inplace=True, drop = False)

            if new.shape[0] == 11:
                for col in rowcols:
                    new[col] = np.array(row[col]).repeat(11)

                new['ShotID'] = np.array(row['IDNum']).repeat(11) + new['PlayerID']
                new['Role'] = new['PlayerID'].map(pDict)
                new.sort_values(by = 'Role', inplace = True)
                new.reset_index(inplace=True, drop = True)
                struct = features(new[new['PlayerID'] != '-1'])
                if len(struct):
                    self.rowDict[i] = struct

        self.gameFrame = pd.concat(self.rowDict.values(), ignore_index=True)
        self.gameFrame.reset_index(inplace = True, drop = True)
        # self.gameFrame.to_csv('gamframetest.csv')


    def run(self):
        self.playbyplay()
        # print 'pbp done'
        self.sportvu()
        # print 'sportvu done'
        self.rimshots()
        # print 'rimshots done'
        self.obConstructor()
        # print 'constructor done'
        # pass

if __name__ == '__main__':
    import time, os
    import random
    t = time.time()
    path = './'
    game = '0021500492.json'
    coo = Coordination(game, path)
    coo.run()
    print 'Runtime: ' + str(time.time() - t) + ' seconds'
