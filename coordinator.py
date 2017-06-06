import pandas as pd
import numpy as np
import PbP_extractor
from SportVU_extractor import sport_digest
import time

def findRim(df, t, q):
    rowSearch = df[((df['RimStart'] >= t) & (df['Quarter'] == q))]
    times = np.where(rowSearch['RimStart'] >= t)[0]
    if rowSearch.shape[0]:
        row = rowSearch.iloc[times.max(),]
    else:
        row = df[(df['rClock'] == t) & (df['Quarter'] == q)].index
        if len(row):
            return max(row), t
        else:
            return max(df.index), t
    if row['rClock'] - t > 5:
        return row.name, t
    else:
        return row.name, row['rClock']

def findShot(df, r, q):
    rowSearch = df[df['Quarter'] == q].iloc[:r[0],]
    times = np.where(rowSearch['HighStart'] >= r[1])[0]
    if rowSearch.shape[0]:
        if len(times):
            row = rowSearch.iloc[times.max(),]
        else:
            return max(df.index), r[1]
    else:
        row = df[(df['rClock'] == r[1]) & (df['Quarter'] == q)].index
        if len(row):
            return max(row), r[1]
        else:
            return max(df.index), r[1]
    if row['rClock'] - r[1] > 5:
        return row.name, r[1]
    else:
        return row.name, row['rClock']


def features(df):
    df['Off'] = df['TeamID'] == df['ShootTeamID']
    df.sort_values(by = ['TeamID', 'Off'], inplace = True)
    numcols = ['pre_PlayerX', 'pre_PlayerY', 'pre_HDist', 'pre_Angle', \
               'post_PlayerX', 'post_PlayerY', 'post_HDist', 'post_Angle', 'pre_ShotClock', 'post_ShotClock']
    shooter = df[df['PlayerID'] == df['ShootPlayerID'].astype(str)][numcols].values
    nums = df[numcols].values
    closer = ((df['pre_HDist'] < df['post_HDist']).astype(int) - 0.5) * 2
    move = nums[:,[0,1]] - nums[:,[4,5]]
    df['MoveV'] = np.sqrt((move ** 2).sum(axis = 1)) * closer
    df['pre_CosSim'] = np.cos(nums[:,3] - shooter[:,3])
    df['post_CosSim'] = np.cos(nums[:,7] - shooter[:,7])
    df['DT'] = df['pre_ShotClock'] - df['post_ShotClock']
    arrpre = df[['pre_PlayerX','pre_PlayerY']].values
    arrpos = df[['post_PlayerX','post_PlayerY']].values
    return df

def boxgen(arr):
    from collections import Counter
    bdist = np.sqrt(((arr - np.array([5.35,25])) ** 2).sum(axis = 1)).reshape(10,1)
    arr = np.concatenate((arr, bdist), axis = 1)
    arr1 = arr[:5, :]
    arr2 = arr[5:,:]
    dists = np.array([np.sqrt(((arr1[:,:2] - row)**2).sum(axis = 1)) for row in arr2[:,:2]])
    o = np.argmin(dists, axis = 1)
    d = np.argmin(dists, axis = 0)
    obox = [np.sum(d == i) for i in xrange(5)]
    dbox = [np.sum(o == i) for i in xrange(5)]
    boxes = np.array(obox + dbox)
    return boxes

class Coordination(object):

    def __init__(self, gameID, path):
        self.jsonID = gameID
        self.pbpID = gameID[:-5]
        self.path = path

    def playbyplay(self):
        self.pbpEX = PbP_extractor.playbyplay(self.pbpID)
        self.pbpEX.run()

    def sportvu(self):
        self.sportEX = sport_digest(self.path, self.jsonID)
        self.sportEX.run()

    def rimshots(self):
        paired = self.pbpEX.paired
        Positions = self.sportEX.Positions
        rims = np.array([findRim(Positions, shot[1]['Clock'], shot[1]['Period']) for shot in paired.iterrows()])
        # print rims
        shots = np.array([findShot(Positions, rims[i], shot[1]['Period']) for i, shot in enumerate(paired.iterrows())])
        rimRows = Positions.iloc[rims[:,0],]
        rimRows.reset_index(inplace=True, drop=True)
        shotRows = Positions.iloc[shots[:,0],]
        shotRows.reset_index(inplace=True, drop=True)
        keep_col = ['ShotClock', 'Position', 'rClock']
        shotRows = shotRows[keep_col]
        rimRows = rimRows[keep_col]
        shotRows['ShotClock'] = shotRows['ShotClock'].astype(float)
        rimRows['ShotClock'] = rimRows['ShotClock'].astype(float)
        rimRows.columns = ['post_'+col for col in rimRows.columns]
        shotRows.columns = ['pre_'+col for col in shotRows.columns]
        srpair = pd.concat((shotRows, rimRows), axis=1)
        self.PSPaired = pd.concat((paired, srpair), axis = 1)
        self.PSPaired['IDNum'] = str(hash(time.time()))
        self.PSPaired.drop_duplicates(subset = 'IDNum', inplace = True)
    def obConstructor(self):
        self.rowDict = {}
        hoop = np.array([5.35, 25])
        cols = ['TeamID', 'PlayerID', 'PlayerX', 'PlayerY', 'PlayerZ']
        tostr = cols[:2]
        rowcols = ['Clock', 'Period', 'RebPlayerID', 'RebPlayerName', 'ShootPlayerID', \
                   'ShootPlayerName', 'ShootTeamID', 'ShootTeamName', 'pre_ShotClock', \
                   'pre_rClock', 'post_ShotClock', 'post_rClock', 'IDNum']
        df = self.PSPaired
        for ID in df['IDNum'].unique():
            row = df[df['IDNum'] == ID]
            dfs = [pd.DataFrame(np.array(row['pre_Position'].values[0]), columns=cols), pd.DataFrame(np.array(row['post_Position'].values[0]), columns=cols)]
            for frame in dfs:
                frame['PlayerX'] = frame['PlayerX'] - 47
                diff = (frame[['PlayerX', 'PlayerY']].values - hoop)
                frame['HDist'] = np.sqrt((diff ** 2).sum(axis = 1))
                frame['Angle'] = np.arctan2(diff[:,0], diff[:,1])
                frame.pop('PlayerZ')
                for col in tostr:
                    frame[col] = frame[col].astype(str).str[:-2]
            ucols = frame.columns.values[2:]
            cols2 = [tostr + ['pre_' + col for col in ucols], tostr + ['post_' + col for col in ucols]]
            dfs[0].columns = cols2[0]
            dfs[1].columns = cols2[1]
            dfs[1].pop('TeamID')
            new = dfs[0].set_index('PlayerID').join(dfs[1].set_index('PlayerID'))
            new.reset_index(inplace=True)
            for col in rowcols:
                new[col] = row[col].values.repeat(11)
            new['ShotID'] = (row['IDNum'].values).repeat(11) + new['PlayerID'].values
            try:
                self.rowDict[ID] = features(new[new['PlayerID'] != '-1'])
            except Exception:
                pass
        try:
            self.gameFrame = pd.concat(self.rowDict.values(), ignore_index=True)
        except Exception:
            pass
        return self.rowDict


    def run(self):
        self.playbyplay()
        # print 'pbp done'
        self.sportvu()
        # print 'sportvu done'
        self.rimshots()
        # print 'rimshots done'
        self.obConstructor()
        # print 'constructor done'
        pass

if __name__ == '__main__':
    import time
    path = './data/games/'
    gameID = '0021500492.json'
    t = time.time()
    coo = Coordination(gameID, path)
    coo.run()

    print 'Runtime: ' + str(time.time() - t) + ' seconds'
