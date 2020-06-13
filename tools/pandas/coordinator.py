import numpy as np
import pandas as pd


def find_rim(df, t, q):
    """
    Based on time signature, finds the point where ball starts to approach rim.
    """
    selection = df[(df['Quarter'] == q) & (df['RimStart'] >= t)]
    try:
        return selection.iloc[-1]
    except Exception:
        return []


def rim_err(df, t, q):
    """
    If the previous function does not work properly, due to various aspects of
    the game, returns a row that can be used as proxy.
    """
    selection = df[(df['Quarter'] == q) & (df['Clock'] >= t)]
    try:
        return selection.iloc[-1]
    except Exception:
        return []


def find_shot(df, srs):
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

    # 1.0 Sets offense and defnse, then sorts.
    df['Off'] = df['TeamID'] == df['ShootTeamID']
    df = df.sort_values(by='Off')

    # 2.0 Separates numeric columns for #math operations.

    numcols = ['pre_PlayerX', 'pre_PlayerY', 'pre_HDist', 'pre_Angle',
               'pos_PlayerX', 'pos_PlayerY', 'pos_HDist', 'pos_Angle', 'pre_GameClock', 'pos_GameClock']

    # 2.1 Separates out shooter specifically for comparison columns.

    shooter = df[df['PlayerID'] == df['ShootPlayerID']][numcols].values
    if shooter.shape[0]:  # removes erroneously identified shots

        # 3.0 Computes various numeric stats including how they move, their
        #     cosine similarity to the shooter, and also how they are boxing out.

        nums = df[numcols].values
        closer = ((df['pre_HDist'] < df['pos_HDist']).astype(int) - 0.5) * 2
        move = nums[:, [0, 1]] - nums[:, [4, 5]]
        df['MoveV'] = np.sqrt((move ** 2).sum(axis=1)) * closer
        df['pre_CosSim'] = np.cos(nums[:, 3] - shooter[:, 3])
        df['pos_CosSim'] = np.cos(nums[:, 7] - shooter[:, 7])
        df['DT'] = df['pre_GameClock'] - df['pos_GameClock']
        arrpre = df[['pre_PlayerX', 'pre_PlayerY']].values
        arrpos = df[['pos_PlayerX', 'pos_PlayerY']].values
        df['pre_Box'] = boxgen(arrpre)
        df['pos_Box'] = boxgen(arrpos)
        df['IsShoot'] = (df['PlayerID'] == df['ShootPlayerID']).astype(int)
        df['Rebounder'] = df['PlayerID'] == df['RebPlayerID']
        if df['Rebounder'].sum():  # filters shots with no rebounder
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
    bdist = np.sqrt(((arr - np.array([41.65, 25])) ** 2).sum(axis=1)).reshape(10, 1)
    arr = np.concatenate((arr, bdist), axis=1)
    arr1 = arr[:5, :]
    arr2 = arr[5:, :]
    dists = np.array([np.sqrt(((arr1[:, :2] - row)**2).sum(axis=1)) for row in arr2[:, :2]])
    d = np.argmin(dists, axis=1)
    o = np.argmin(dists, axis=0)
    dbox = [np.sum(o == i) for i in range(5)]
    obox = [np.sum(d == i) for i in range(5)]
    boxes = np.array(dbox + obox)
    return boxes


def rimshots(sportvu, pbp):
    """
    Uses clock from Play by Play to identify when the ball hits the rim, and
    then uses that as a reference point to search back and find when the
    shot started.
    """
    sportvu['Loc'] = sportvu.index # isolate for consistency, avoids problems with .iloc and indexing
    rims = {}
    prev = set() # avoids repeats
    for i, row in pbp.iterrows():
        t = row['Clock']
        q = row['Period']
        r = find_rim(sportvu, t, q)
        # makes sure shot is findable, due to problems with game clock
        if len(r):
                l = r['Loc'].astype(str)
                if l not in prev:
                    rims[str(i)] = r
                    prev.update([l])
                else:
                    r = rim_err(sportvu, t, q)
                    rims[str(i)] = r

    shots = {}
    prev = set()
    for key, value in rims.iteritems():
        r = find_shot(sportvu, value)
        # same as above, game log will sometimes be incomplete.
        if len(r):
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
    shot_df = pd.DataFrame(ups, index=pnum)
    reb_df = pd.DataFrame(rebs, index=pnum)
    keep_col = ['GameClock', 'Quarter', 'Position']
    shot_df = shot_df[keep_col]

    # Creates separate pre/post labeling

    shot_df.columns = ['pre_' + col for col in keep_col]
    reb_df = reb_df[keep_col]
    reb_df.columns = ['pos_' + col for col in keep_col]
    reb_df.head()
    pnum = np.array(pnum).astype(int)
    pbp = pbp.iloc[list(pnum), :]
    snr = shot_df.join(reb_df).reset_index(drop=True)
    pbp = pbp.join(snr)
    arr = pbp[['GameID', 'RebPlayerID', 'ShootPlayerID', 'Period', 'Clock']].values
    a = arr[:, 0]
    for i in range(1, 5):
        a += arr[:, i].astype(str)
    pbp['IDNum'] = a
    pbp = pbp.drop_duplicates(subset='IDNum').dropna().reset_index(drop=True)
    return pbp


class Coordinator:
    pass

    def run(self, tracking, pbp):
        pass
