# SportVU tracking corpus

636 games, 3.79 GB compressed. Each archive holds one JSON of
roughly 100 MB: every player and the ball at 25 Hz for a full game.

## Provenance, and why this matters

Pulled in 2016 from the public STATS SportVU dumps (mirrored at
`riders994/BasketballData`). STATS withdrew that feed later the same year, and from
the 2017-18 season the NBA moved to Second Spectrum, whose tracking data has never
been public. **There is no way to obtain more of this data.** Treat the corpus as
irreplaceable and verify against `SHA256SUMS` before anything that rewrites it.

It is also a hard modelling constraint: roughly 60k rebounds is the ceiling, which
argues for modest model capacity over large architectures.

## Verifying

```
cd data/7zips && sha256sum -c ../SHA256SUMS
```

## Known-bad archives

5 of 636 files (0.8%) are truncated downloads that cannot be
read. Listed here so a failed build is recognised as a known data defect rather than
a new bug; `build_many` reports them by name.

| file | bytes |
|---|---|
| `01.14.2016.CHI.at.PHI.7z` | 32 |
| `01.14.2016.CLE.at.SAS.7z` | 32 |
| `01.14.2016.DET.at.MEM.7z` | 32 |
| `01.23.2016.CHI.at.CLE.7z` | 2350 |
| `01.23.2016.UTA.at.WAS.7z` | 32 |

## Filename quirk

10 archives from 2015-12-05 have a directory path mangled into the filename
(`2016.NBA.Raw.SportVU.Game.Logs12.05.2015.BOS.at.SAS.7z`). They are valid; nothing
reads the filename, because the NBA game id is inside the JSON.

## Play-by-play, and why it is no longer the NBA's

`pbp/` holds cached NBA `playbyplayv2` payloads. It has exactly one game in it, and
cannot get more: `stats.nba.com/stats/*` accepts a connection and then holds it open
until timeout, identically for a script and for a logged-in browser. `data.nba.net`
no longer resolves; `cdn.nba.com`'s live feed 403s a 2015-16 game.

The surviving payload is worth keeping despite being a single game: it is the only
record of what the original feed said, and `tests/test_bref.py` uses it to hold the
replacement source to it.

`bref/` holds cached Basketball-Reference pages, one per game, ~250 KB each. It is
gitignored and refetchable in about 35 minutes at their 20-requests-a-minute limit.
Unlike the tracking corpus, **nothing here is irreplaceable** — delete it freely.

## Coverage

2015-16 regular season.

| month | games |
|---|---|
| 2015-10 | 34 |
| 2015-11 | 220 |
| 2015-12 | 226 |
| 2016-01 | 156 |

_Generated 2026-08-14._
