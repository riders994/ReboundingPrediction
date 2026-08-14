"""Court geometry and shared lookups.

Every module that needs a court dimension or the hoop location reads it from here.
The original code hardcoded the hoop in two places (``coordinator.py`` and
``SportVU_extractor.py``) with slightly different values.
"""

# SportVU reports positions in feet on a full court with the origin at a corner.
COURT_LENGTH = 94.0
COURT_WIDTH = 50.0
HALF_COURT_X = COURT_LENGTH / 2  # 47.0
CENTER_Y = COURT_WIDTH / 2  # 25.0

# Rim centers on the full court. 5.25 ft from each baseline, centered in y.
RIM_INSET = 5.25
RIM_LEFT = (RIM_INSET, CENTER_Y)  # (5.25, 25.0)
RIM_RIGHT = (COURT_LENGTH - RIM_INSET, CENTER_Y)  # (88.75, 25.0)

# After folding to a single half court (see rebounding.data.court), the attacking
# rim always lands here. The original code used 41.65, which is 0.1 ft short.
HOOP = (HALF_COURT_X - RIM_INSET, CENTER_Y)  # (41.75, 25.0)

RIM_HEIGHT = 10.0

# The ball is reported as a player row with this team/player id.
BALL_ID = "-1"

# Ordinal encoding of listed position, used to give players a stable ordering.
POSITION_MAP = {
    "G": 1.0,
    "G-F": 5.0 / 3,
    "F-G": 7.0 / 3,
    "F": 3.0,
    "F-C": 11.0 / 3,
    "C-F": 13.0 / 3,
    "C": 5.0,
}
# Players occasionally have a missing or unlisted position; the original code
# raised KeyError on these and the game was dropped by a blanket except.
DEFAULT_POSITION = 3.0

# NBA play-by-play EVENTMSGTYPE values we care about.
EVENT_MADE_SHOT = 1
EVENT_MISSED_SHOT = 2
EVENT_FREE_THROW = 3
EVENT_REBOUND = 4

TRACKING_HZ = 25.0
