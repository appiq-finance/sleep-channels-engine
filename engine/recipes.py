"""Sound recipes. Each returns an un-mastered stereo block (2, n).
fire_level lets the long renderer die the fire down slowly at the end of a video."""
import numpy as np
from sound_engine import *


def _roof_rain(n, r, heavy=1.0):
    return (0.45 * norm(rain_bed(n, r)) + 0.65 * norm(patter(n, r, int(1700 * heavy)))
            + 0.50 * norm(roof_drops(n, r, int(240 * heavy))) + 0.06 * norm(eave_drips(n, r, 0.8)))


def _fire(n, r, level=1.0):
    crackle, pops, hiss, roar = fire_v2(n, r, 1.6 * max(level, 0.15), 0.18 * max(level, 0.15))
    return level * (pan(0.55 * norm(crackle), 0.32) + pan(0.30 * norm(pops), 0.32)
                    + pan(0.05 * norm(hiss), 0.35) + pan(0.18 * norm(roar), 0.4))


def rain_only(n, r, fire_level=1.0):
    return _roof_rain(n, r)


def rain_fire(n, r, fire_level=1.0):
    return _roof_rain(n, r) + _fire(n, r, fire_level)


def thunder_fire(n, r, fire_level=1.0):
    return _roof_rain(n, r, 1.1) + _fire(n, r, fire_level) + 0.9 * norm(distant_thunder(n, r, 10)) * 0.5


RECIPES = {f.__name__: f for f in (rain_only, rain_fire, thunder_fire)}
HAS_FIRE = {"rain_fire", "thunder_fire"}
