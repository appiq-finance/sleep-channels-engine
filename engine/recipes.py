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


# ---------- new layers (Nov 16) ----------
def _wind(n, r, strength=1.0):
    """gusting wind: several noise bands whose levels drift, so the pitch seems to rise and fall.
    Gusts are shared by both ears (the wind is outside), texture is independent; swells stay gentle."""
    bands = ((90, 260, 1.0), (220, 520, 0.8), (450, 900, 0.5), (800, 1600, 0.25))
    gust = [slow_lfo(n, r, (9, 22), 0.3) for _ in bands]
    swell = slow_lfo(n, r, (40, 90), 0.18)
    out = []
    for _ in range(2):
        x = np.zeros(n)
        for (lo, hi, g), e in zip(bands, gust):
            x += g * norm(bandpass(coloured_noise(n, "pink", r), lo, hi, 2)) * e
        out.append(x * swell)
    return strength * norm(np.stack(out))


def _snow_on_glass(n, r):
    """fine icy grains ticking on the window pane."""
    k = [drop_kernel(r, 4500, 9000, r.uniform(0.4, 1.0)) for _ in range(6)]
    return norm(np.stack([grains(n, 900, k, r, 0.8) for _ in range(2)])) * slow_lfo(n, r, (6, 20), 0.6)


def _tin_rain(n, r):
    """rain on a metal roof: brighter, ringing drops."""
    def ring(f, ms):
        L = int(SR * ms * 5 / 1000); t = np.arange(L) / SR
        k = np.exp(-t / (ms / 1000)) * (np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * f * 2.76 * t))
        return k / np.max(np.abs(k))
    k = [ring(r.uniform(1400, 5200), r.uniform(3, 9)) for _ in range(10)]
    drops = np.stack([grains(n, 520, k, r, 0.9) for _ in range(2)])
    return 0.35 * norm(rain_bed(n, r)) + 0.55 * norm(patter(n, r, 2000)) + 0.6 * norm(drops)


def _ocean(n, r):
    """waves: each one swells, breaks (bright) and draws back (dark hiss), at irregular 7-13 s spacing."""
    out = np.zeros((2, n)); t = 0
    low = np.stack([lowpass(coloured_noise(n, "brown", r), 500) for _ in range(2)])
    hi = np.stack([bandpass(coloured_noise(n, "pink", r), 600, 7000, 2) for _ in range(2)])
    env_lo, env_hi = np.zeros(n), np.zeros(n)
    while t < n:
        dur = int(SR * r.uniform(7, 13)); a = r.lognormal(0, 0.3)
        m = min(dur, n - t); x = np.arange(m) / dur
        env_lo[t:t + m] += a * np.sin(np.pi * np.clip(x / 0.75, 0, 1)) ** 2
        env_hi[t:t + m] += a * np.exp(-((x - 0.5) / 0.16) ** 2) + 0.35 * a * np.exp(-((x - 0.78) / 0.14) ** 2)
        t += int(dur * r.uniform(0.75, 0.95))
    sm = lambda e: np.clip(lowpass(e, 0.6, 2), 0, 1.6)
    pan_l = 0.5 + 0.2 * (slow_lfo(n, r, (9, 20), 1.0) - 1)
    out[0] = norm(low[0]) * (0.45 + 0.8 * sm(env_lo)) + 0.8 * norm(hi[0]) * (0.2 + 0.8 * sm(env_hi)) * 1.0
    out[1] = norm(low[1]) * (0.45 + 0.8 * sm(env_lo)) + 0.8 * norm(hi[1]) * (0.2 + 0.8 * sm(env_hi)) * 1.0
    return norm(out)


def _stream(n, r):
    """running water: steady rush plus thousands of tiny bubbling 'bloops'."""
    def bloop():
        ms = r.uniform(8, 30); L = int(SR * ms * 4 / 1000); t = np.arange(L) / SR
        f0 = r.uniform(350, 2400); f = f0 * (1 + r.uniform(0.2, 0.9) * t / t[-1])
        k = np.exp(-t / (ms / 1000)) * np.sin(2 * np.pi * np.cumsum(f) / SR)
        return k / np.max(np.abs(k))
    k = [bloop() for _ in range(14)]
    bub = np.stack([grains(n, 260, k, r, 0.8) for _ in range(2)])
    rush = np.stack([bandpass(coloured_noise(n, "pink", r), 500, 5000, 2) * slow_lfo(n, r, (10, 30), 0.15) for _ in range(2)])
    return 0.6 * norm(rush) + 0.75 * norm(bub)


def _brown(n, r):
    return norm(np.stack([lowpass(coloured_noise(n, "brown", r), 700) for _ in range(2)]))


def blizzard_fire(n, r, fire_level=1.0):
    return 1.0 * _wind(n, r, 1.0) + 0.22 * _snow_on_glass(n, r) + 0.9 * _fire(n, r, fire_level) + 0.08 * norm(room_tone(n, r))


def wind_snow(n, r, fire_level=1.0):
    return 1.0 * _wind(n, r, 0.8) + 0.3 * _snow_on_glass(n, r) + 0.1 * norm(room_tone(n, r))


def tin_roof_rain(n, r, fire_level=1.0):
    return _tin_rain(n, r)


def brown_rain(n, r, fire_level=1.0):
    return 1.1 * _brown(n, r) + 0.7 * _roof_rain(n, r)


def storm_cycle(n, r, fire_level=1.0):
    return _roof_rain(n, r, 1.15) + 0.35 * _wind(n, r) + 1.3 * norm(distant_thunder(n, r, 14)) * 0.5


def valley_rain(n, r, fire_level=1.0):
    """outdoor rain on leaves and ground: softer, wider, no roof."""
    leaves = [drop_kernel(r, 1200, 4500, r.uniform(1.5, 4)) for _ in range(10)]
    return (0.7 * norm(rain_bed(n, r)) + 0.5 * norm(patter(n, r, 1300))
            + 0.4 * norm(np.stack([grains(n, 320, leaves, r, 0.9) for _ in range(2)])))


def ocean_waves(n, r, fire_level=1.0):
    return _ocean(n, r)


def mountain_stream(n, r, fire_level=1.0):
    return _stream(n, r)


def valley_thunderstorm(n, r, fire_level=1.0):
    return valley_rain(n, r) + 0.3 * _wind(n, r) + 1.4 * norm(distant_thunder(n, r, 16)) * 0.5


for f in (blizzard_fire, wind_snow, tin_roof_rain, brown_rain, storm_cycle, valley_rain, ocean_waves, mountain_stream, valley_thunderstorm):
    RECIPES[f.__name__] = f
HAS_FIRE.add("blizzard_fire")

# sounds that are meant to rise and fall (waves, gusts): the quality check allows gentle swells for these
DYNAMIC = {"ocean_waves", "blizzard_fire", "wind_snow"}
