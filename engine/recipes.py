"""Sound recipes. Each returns an un-mastered stereo block (2, n).
fire_level lets the long renderer die the fire down slowly at the end of a video."""
import os
import numpy as np
from scipy import signal
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
    """slow waves in a sheltered cove (v2, 17 Nov): each wave swells and draws back at 8-14 s spacing over a
    constant surf bed, so the sound rises and falls by only ~4-5 dB - audible waves, no sudden surges."""
    low = np.stack([lowpass(coloured_noise(n, "brown", r), 500) for _ in range(2)])
    hi = np.stack([bandpass(coloured_noise(n, "pink", r), 600, 7000, 2) for _ in range(2)])
    surf = np.stack([bandpass(coloured_noise(n, "pink", r), 150, 2500, 2) for _ in range(2)])
    env_lo, env_hi = np.zeros(n), np.zeros(n); t = 0
    while t < n:
        dur = int(SR * r.uniform(8, 14)); a = r.lognormal(0, 0.2)
        m = min(dur, n - t); x = np.arange(m) / dur
        env_lo[t:t + m] += a * np.sin(np.pi * np.clip(x / 0.8, 0, 1)) ** 2
        env_hi[t:t + m] += a * (np.exp(-((x - 0.5) / 0.22) ** 2) + 0.4 * np.exp(-((x - 0.78) / 0.18) ** 2))
        t += int(dur * r.uniform(0.8, 0.95))
    sm = lambda e: np.clip(lowpass(e, 0.35, 2), 0, 1.5)
    out = np.zeros((2, n))
    for c in range(2):
        out[c] = (norm(low[c]) * (0.75 + 0.35 * sm(env_lo)) + norm(hi[c]) * (0.45 + 0.5 * sm(env_hi))
                  + 0.35 * norm(surf[c]))
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


def _rolling_thunder(n, r, per_hour=14):
    """far-off rolling thunder for sleep: slow 1.5-3 s swell, long 10-20 s roll, no sharp crack.
    Each roll is scaled so its loudest second has strength 1."""
    out = np.zeros((2, n))
    for _ in range(r.poisson(per_hour * n / SR / 3600)):
        dur = int(SR * r.uniform(10, 20)); start = r.integers(0, max(1, n - dur)); t = np.arange(dur) / SR
        env = (1 - np.exp(-t / r.uniform(1.5, 3.0))) * np.exp(-t / r.uniform(3.0, 6.0))
        env *= 1 + 0.35 * np.sin(2 * np.pi * t / r.uniform(2.5, 4.5) + r.uniform(0, 6))   # the roll
        env /= env.max()
        b = bandpass(coloured_noise(dur, "brown", r), 35, r.uniform(110, 220), 2)   # audible rumble, not sub-bass
        mid = bandpass(coloured_noise(dur, "pink", r), 140, 600, 2)                # so phones and earbuds hear it too
        body = (b / (np.std(b) + 1e-12) + 0.35 * mid / (np.std(mid) + 1e-12)) * env * r.uniform(0.6, 1.0)
        p = r.uniform(0.3, 0.7); out[0, start:start + dur] += body * (1 - p); out[1, start:start + dur] += body * p
    return out


THUNDER_GAIN = float(os.environ.get("THUNDER_GAIN", "1.5"))   # rumble ~17 dB above the quiet bed in its band, overall level moves ~1 dB


def valley_thunderstorm(n, r, fire_level=1.0):
    return valley_rain(n, r) + 0.25 * _wind(n, r) + THUNDER_GAIN * _rolling_thunder(n, r)

for f in (blizzard_fire, wind_snow, tin_roof_rain, brown_rain, storm_cycle, valley_rain, ocean_waves, mountain_stream, valley_thunderstorm):
    RECIPES[f.__name__] = f
HAS_FIRE.add("blizzard_fire")

# sounds that are meant to rise and fall (waves, gusts): the quality check allows gentle swells for these
DYNAMIC = {"ocean_waves", "blizzard_fire", "wind_snow"}


# ---------- new recipes (17 Nov, 4-week plan) ----------
def _hearth(n, r, level=1.0):
    """a fire on its own, for sleep: an independent burning log in each ear (a little bleed between them),
    soft crackles, and a steady ember bed with only gentle flicker, so there are no gaps or spikes.
    The end-of-video die-down is softened (about -3 dB at most) because the fire is the whole sound."""
    lv = 0.6 + 0.4 * max(level, 0.0)
    ch = []
    for _ in range(2):
        crackle, pops, _, _ = fire_v2(n, r, 1.2, 0.06)
        flick = slow_lfo(n, r, (4, 12), 0.15)
        roar = lowpass(coloured_noise(n, "brown", r), 320) * flick
        hiss = bandpass(coloured_noise(n, "pink", r), 1500, 6000, 2) * flick
        ch.append((0.13 * norm(crackle) + 0.04 * norm(pops), 0.50 * norm(roar) + 0.16 * norm(hiss)))
    out = np.stack([ch[0][0] + 0.3 * ch[1][0] + ch[0][1], ch[1][0] + 0.3 * ch[0][0] + ch[1][1]])
    k = 2.5 * np.sqrt(np.mean(out ** 2))          # round off the sharpest crackle peaks (sleep-safe, keeps loudness on target)
    return lv * k * np.tanh(out / k)


def fire_only(n, r, fire_level=1.0):
    """Stormwood hearth on a still winter night: fire alone, quiet room."""
    return _hearth(n, r, fire_level) + 0.10 * norm(room_tone(n, r))


def christmas_fire(n, r, fire_level=1.0):
    """Christmas night: fire inside, soft snowfall and a faint muffled breeze outside. No music, no bells."""
    hush = np.stack([lowpass(c, 700) for c in _wind(n, r, 0.5)])     # breeze heard through log walls
    return _hearth(n, r, fire_level) + 0.22 * norm(hush) + 0.06 * _snow_on_glass(n, r) + 0.10 * norm(room_tone(n, r))


def waterfall(n, r, fire_level=1.0):
    """fern-gorge waterfall: broad steady roar, a low plunge-pool rumble and close splashing."""
    roar = np.stack([bandpass(coloured_noise(n, "pink", r), 180, 9000, 2) * slow_lfo(n, r, (20, 60), 0.06)
                     for _ in range(2)])
    pool = np.stack([lowpass(coloured_noise(n, "brown", r), 220) for _ in range(2)])
    return 0.85 * norm(roar) + 0.35 * norm(pool) + 0.45 * _stream(n, r)


def _crickets(n, r, count=16):
    """a chorus of field crickets: each sings short pulsed chirps at its own pitch and pace,
    fading in and out slowly. Different crickets in each ear (wide, natural stereo)."""
    t = np.arange(n) / SR
    out = np.zeros((2, n))
    for c in range(2):
        for _ in range(count):
            f = r.uniform(3900, 5200); per = 1 / r.uniform(1.4, 3.2)
            pulses = r.integers(2, 5); pw, gap = r.uniform(0.010, 0.016), r.uniform(0.016, 0.024)
            L = int(SR * (pulses * gap + pw)) + 1; k = np.zeros(L)
            for p in range(pulses):
                s = int(SR * p * gap); m = int(SR * pw)
                k[s:s + m] += np.sin(np.pi * np.arange(m) / m) ** 2
            train = np.zeros(n)
            tk = r.uniform(0, per); jit = r.uniform(0.01, 0.04)
            idx = []
            while tk < n / SR:
                idx.append(int(tk * SR)); tk += per * (1 + jit * r.standard_normal())
            np.add.at(train, np.array(idx, dtype=int), 1.0)
            env = signal.oaconvolve(train, k, mode="full")[:n]
            amp = slow_lfo(n, r, (40, 150), 0.25) * r.uniform(0.5, 1.0)
            out[c] += env * amp * np.sin(2 * np.pi * f * t + r.uniform(0, 6))
    return out


def _frogs(n, r, per_min=10):
    """soft pond frogs: low pulsed croaks, sparse and far off."""
    out = np.zeros((2, n))
    for _ in range(r.poisson(per_min * n / SR / 60)):
        dur = r.uniform(0.25, 0.7); L = int(SR * dur); tt = np.arange(L) / SR
        f, pr = r.uniform(380, 900), r.uniform(18, 40)
        env = np.sin(np.pi * tt / dur) ** 2 * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * pr * tt)))
        env = lowpass(env, 150, 2)
        body = env * (np.sin(2 * np.pi * f * tt) + 0.4 * np.sin(2 * np.pi * 2 * f * tt)) * r.lognormal(0, 0.3)
        s = r.integers(0, max(1, n - L)); p = r.uniform(0.15, 0.85)
        out[0, s:s + L] += body * (1 - p); out[1, s:s + L] += body * p
    return out


def pond_night(n, r, fire_level=1.0):
    """summer pond at night: cricket chorus, far-off frogs, faint night air. No music."""
    air = np.stack([bandpass(coloured_noise(n, "pink", r), 150, 3000, 2) for _ in range(2)])
    return 0.55 * norm(_crickets(n, r)) + 0.12 * norm(_frogs(n, r)) + 0.35 * norm(air)


for f in (fire_only, christmas_fire, waterfall, pond_night):
    RECIPES[f.__name__] = f
HAS_FIRE.update({"fire_only", "christmas_fire"})
