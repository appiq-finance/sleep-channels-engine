"""Stormwood / Valley & Tide sound engine v1.

Builds sleep-safe ambient soundscapes from code-generated layers:
- every layer is synthesised from randomness, so no two renders are identical
- stereo is genuinely de-correlated (independent left/right sources)
- loudness is mastered to a target LUFS with a true-peak ceiling
"""
import numpy as np
from scipy import signal

SR = 48000


def rng(seed):
    return np.random.default_rng(seed)


# ---------- basic generators ----------
def coloured_noise(n, colour, r):
    """white / pink / brown noise via spectral shaping."""
    white = r.standard_normal(n)
    if colour == "white":
        return white
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1]
    if colour == "pink":
        spec /= np.sqrt(f)
    elif colour == "brown":
        spec /= f
    out = np.fft.irfft(spec, n)
    return out / (np.std(out) + 1e-12)


def bandpass(x, lo, hi, order=4):
    sos = signal.butter(order, [lo, hi], btype="band", fs=SR, output="sos")
    return signal.sosfiltfilt(sos, x)


def lowpass(x, fc, order=4):
    sos = signal.butter(order, fc, btype="low", fs=SR, output="sos")
    return signal.sosfiltfilt(sos, x)


def highpass(x, fc, order=2):
    sos = signal.butter(order, fc, btype="high", fs=SR, output="sos")
    return signal.sosfiltfilt(sos, x)


def slow_lfo(n, r, period_s=(20, 90), depth=0.25):
    """smooth random modulation (gusts / intensity drift), never periodic."""
    pts = max(4, int(n / SR / np.mean(period_s)) + 4)
    knots = 1 + depth * (r.random(pts) * 2 - 1)
    x = np.linspace(0, pts - 1, n)
    from scipy.interpolate import CubicSpline
    return np.clip(CubicSpline(np.arange(pts), knots)(x), 0.3, 2.0)


def grains(n, rate, kernels, r, amp_sigma=0.7):
    """Poisson-distributed impulses convolved with randomly chosen short kernels."""
    out = np.zeros(n)
    for k in kernels:
        count = r.poisson(rate * n / SR / len(kernels))
        idx = r.integers(0, n, count)
        train = np.zeros(n)
        np.add.at(train, idx, r.lognormal(0, amp_sigma, count))
        out += signal.fftconvolve(train, k, mode="same")
    return out


def drop_kernel(r, f_lo, f_hi, decay_ms):
    length = int(SR * decay_ms * 5 / 1000)
    t = np.arange(length) / SR
    f = r.uniform(f_lo, f_hi)
    env = np.exp(-t / (decay_ms / 1000))
    k = env * (np.sin(2 * np.pi * f * t) * 0.6 + r.standard_normal(length) * 0.4)
    k = bandpass(np.pad(k, 64), f_lo * 0.6, min(f_hi * 1.6, SR / 2 - 100))[64:-64]
    return k / (np.max(np.abs(k)) + 1e-12)


# ---------- layers ----------
def rain_bed(n, r, intensity=1.0):
    """the steady 'hiss' of rain on wood shingles: band-limited pink noise."""
    ch = []
    for _ in range(2):
        x = 0.7 * coloured_noise(n, "pink", r) + 0.3 * coloured_noise(n, "brown", r)
        x = bandpass(x, 450, 6500)
        x *= slow_lfo(n, r, (25, 80), 0.15 * intensity)
        ch.append(x)
    return np.stack(ch)


def patter(n, r, rate=1500):
    """dense tiny impacts: the crisp 'texture' that makes noise sound like rain."""
    kernels = [drop_kernel(r, 2000, 7000, r.uniform(0.8, 2.5)) for _ in range(10)]
    return np.stack([grains(n, rate, kernels, r, 0.9) for _ in range(2)])


def roof_drops(n, r, rate=260):
    """individual drops hitting the roof, scattered across the stereo field."""
    kernels = [drop_kernel(r, 900, 2600, 6) for _ in range(6)] + \
              [drop_kernel(r, 2500, 6000, 3) for _ in range(6)]
    return np.stack([grains(n, rate, kernels, r) for _ in range(2)])


def eave_drips(n, r, rate=1.6):
    """slow, heavier drips from the eaves - low, soft, irregular."""
    kernels = [drop_kernel(r, 180, 650, 35) for _ in range(5)]
    return np.stack([grains(n, rate, kernels, r, 0.5) for _ in range(2)])


def room_tone(n, r):
    """quiet low cabin interior resonance."""
    return np.stack([lowpass(coloured_noise(n, "brown", r), 260) for _ in range(2)])


def fire_crackle(n, r, rate=7):
    kernels = [drop_kernel(r, 1500, 5500, 2) for _ in range(5)] + \
              [drop_kernel(r, 300, 1200, 12) for _ in range(3)]
    crack = np.stack([grains(n, rate, kernels, r, 1.0) for _ in range(2)])
    roar = np.stack([lowpass(coloured_noise(n, "brown", r), 400) * slow_lfo(n, r, (4, 12), 0.35)
                     for _ in range(2)])
    return crack, roar


def distant_thunder(n, r, events_per_hour=6):
    out = np.zeros((2, n))
    count = r.poisson(events_per_hour * n / SR / 3600)
    for _ in range(count):
        dur = int(SR * r.uniform(6, 14))
        start = r.integers(0, max(1, n - dur))
        t = np.arange(dur) / SR
        env = (1 - np.exp(-t / r.uniform(0.6, 1.5))) * np.exp(-t / r.uniform(2.0, 4.0))
        body = lowpass(coloured_noise(dur, "brown", r), r.uniform(90, 160)) * env
        pan = r.uniform(0.3, 0.7)
        out[0, start:start + dur] += body * (1 - pan)
        out[1, start:start + dur] += body * pan
    return out


# ---------- mastering ----------
def norm(x):
    return x / (np.sqrt(np.mean(x ** 2)) + 1e-12)


def master(mix, target_lufs=-22.0, tp_ceiling_db=-2.0, fade_in=10, fade_out=30):
    import pyloudnorm as pyln
    # gentle sleep EQ: remove sub-rumble, soften the top
    mix = highpass(mix, 25)
    sos = signal.butter(2, 12000, btype="low", fs=SR, output="sos")
    mix = 0.75 * mix + 0.25 * signal.sosfiltfilt(sos, mix)
    n = mix.shape[1]
    fi, fo = min(n // 4, fade_in * SR), min(n // 4, fade_out * SR)
    env = np.ones(n)
    env[:fi] = np.sin(np.linspace(0, np.pi / 2, fi)) ** 2
    env[n - fo:] = np.cos(np.linspace(0, np.pi / 2, fo)) ** 2
    mix = mix * env
    meter = pyln.Meter(SR)
    loud = meter.integrated_loudness(mix.T)
    mix = mix * 10 ** ((target_lufs - loud) / 20)
    # soft limiter, re-run until the 4x-oversampled (true) peak is under the ceiling
    from scipy.signal import resample_poly
    target = 10 ** (tp_ceiling_db / 20)
    ceiling = target
    pre = mix.copy()
    for _ in range(6):
        knee = 0.6 * ceiling
        mix = pre.copy()
        a = np.abs(mix)
        over = a > knee
        mix[over] = np.sign(mix[over]) * (knee + (ceiling - knee) * np.tanh((a[over] - knee) / (ceiling - knee)))
        tp = max(np.max(np.abs(resample_poly(mix[c], 4, 1))) for c in range(mix.shape[0]))
        if tp <= target:
            break
        ceiling *= target / tp * 0.97
    return mix.astype(np.float32)


# ---------- v2 fire: clustered crackles, pops, hiss and roar ----------
def click_kernel(r, lo=1200, hi=9000):
    length = int(SR * r.uniform(0.0015, 0.004))
    t = np.arange(length) / SR
    k = r.standard_normal(length) * np.exp(-t / r.uniform(0.0003, 0.0011))
    k = bandpass(np.pad(k, 256), lo, hi, 2)[256:-256]
    return k / (np.max(np.abs(k)) + 1e-12)


def fire_v2(n, r, crackle_clusters_per_s=1.6, pops_per_s=0.18):
    out = np.zeros(n)
    # clustered crackles: a burst of 3-14 tiny clicks within 40-350 ms
    clicks = [click_kernel(r) for _ in range(16)]
    c = r.poisson(crackle_clusters_per_s * n / SR)
    for start in r.integers(0, n, c):
        m = r.integers(3, 15)
        offs = np.sort(r.integers(0, int(SR * r.uniform(0.04, 0.35)), m))
        base = r.lognormal(0, 0.6)
        for o in offs:
            k = clicks[r.integers(len(clicks))]
            s = start + o
            if s + len(k) < n:
                out[s:s + len(k)] += k * base * r.lognormal(0, 0.5)
    crackle = out
    # occasional louder wood pops
    pops = grains(n, pops_per_s, [drop_kernel(r, 500, 2200, r.uniform(4, 12)) for _ in range(4)], r, 0.4)
    flick = slow_lfo(n, r, (3, 9), 0.4)
    hiss = bandpass(coloured_noise(n, "pink", r), 1800, 6000) * flick
    roar = lowpass(coloured_noise(n, "brown", r), 320) * flick
    return crackle, pops, hiss, roar


def pan(mono, position):
    """position 0 = hard left, 0.5 = centre, 1 = hard right (constant power)."""
    a = position * np.pi / 2
    return np.stack([mono * np.cos(a), mono * np.sin(a)])
