"""Long-form renderer: builds hours of audio in 5-minute blocks, each from fresh randomness
(nothing ever loops), crossfades the joins, masters to a fixed loudness and streams to disk.
usage: render_long.py <recipe> <seconds> <seed> <out.flac> [target_lufs]"""
import sys, os, numpy as np, soundfile as sf
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sound_engine import SR, rng, highpass
from scipy import signal
from recipes import RECIPES, HAS_FIRE

BLOCK, XF = 300, 6          # seconds per block, crossfade seconds
CEIL = 10 ** (-2.0 / 20)    # true-peak ceiling


def eq(mix):
    mix = highpass(mix, 25)
    sos = signal.butter(2, 12000, btype="low", fs=SR, output="sos")
    return 0.75 * mix + 0.25 * signal.sosfiltfilt(sos, mix)


def make_block(args):
    recipe, seed, i, secs, fire_level = args
    n = (secs + XF) * SR
    return i, eq(RECIPES[recipe](n, rng([seed, i]), fire_level)).astype(np.float32)


def limit(x):
    """look-ahead true-peak limiter: measures 4x-oversampled peaks, turns the gain down smoothly
    2 ms before each peak and back up after it. Only the rare loud pop is touched."""
    from scipy.signal import resample_poly
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    n = x.shape[1]
    pk = np.zeros(n, dtype=np.float32)
    for c in range(2):
        pk = np.maximum(pk, np.abs(resample_poly(x[c], 4, 1)).reshape(n, 4).max(axis=1))
    if pk.max() <= CEIL:
        return x
    g = np.minimum(1.0, CEIL / np.maximum(pk, 1e-9)).astype(np.float32)
    w = int(0.002 * SR) | 1
    g = uniform_filter1d(minimum_filter1d(g, 2 * w + 1), w)
    return x * g


def main():
    recipe, total, seed, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    target = float(sys.argv[5]) if len(sys.argv) > 5 else -17.0
    import pyloudnorm as pyln
    meter = pyln.Meter(SR)
    nb = int(np.ceil(total / BLOCK))
    jobs = []
    for i in range(nb):
        fl = 1.0
        if recipe in HAS_FIRE:   # fire dies down slowly over the last 15 minutes
            fl = {nb - 3: 0.75, nb - 2: 0.5, nb - 1: 0.25}.get(i, 1.0)
        jobs.append((recipe, seed, i, BLOCK, fl))
    xn = XF * SR
    up = np.sin(np.linspace(0, np.pi / 2, xn, dtype=np.float32)) ** 2
    tmp, pre = out + ".tmp.flac", None
    tail, written, need, energy = None, 0, total * SR, []
    # pass 1: build every block, crossfade the joins, store quietly, measure loudness
    with sf.SoundFile(tmp, "w", SR, 2, subtype="PCM_24", format="FLAC") as f, Pool(os.cpu_count()) as p:
        for i, blk in p.imap(make_block, jobs, chunksize=1):
            if pre is None:
                pre = 0.05 / (np.sqrt(np.mean(blk ** 2)) + 1e-12)
            blk = blk * pre
            if tail is not None:
                blk[:, :xn] = blk[:, :xn] * up + tail * (1 - up)
            body, tail = blk[:, :-xn], blk[:, -xn:].copy()
            body = body[:, :max(0, need - written)]
            if body.shape[1] > SR:
                energy.append((10 ** (meter.integrated_loudness(body.T.astype(np.float64)) / 10), body.shape[1]))
            f.write(body.T); written += body.shape[1]
            print(f"built {i + 1}/{nb}", flush=True)
    e = np.array(energy)
    gain = 10 ** ((target - 10 * np.log10(np.sum(e[:, 0] * e[:, 1]) / np.sum(e[:, 1]))) / 20)
    # pass 2: exact loudness, true-peak limiter, fade in / fade out
    done = 0
    with sf.SoundFile(tmp) as src, sf.SoundFile(out, "w", SR, 2, subtype="PCM_24", format="FLAC") as f:
        for x in src.blocks(blocksize=BLOCK * SR, dtype="float32", always_2d=True):
            x = limit(x.T * gain)
            m = x.shape[1]
            if done == 0:
                fi = min(10 * SR, m)
                x[:, :fi] *= np.sin(np.linspace(0, np.pi / 2, fi, dtype=np.float32)) ** 2
            if done + m >= need:
                fo = min(30 * SR, m)
                x[:, -fo:] *= np.cos(np.linspace(0, np.pi / 2, fo, dtype=np.float32)) ** 2
            f.write(x.T); done += m
    os.remove(tmp)
    print("done", done / SR, "s")


if __name__ == "__main__":
    main()
