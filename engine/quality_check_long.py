"""Audio Quality Engineer for full-length videos. Streams the whole file in 5-minute pieces and
checks: loudness, true peak, clipping, sudden jumps, silent dropouts, left/right balance, stereo width.
usage: quality_check_long.py <file> <target_lufs> <expected_seconds>   (exit code 1 = FAIL, do not upload)"""
import sys, json, numpy as np, soundfile as sf, pyloudnorm as pyln
from scipy.signal import resample_poly

path, target, expect = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
f = sf.SoundFile(path); sr = f.samplerate; meter = pyln.Meter(sr)
energy, st, tp, clipped, l2, r2, lr, n = [], [], -200.0, 0, 0.0, 0.0, 0.0, 0
for x in f.blocks(blocksize=300 * sr, dtype="float64", always_2d=True):
    if len(x) > sr:
        energy.append(10 ** (meter.integrated_loudness(x) / 10))
    w = 3 * sr
    for s in range(0, len(x) - w + 1, w):
        st.append(10 * np.log10(np.mean(x[s:s + w] ** 2) + 1e-12))
    pk = np.argmax(np.abs(x).max(axis=1))            # true peak around the loudest sample of the block
    seg = x[max(0, pk - 2000):pk + 2000]
    tp = max(tp, 20 * np.log10(np.max(np.abs(resample_poly(seg, 4, 1, axis=0))) + 1e-12))
    clipped += int(np.sum(np.abs(x) >= 0.999))
    l2 += np.sum(x[:, 0] ** 2); r2 += np.sum(x[:, 1] ** 2); lr += np.sum(x[:, 0] * x[:, 1]); n += len(x)
st = np.array(st); body = st[5:-12]
rep = {
    "file": path, "duration_s": round(n / sr, 1), "expected_s": expect,
    "integrated_LUFS": round(10 * np.log10(np.mean(energy)), 2), "target_LUFS": target,
    "true_peak_dBTP": round(tp, 2), "clipped_samples": clipped,
    "loudness_jumps_over_3dB": int(np.sum(np.abs(np.diff(body)) > 3.0)),
    "silent_dropouts": int(np.sum(body < np.median(body) - 20)),
    "left_right_balance_dB": round(10 * np.log10(l2 / r2), 2),
    "stereo_correlation": round(float(lr / np.sqrt(l2 * r2)), 3),
    "loudness_spread_dB": round(float(np.percentile(body, 95) - np.percentile(body, 5)), 2),
}
rep["PASS"] = bool(abs(rep["integrated_LUFS"] - target) < 1 and tp <= -1.0 and clipped == 0
                   and rep["loudness_jumps_over_3dB"] == 0 and rep["silent_dropouts"] == 0
                   and abs(rep["left_right_balance_dB"]) < 1.5 and rep["stereo_correlation"] < 0.5
                   and abs(rep["duration_s"] - expect) < 2)
print(json.dumps(rep, indent=2))
sys.exit(0 if rep["PASS"] else 1)
