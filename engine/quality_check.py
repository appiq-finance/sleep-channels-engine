"""Audio Quality Engineer v1 - measures a finished soundscape and writes a report."""
import json, sys
import numpy as np
import soundfile as sf
import pyloudnorm as pyln

def check(path, target_lufs=-22.0):
    x, sr = sf.read(path, always_2d=True)
    meter = pyln.Meter(sr)
    integrated = meter.integrated_loudness(x)
    # short-term loudness, 3 s windows every second
    win, hop = 3 * sr, sr
    st = []
    for s in range(0, len(x) - win, hop):
        seg = x[s:s + win]
        rms = np.sqrt(np.mean(seg ** 2)) + 1e-12
        st.append(20 * np.log10(rms))
    st = np.array(st)
    body = st[15:-35] if len(st) > 60 else st  # ignore fades
    jumps = np.where(np.abs(np.diff(body)) > 3.0)[0]
    # sample peak with 4x oversampling as a true-peak estimate
    from scipy.signal import resample_poly
    tp = 20 * np.log10(np.max(np.abs(resample_poly(x, 4, 1, axis=0))) + 1e-12)
    clipped = int(np.sum(np.abs(x) >= 0.999))
    silence = int(np.sum(body < (np.median(body) - 20)))
    lr_balance = 20 * np.log10((np.sqrt(np.mean(x[:, 0] ** 2)) + 1e-12) / (np.sqrt(np.mean(x[:, 1] ** 2)) + 1e-12))
    corr = float(np.corrcoef(x[:, 0], x[:, 1])[0, 1])
    rep = {
        "file": path, "duration_s": round(len(x) / sr, 1),
        "integrated_LUFS": round(integrated, 2), "target_LUFS": target_lufs,
        "true_peak_dBTP_est": round(tp, 2), "clipped_samples": clipped,
        "loudness_jumps_over_3dB": len(jumps), "silent_dropouts_s": silence,
        "left_right_balance_dB": round(lr_balance, 2), "stereo_correlation": round(corr, 3),
        "loudness_spread_dB": round(float(np.percentile(body, 95) - np.percentile(body, 5)), 2),
    }
    rep["PASS"] = bool(abs(integrated - target_lufs) < 1 and tp <= -1.5 and clipped == 0
                   and len(jumps) == 0 and silence == 0 and abs(lr_balance) < 1.5 and corr < 0.5)
    return rep

if __name__ == "__main__":
    print(json.dumps(check(sys.argv[1], float(sys.argv[2]) if len(sys.argv)>2 else -22.0), indent=2))
