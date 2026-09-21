"""
SOFI Algorithm 2 -- dynamic, local-minima-based segmentation.

Background
----------
Algorithm 1 in "Decoding the Speaking Hands" thresholds the optical-flow
intensity (OFI) signal at a single global value. We reproduced it in
`sofi_segmentation.ipynb` and confirmed the paper's own motivation for moving
on: no single threshold transfers across words, because absolute motion
magnitude varies with signer speed, hand size and camera angle.

Algorithm 2 drops the global threshold entirely. Fingerspelling alternates
between *holds* (the hand is posed on a letter, motion ~0) and *transitions*
(the hand travels to the next letter, motion high). So each letter shows up as
a local MINIMUM of the OFI signal, whatever that signal's absolute scale is.
Detecting minima is scale-free, which is exactly the property the fixed
threshold lacked.

Deviations from the paper, and why
----------------------------------
1. OFI proxy: the paper runs Farneback dense optical flow on raw pixels. We
   only have MediaPipe keypoints in the CSV, so OFI here is the mean
   frame-to-frame Euclidean displacement of the 21 landmarks. Same quantity
   conceptually (how much did the hand move this frame), different estimator.

2. Minima test: a literal reading of the paper's spec -- >=5 strictly
   decreasing frames followed by >=5 strictly increasing frames -- is far too
   brittle on a real smoothed signal. A single frame of noise anywhere in the
   10-frame window kills the detection. Measured over 218 clips it finds ~5
   FEWER segments than the word has letters (MAE 4.96, bias -4.94). We
   therefore use the tolerant form: frame i is a local minimum if it is the
   smallest value in +/-`run` frames AND both endpoints of that window are
   strictly above it. This keeps the paper's intent (a sustained dip, not a
   1-frame spike) while surviving noise -- MAE drops to ~1.2.

3. Undetected frames: rows with hand_detected == 0 are zero-filled in the CSV,
   which would inject enormous fake displacement spikes. We NaN them out and
   linearly interpolate before differencing.
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

KP_COLS_X = [f"kp{i}_x" for i in range(21)]
KP_COLS_Y = [f"kp{i}_y" for i in range(21)]

# Defaults chosen by the sweep in sofi_algorithm2.ipynb.
DEFAULT_SMOOTH_WIN = 13
DEFAULT_RUN = 7
DEFAULT_MIN_SEP = 15
DEFAULT_SEG_LEN = 11


def ofi_signal(sub: pd.DataFrame, smooth_win: int = DEFAULT_SMOOTH_WIN):
    """Return (raw_ofi, smoothed_ofi, bad_mask) for one clip.

    `sub` must be a single (signer, video) slice sorted by frame_id.
    `bad_mask[i]` is True where MediaPipe failed and the values were
    interpolated -- callers should refuse to trust those frames.
    """
    xs = sub[KP_COLS_X].to_numpy(dtype=float).copy()
    ys = sub[KP_COLS_Y].to_numpy(dtype=float).copy()
    bad = sub["hand_detected"].to_numpy() == 0
    xs[bad] = np.nan
    ys[bad] = np.nan
    xs = pd.DataFrame(xs).interpolate(limit_direction="both").to_numpy()
    ys = pd.DataFrame(ys).interpolate(limit_direction="both").to_numpy()

    disp = np.sqrt(np.diff(xs, axis=0) ** 2 + np.diff(ys, axis=0) ** 2).mean(axis=1)
    raw = np.concatenate([[0.0], disp])
    smooth = pd.Series(raw).rolling(smooth_win, min_periods=1, center=True).mean().to_numpy()
    return raw, smooth, bad


def find_local_minima(ofi: np.ndarray, run: int = DEFAULT_RUN, strict: bool = False):
    """Indices of sustained local minima in the OFI signal.

    strict=True  -- the paper's literal spec (monotone decrease then increase
                    over `run` frames). Kept for comparison; under-segments.
    strict=False -- tolerant form described in the module docstring (default).

    Vectorised over sliding windows; equivalent to the obvious per-frame loop.
    """
    n = len(ofi)
    if n < 2 * run + 1:
        return []

    if strict:
        d = np.diff(ofi)
        if len(d) < run:
            return []
        dec = sliding_window_view(d <= 0, run).all(axis=1)   # dec[j] covers d[j:j+run]
        inc = sliding_window_view(d >= 0, run).all(axis=1)
        idx = np.arange(run, n - run)
        keep = dec[idx - run] & inc[np.minimum(idx, len(inc) - 1)]
        return list(idx[keep])

    win = sliding_window_view(ofi, 2 * run + 1)   # win[j] covers ofi[j:j+2run+1]
    center = ofi[run:n - run]
    keep = (center <= win.min(axis=1) + 1e-12) & (win[:, 0] > center) & (win[:, -1] > center)
    return list(np.nonzero(keep)[0] + run)


def collapse_minima(minima, ofi: np.ndarray, min_sep: int = DEFAULT_MIN_SEP):
    """Merge minima closer than `min_sep` frames, keeping the deepest one.

    A flat hold produces a plateau of adjacent qualifying frames; without this
    one letter would be counted many times.
    """
    if not minima:
        return []
    groups = [[minima[0]]]
    for m in minima[1:]:
        if m - groups[-1][-1] <= min_sep:
            groups[-1].append(m)
        else:
            groups.append([m])
    return [min(g, key=lambda i: ofi[i]) for g in groups]


def minima_to_segments(minima, n_frames: int, seg_len: int = DEFAULT_SEG_LEN):
    """Center a `seg_len`-frame window on each minimum, clipped to the clip."""
    half = seg_len // 2
    return [(max(0, m - half), min(n_frames - 1, m + half)) for m in minima]


def segment_clip(sub: pd.DataFrame,
                 smooth_win: int = DEFAULT_SMOOTH_WIN,
                 run: int = DEFAULT_RUN,
                 min_sep: int = DEFAULT_MIN_SEP,
                 seg_len: int = DEFAULT_SEG_LEN,
                 strict: bool = False):
    """Full Algorithm 2 for one clip.

    Returns dict with raw/smooth OFI, bad-frame mask, minima frame indices and
    (start, end) segment bounds -- all indices are positional within `sub`.
    """
    raw, smooth, bad = ofi_signal(sub, smooth_win)
    minima = collapse_minima(find_local_minima(smooth, run, strict), smooth, min_sep)
    return {
        "raw_ofi": raw,
        "smooth_ofi": smooth,
        "bad_mask": bad,
        "minima": minima,
        "segments": minima_to_segments(minima, len(sub), seg_len),
    }
