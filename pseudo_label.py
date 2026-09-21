"""
Frame-level pseudo-labelling of PSL-CFRT clips.

Key idea: pseudo-labelling is NOT the same problem as inference. At inference
time we would not know how many letters a clip contains. At labelling time we
do -- the filename gives the word, and `word_letter_mapping.json` gives that
word's ordered letter sequence. So we can run segmentation *constrained* to
produce exactly k = len(letters) holds, rather than hoping free segmentation
lands on the right count.

That turns a brittle detection problem into a cheap alignment problem:
choose exactly k frames, one per letter, that are (a) deep minima of the OFI
signal and (b) far enough apart to be distinct holds. A DP solves it exactly.

Alignment is order-preserving because fingerspelling is strictly sequential --
the i-th hold is the i-th letter, with no reordering and no insertion. Assumes
every letter is actually spelled and none is skipped; clips where that breaks
should surface as low-confidence in `alignment_confidence`.
"""
import numpy as np
import pandas as pd

from sofi_dynamic import (
    DEFAULT_MIN_SEP,
    DEFAULT_RUN,
    DEFAULT_SEG_LEN,
    DEFAULT_SMOOTH_WIN,
    collapse_minima,
    find_local_minima,
    minima_to_segments,
    ofi_signal,
)

TRANSITION = "<TRANS>"


def select_k_minima(ofi: np.ndarray, k: int, min_gap: int = DEFAULT_MIN_SEP,
                    margin: int = DEFAULT_RUN):
    """Pick exactly k frame indices minimising total OFI, >= min_gap apart.

    Exact DP over (position, number chosen). `margin` keeps picks away from the
    clip edges, where the hand is entering/leaving frame and the smoothed
    signal is unreliable. Returns [] if k holds cannot physically fit.
    """
    n = len(ofi)
    lo, hi = margin, n - margin  # candidate range [lo, hi)
    if k <= 0 or hi - lo < 1 or (k - 1) * min_gap >= (hi - lo):
        return []

    cand = np.arange(lo, hi)
    cost = ofi[cand]
    m = len(cand)
    INF = np.inf

    # best[j][i] = min total cost choosing j+1 picks, last pick at cand[i]
    best = np.full((k, m), INF)
    back = np.full((k, m), -1, dtype=int)
    best[0] = cost

    for j in range(1, k):
        # running min over i' with cand[i] - cand[i'] >= min_gap
        run_min, run_arg = INF, -1
        p = 0
        for i in range(m):
            while p < m and cand[p] <= cand[i] - min_gap:
                if best[j - 1][p] < run_min:
                    run_min, run_arg = best[j - 1][p], p
                p += 1
            if run_arg >= 0 and run_min < INF:
                best[j][i] = run_min + cost[i]
                back[j][i] = run_arg

    end = int(np.argmin(best[k - 1]))
    if not np.isfinite(best[k - 1][end]):
        return []

    picks = []
    j = k - 1
    while j >= 0 and end >= 0:
        picks.append(int(cand[end]))
        end = back[j][end]
        j -= 1
    return sorted(picks)


def label_clip(sub: pd.DataFrame, letters,
               smooth_win: int = DEFAULT_SMOOTH_WIN,
               run: int = DEFAULT_RUN,
               min_sep: int = DEFAULT_MIN_SEP,
               seg_len: int = DEFAULT_SEG_LEN):
    """Pseudo-label one clip against its known letter sequence.

    Returns (frame_labels, info). `frame_labels` is a per-frame array of letter
    strings / TRANSITION / None (None = MediaPipe failed, do not train on it).
    `info` carries the diagnostics used to rank clips for manual spot-checking.
    """
    k = len(letters)
    raw, smooth, bad = ofi_signal(sub, smooth_win)
    n = len(sub)

    picks = select_k_minima(smooth, k, min_gap=min_sep, margin=run)
    if not picks:
        # Holds cannot fit at the default spacing -- relax the gap rather than
        # discard a long word outright.
        for relaxed in (min_sep - 3, min_sep - 6, 5, 3):
            if relaxed < 2:
                break
            picks = select_k_minima(smooth, k, min_gap=relaxed, margin=run)
            if picks:
                min_sep = relaxed
                break
    if not picks:
        return None, {"n_frames": n, "n_letters": k, "status": "no_alignment"}

    segments = minima_to_segments(picks, n, seg_len)

    labels = np.full(n, TRANSITION, dtype=object)
    for (a, b), letter in zip(segments, letters):
        labels[a:b + 1] = letter
    labels[bad] = None

    # --- diagnostics -------------------------------------------------------
    free = collapse_minima(find_local_minima(smooth, run), smooth, min_sep)
    # how many constrained picks have a free-detection minimum nearby
    agree = sum(1 for p in picks if free and min(abs(p - f) for f in free) <= seg_len // 2)

    seg_frames = np.zeros(n, dtype=bool)
    for a, b in segments:
        seg_frames[a:b + 1] = True

    median_ofi = float(np.median(smooth)) or 1.0
    pick_depth = float(np.mean(smooth[picks]) / median_ofi)

    # Constraint-dominated alignment: when a word has more letters than the clip
    # has room for at the default spacing, the DP stops choosing troughs and
    # just tiles the clip at the separation floor. The picks then come out
    # perfectly evenly spaced and land mid-transition -- visible in the contact
    # sheets as a row of motion-blurred, near-identical hands. Labels from such
    # clips are not trustworthy. Three independent symptoms, any of which flags:
    gaps = np.diff(picks) if len(picks) > 1 else np.array([0])
    gap_cv = float(gaps.std() / gaps.mean()) if len(gaps) > 3 and gaps.mean() > 0 else np.nan
    frames_per_letter = n / k
    constraint_dominated = bool(
        min_sep < DEFAULT_MIN_SEP            # the relaxation loop above fired
        or pick_depth > 0.90                 # picks are not in troughs at all
        or frames_per_letter < 15            # <0.3 s per letter at 50 fps
    )

    info = {
        "n_frames": n,
        "n_letters": k,
        "status": "ok",
        "n_free_segments": len(free),
        "count_error": len(free) - k,
        # fraction of picked holds corroborated by unconstrained detection
        "alignment_confidence": agree / k,
        # how deep the picked minima are vs the clip's typical motion (lower=better)
        "mean_pick_depth": pick_depth,
        # fraction of labelled hold frames MediaPipe actually saw
        "seg_detect_rate": float((~bad[seg_frames]).mean()),
        # spacing regularity; ~0 with >3 gaps means uniform tiling, not detection
        "gap_cv": gap_cv,
        "frames_per_letter": frames_per_letter,
        "constraint_dominated": constraint_dominated,
        "minima": picks,
        "segments": segments,
        "min_sep_used": min_sep,
    }
    return labels, info
