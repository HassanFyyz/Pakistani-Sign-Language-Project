"""
Causal (online) SOFI Algorithm 2, for live camera input.

`sofi_dynamic.segment_clip` is an OFFLINE algorithm and cannot be used live:

  * it smooths the OFI signal with a *centred* rolling mean (needs future frames),
  * a frame is declared a local minimum only if it is the smallest in a +/-`run`
    window (needs `run` future frames),
  * `collapse_minima` merges minima within `min_sep` and keeps the deepest,
    so a candidate cannot be confirmed until `min_sep` further frames are seen.

Porting it naively to a live loop silently changes its behaviour. Instead this
module runs the *same* algorithm on a delay: at wall-clock frame t it emits
decisions about frame t - DELAY, where

    DELAY = max((smooth_win - 1) // 2, run, min_sep)

which at the tuned 50 fps defaults is 15 frames = 300 ms. That is the inherent
latency of the method, not an implementation artifact. It is comfortably within
tolerance for fingerspelling.

Parameters are specified in SECONDS, not frames
-----------------------------------------------
The offline defaults were tuned on 50 fps dataset video. A webcam is typically
30 fps, where the same 15-frame separation would mean 0.50 s instead of 0.30 s
and the segmenter would under-detect. Expressing everything in seconds and
converting at runtime is what makes the tuned values transfer; see
streaming_segmentation.ipynb for the measurement.
"""
from collections import deque

import numpy as np

from sofi_dynamic import (
    DEFAULT_MIN_SEP,
    DEFAULT_RUN,
    DEFAULT_SEG_LEN,
    DEFAULT_SMOOTH_WIN,
)

TUNED_FPS = 50.0

# The offline defaults, re-expressed as durations.
SMOOTH_SEC = DEFAULT_SMOOTH_WIN / TUNED_FPS   # 0.26 s
RUN_SEC = DEFAULT_RUN / TUNED_FPS             # 0.14 s
MIN_SEP_SEC = DEFAULT_MIN_SEP / TUNED_FPS     # 0.30 s
SEG_LEN_SEC = DEFAULT_SEG_LEN / TUNED_FPS     # 0.22 s


def frames_for(seconds: float, fps: float, odd: bool = False, minimum: int = 1) -> int:
    """Convert a duration to a frame count at the given fps."""
    n = max(minimum, int(round(seconds * fps)))
    if odd and n % 2 == 0:
        n += 1
    return n


class StreamingSegmenter:
    """Online hold detector. Feed it one frame of hand keypoints at a time.

    Usage:
        seg = StreamingSegmenter(fps=30)
        for kx, ky in webcam_keypoints():        # each (21,) arrays, or None
            for hold in seg.push(kx, ky):
                # hold.frame is the index of the detected hold's centre,
                # already DELAY frames in the past
                crop_and_classify(hold)
        for hold in seg.finish():                # flush the tail at end of stream
            ...

    Emits a hold as soon as it is confirmed, i.e. `delay` frames after it
    occurred. Holds are emitted in temporal order and never retracted.
    """

    def __init__(self, fps: float = TUNED_FPS, smooth_sec: float = SMOOTH_SEC,
                 run_sec: float = RUN_SEC, min_sep_sec: float = MIN_SEP_SEC,
                 seg_len_sec: float = SEG_LEN_SEC):
        self.fps = float(fps)
        self.smooth_win = frames_for(smooth_sec, fps, odd=True)
        self.run = frames_for(run_sec, fps)
        self.min_sep = frames_for(min_sep_sec, fps)
        self.seg_len = frames_for(seg_len_sec, fps, odd=True)
        self.delay = max((self.smooth_win - 1) // 2, self.run, self.min_sep)

        self._raw = []          # raw OFI per frame
        self._smooth = []       # centred-smoothed OFI, valid up to t - half
        self._prev = None       # previous frame's keypoints
        self._n = 0             # frames pushed
        self._half = (self.smooth_win - 1) // 2

        # candidate minima awaiting confirmation: list of (frame_index, ofi)
        self._pending = deque()
        self._last_emitted = -10 ** 9
        self._scanned_to = -1   # highest frame index tested for minimum-ness

    # -- internals ---------------------------------------------------------
    def _push_ofi(self, kx, ky):
        """Append this frame's raw motion intensity."""
        if kx is None or ky is None:
            # MediaPipe lost the hand. Offline we interpolate; live we cannot
            # see the future, so carry the last pose forward (zero motion) and
            # let the hold logic treat it as stillness.
            self._raw.append(0.0 if self._prev is None else 0.0)
        else:
            cur = np.stack([np.asarray(kx, float), np.asarray(ky, float)], axis=1)
            if self._prev is None:
                self._raw.append(0.0)
            else:
                d = np.sqrt(((cur - self._prev) ** 2).sum(axis=1)).mean()
                self._raw.append(float(d))
            self._prev = cur
        self._n += 1

    def _update_smooth(self):
        """Centred rolling mean is defined for frames up to n-1-half."""
        target = self._n - 1 - self._half
        while len(self._smooth) <= target:
            i = len(self._smooth)
            lo = max(0, i - self._half)
            hi = min(self._n, i + self._half + 1)
            self._smooth.append(float(np.mean(self._raw[lo:hi])))

    def _scan_minima(self):
        """Find newly-confirmable local minima in the smoothed signal."""
        s = self._smooth
        limit = len(s) - self.run - 1      # last index whose +run window exists
        i = max(self._scanned_to + 1, self.run)
        while i <= limit:
            w = s[i - self.run: i + self.run + 1]
            c = s[i]
            if c <= min(w) + 1e-12 and w[0] > c and w[-1] > c:
                self._pending.append((i, c))
            self._scanned_to = i
            i += 1

    def _emit_ready(self, flush=False):
        """Confirm pending minima whose min_sep window has fully elapsed.

        Mirrors collapse_minima: chain candidates within min_sep, keep the
        deepest. A candidate is safe to emit once no later candidate could
        still join its chain.
        """
        out = []
        while self._pending:
            frame, val = self._pending[0]
            # the chain starting here is closed when we have scanned beyond
            # frame + min_sep, or the stream has ended
            if not flush and self._scanned_to < frame + self.min_sep:
                break
            # gather the chain
            chain = [self._pending.popleft()]
            while self._pending and self._pending[0][0] - chain[-1][0] <= self.min_sep:
                if not flush and self._scanned_to < self._pending[0][0] + self.min_sep:
                    # chain may still grow -- put it back and wait
                    self._pending.appendleft(chain.pop())
                    while chain:
                        self._pending.appendleft(chain.pop())
                    return out
                chain.append(self._pending.popleft())
            best = min(chain, key=lambda t: t[1])
            if best[0] - self._last_emitted > self.min_sep:
                self._last_emitted = best[0]
                half = self.seg_len // 2
                out.append(Hold(frame=best[0], ofi=best[1],
                                start=max(0, best[0] - half), end=best[0] + half))
        return out

    # -- public API --------------------------------------------------------
    def push(self, kx=None, ky=None):
        """Feed one frame. Returns a (possibly empty) list of confirmed Holds."""
        self._push_ofi(kx, ky)
        self._update_smooth()
        self._scan_minima()
        return self._emit_ready()

    def finish(self):
        """End of stream: flush whatever is still pending."""
        self._update_smooth()
        # smoothing tail: fill remaining frames with shrinking windows
        while len(self._smooth) < self._n:
            i = len(self._smooth)
            lo = max(0, i - self._half)
            self._smooth.append(float(np.mean(self._raw[lo:self._n])))
        self._scan_minima()
        return self._emit_ready(flush=True)

    @property
    def latency_ms(self):
        return 1000.0 * self.delay / self.fps


class Hold:
    """A confirmed letter hold."""
    __slots__ = ("frame", "ofi", "start", "end")

    def __init__(self, frame, ofi, start, end):
        self.frame, self.ofi, self.start, self.end = frame, ofi, start, end

    def __repr__(self):
        return f"Hold(frame={self.frame}, ofi={self.ofi:.5f}, span=({self.start},{self.end}))"
