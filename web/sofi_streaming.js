/**
 * Causal SOFI Algorithm 2 — JavaScript port of sofi_streaming.py.
 *
 * Runs in the browser on MediaPipe keypoints so no video ever leaves the
 * client. Verified frame-for-frame against the Python implementation on real
 * PSL-CFRT clips (see web/test_parity.mjs).
 *
 * Parameters are DURATIONS, not frame counts. The offline algorithm was tuned
 * on 50 fps dataset video; a webcam is usually 30 fps. Reusing the tuned frame
 * counts directly degrades segment-count MAE from ~1.2 to ~2.6 at 30 fps and
 * ~4.6 at 15 fps. Converting durations at runtime holds MAE at ~1.2 across
 * 50/30/25 fps. This is measured in streaming_segmentation.ipynb.
 */

export const TUNED_FPS = 50.0;
export const SMOOTH_SEC = 13 / TUNED_FPS;   // 0.26 s
export const RUN_SEC = 7 / TUNED_FPS;       // 0.14 s
export const MIN_SEP_SEC = 15 / TUNED_FPS;  // 0.30 s
export const SEG_LEN_SEC = 11 / TUNED_FPS;  // 0.22 s

function framesFor(seconds, fps, odd = false, minimum = 1) {
  let n = Math.max(minimum, Math.round(seconds * fps));
  if (odd && n % 2 === 0) n += 1;
  return n;
}

function mean(arr, lo, hi) {
  let s = 0;
  for (let i = lo; i < hi; i++) s += arr[i];
  return s / (hi - lo);
}

export class StreamingSegmenter {
  /**
   * @param {object} opts
   * @param {number} opts.fps        camera framerate
   * @param {number} [opts.smoothSec] centred smoothing window, seconds
   * @param {number} [opts.runSec]    minimum-window half-width, seconds
   * @param {number} [opts.minSepSec] minimum separation between holds, seconds
   * @param {number} [opts.segLenSec] emitted hold span, seconds
   */
  constructor({ fps = TUNED_FPS, smoothSec = SMOOTH_SEC, runSec = RUN_SEC,
                minSepSec = MIN_SEP_SEC, segLenSec = SEG_LEN_SEC } = {}) {
    this.fps = fps;
    this.smoothWin = framesFor(smoothSec, fps, true);
    this.run = framesFor(runSec, fps);
    this.minSep = framesFor(minSepSec, fps);
    this.segLen = framesFor(segLenSec, fps, true);
    this.half = (this.smoothWin - 1) >> 1;
    this.delay = Math.max(this.half, this.run, this.minSep);

    this.raw = [];
    this.smooth = [];
    this.prev = null;
    this.n = 0;
    this.pending = [];
    this.lastEmitted = -1e9;
    this.scannedTo = -1;
  }

  /** Inherent latency of the method, in milliseconds. */
  get latencyMs() { return 1000 * this.delay / this.fps; }

  /**
   * Feed one frame.
   * @param {Float64Array|number[]|null} kx 21 x-coords, or null if no hand
   * @param {Float64Array|number[]|null} ky 21 y-coords, or null if no hand
   * @returns {Array<{frame:number, ofi:number, start:number, end:number}>}
   */
  push(kx, ky) {
    this._pushOfi(kx, ky);
    this._updateSmooth();
    this._scanMinima();
    return this._emitReady(false);
  }

  /** End of stream: flush anything still pending. */
  finish() {
    this._updateSmooth();
    while (this.smooth.length < this.n) {
      const i = this.smooth.length;
      this.smooth.push(mean(this.raw, Math.max(0, i - this.half), this.n));
    }
    this._scanMinima();
    return this._emitReady(true);
  }

  _pushOfi(kx, ky) {
    if (kx == null || ky == null) {
      // MediaPipe lost the hand. We cannot interpolate forward in a live
      // stream, so treat it as stillness and carry the last pose.
      this.raw.push(0.0);
    } else if (this.prev === null) {
      this.raw.push(0.0);
      this.prev = [Float64Array.from(kx), Float64Array.from(ky)];
    } else {
      const [px, py] = this.prev;
      let s = 0;
      for (let i = 0; i < kx.length; i++) {
        const dx = kx[i] - px[i], dy = ky[i] - py[i];
        s += Math.sqrt(dx * dx + dy * dy);
      }
      this.raw.push(s / kx.length);
      this.prev = [Float64Array.from(kx), Float64Array.from(ky)];
    }
    this.n += 1;
  }

  _updateSmooth() {
    const target = this.n - 1 - this.half;
    while (this.smooth.length <= target) {
      const i = this.smooth.length;
      this.smooth.push(mean(this.raw, Math.max(0, i - this.half),
                            Math.min(this.n, i + this.half + 1)));
    }
  }

  _scanMinima() {
    const s = this.smooth;
    const limit = s.length - this.run - 1;
    let i = Math.max(this.scannedTo + 1, this.run);
    for (; i <= limit; i++) {
      const c = s[i];
      let isMin = true;
      for (let j = i - this.run; j <= i + this.run; j++) {
        if (s[j] < c - 1e-12) { isMin = false; break; }
      }
      if (isMin && s[i - this.run] > c && s[i + this.run] > c) {
        this.pending.push([i, c]);
      }
      this.scannedTo = i;
    }
  }

  /**
   * Mirrors collapse_minima: chain candidates within minSep, keep the deepest.
   * A chain is only safe to resolve once no later candidate could still join.
   */
  _emitReady(flush) {
    const out = [];
    while (this.pending.length) {
      const [frame] = this.pending[0];
      if (!flush && this.scannedTo < frame + this.minSep) break;

      const chain = [this.pending.shift()];
      let stalled = false;
      while (this.pending.length &&
             this.pending[0][0] - chain[chain.length - 1][0] <= this.minSep) {
        if (!flush && this.scannedTo < this.pending[0][0] + this.minSep) {
          // chain may still grow — put everything back and wait
          while (chain.length) this.pending.unshift(chain.pop());
          stalled = true;
          break;
        }
        chain.push(this.pending.shift());
      }
      if (stalled) return out;

      let best = chain[0];
      for (const c of chain) if (c[1] < best[1]) best = c;
      if (best[0] - this.lastEmitted > this.minSep) {
        this.lastEmitted = best[0];
        const h = this.segLen >> 1;
        out.push({ frame: best[0], ofi: best[1],
                   start: Math.max(0, best[0] - h), end: best[0] + h });
      }
    }
    return out;
  }
}
