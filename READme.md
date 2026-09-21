# PSL Continuous Fingerspelling Recognition — DRP README

## 📌 Current Status / Progress Log

*(Last updated: this week. Update this section as work progresses — treat it as a running log, not a one-time summary.)*

**What we have in hand:**
- Full PSL-CFRT video corpus: 353 clips across 5 signers (70–73 each), as **hand-cropped** clips (`PSL_Fingerspelling_Video_Dataset_Cropped/`) and a mirrored/augmented copy (`..._Mirrored/`), 240×350, 50fps.
- `hand_keypoints_features.csv` — per-frame MediaPipe hand keypoints (21 landmarks × x/y), bounding boxes, detection confidence, and ~25 engineered geometric distance features, for every frame of every clip.
- **Not yet received:** the letter-level annotated frame set (~2,913 YOLO-format images with per-letter bounding boxes + class IDs) that the published pipeline was actually trained on, and the noun→letter-sequence mapping behind the PSL-CFRT paper's Fig. 3 frequency table. Still the blocker for *reproducing the group's published baseline* (checklist item 1) — currently following up with Kamran. Note this is **no longer the blocker for making progress**: we now build our own letter-level labels (Progress Log #7), and the current bottleneck is the feature representation, not the labels (#8).

**What we've done with it so far:**
1. **Data quality audit.** MediaPipe hand detection is near-100% for 4 of 5 signers, but **signer5 (recorded from the "converse" 45° angle) has only ~50% detection overall, with 16 of 73 clips at 0% detection across every frame**. Visually confirmed this isn't corrupted footage — the hand is clearly visible in those frames, but held directly in front of the (blurred) face, which appears to be a genuine MediaPipe hand-vs-face occlusion failure specific to this signer's signing style. This is a concrete, quantified instance of the viewpoint-sensitivity problem flagged in weakness #2 — worth citing directly when we get to the cross-angle evaluation.
2. **Segmentation prototype (SOFI-style).** Since we don't have shoulder/pose landmarks (only hand keypoints), WNSS isn't reproducible from this CSV as-is. Built a SOFI-style proxy instead — frame-to-frame hand-keypoint displacement standing in for Farneback optical flow — and implemented the paper's fixed-threshold segmentation (Algorithm 1). **Result: a single global threshold does not generalize across words** (segment counts don't track word length consistently, no threshold value works across multiple test videos). This independently reproduces the motivation the original authors gave for moving to their dynamic Algorithm 2. **Next step, not yet done:** implement the dynamic/local-minima version.
3. **Word-level classification baseline.** As a first trainable model (before letter-level labels exist), trained a RandomForest to predict *which word* was signed from aggregated per-video keypoint features, comparing leave-one-signer-out vs. stratified splits. **Result: both land near chance (~1–4%, vs. ~1.2% chance level)** — not because the approach is broken, but because each of the 86 words only has ~3–4 examples total across all signers. **Key implication:** word-level classification is structurally low-data by construction; letter-level classification is much better posed, since individual letters repeat dozens to hundreds of times across the corpus (e.g. 'ی' appears 455 times per the PSL-CFRT frequency table). This reinforces that getting/building frame-level letter labels should be the immediate priority over further baseline tuning.
4. **Repo set up on GitHub**, with `.gitignore` excluding the raw video folders/zips and the CSV (kept local/shared via Drive, not git, due to size) and excluding working `.py` scripts in favor of the notebook versions as the source of truth.

5. **Word canonicalization + letter decomposition** (`canonicalize_words.py`, `decompose_letters.py`). The 87 raw filenames contain both Unicode codepoint variants that render identically (Arabic Kaf `ك` vs Urdu Keheh `ک`) and genuine transcription inconsistencies, silently fragmenting classes. After canonicalization: **72 unique words** (not 74 — two aliases were written in pre-normalization form and could never match; fixed). 22 are acronym-style (`او پی ڈی` = O.P.D) whose tokens are spelled-out *English* letter names, so they are held out pending Kamran's mapping. The remaining **54 plain words** decompose into ordered letter sequences covering **40 codepoints = 37 alphabet letters + 3 hamza/madda forms** (`آ ؤ ئ`), which explains the gap against the paper's 38-letter alphabet — whether PSL gives these their own handshape is an open question for the authors.

   *Subtlety worth remembering:* canonicalization has **two incompatible jobs**. For word *identity* you want `سٹینڈرڈ چارٹرڈ بنک` merged into `سٹینڈرڈ چارٹرڈ`; for letter *labels* you must not, because that clip's signer demonstrably spelled 16 letters, not 13. Hence `canonicalize()` vs `spelling_for_labels()`, and 4 clips flagged `is_ambiguous_spelling`.

6. **SOFI Algorithm 2 implemented and validated** (`sofi_dynamic.py`, `sofi_algorithm2.ipynb`). Detects letters as *local minima* of the motion signal — scale-free, so no global threshold. Two findings:
   - The paper's spec read literally (≥5 strictly decreasing then ≥5 increasing frames) is far too brittle: it finds ~5.7 **fewer** segments than the word has letters. A tolerant form (frame is the minimum of a ±7 window, both endpoints strictly above it) is essentially unbiased.
   - **Algorithm 2 beats Algorithm 1 by ~3.5×**: segment-count MAE **1.19 vs 4.28**, exact-count **32% vs 6%**, corr(detected, true) **0.79 vs 0.21** — and Algorithm 1 was given an *oracle* threshold tuned on all 218 test clips while Algorithm 2 used fixed defaults. Holds up leave-one-signer-out (mean held-out MAE 1.23 vs 1.19 in-sample; tuned params near-identical for every held-out signer).

7. **Frame-level pseudo-labels built** (`pseudo_label.py`, `pseudo_labeling.ipynb`). Key insight: labelling is not inference — we *know* the letter count at labelling time, so the segmenter is **constrained** to emit exactly k holds via DP (min total motion subject to a separation constraint), removing count error from the labelling path entirely. Output: **218 clips, 41,930 usable frames, 14,530 letter-hold frames, 40 letter classes**, plus per-clip quality diagnostics and 20 contact sheets in `spot_check/` for human review. Automated checks pass: holds are 2.1× quieter than transitions in 100% of clips; picks are evenly paced.
   - **Known failure mode, now auto-flagged:** when a word has more letters than the clip has room for, the DP tiles the clip at its separation floor instead of finding troughs (e.g. a 16-letter word in 150 frames → picks exactly 9 frames apart, landing mid-transition). 6 of 218 clips flagged `constraint_dominated`.
   - **Second limitation, measured but deliberately not yet fixed:** picks are only kept 7 frames from the clip edges, which at 50fps is too little — the hand is still entering/leaving frame, so the motion signal is low for the wrong reason. ~1 clip in 6 has a boundary pick within 0.3s of an edge (34/218 last picks, 38/218 first picks). Raising the margin to ~15 frames is the obvious fix but can make short clips infeasible, and it shouldn't be applied to every label in the dataset before the human spot-check validates the tradeoff.

8. **Letter classifier trained — and it does not generalize across signers.** This is the most important result so far, and it is negative:
   - Leave-one-signer-out letter accuracy is **6.8% against a 9.1% majority-class baseline** — *below* always guessing the commonest letter. True for both RandomForest and MLP, and for every held-out signer.
   - The same features predict **signer identity at 97%**. They encode *whose hand it is* far more strongly than *what shape it is in*.
   - Within-signer (words held out, signer confound gone), 4 of 5 signers score inside their own label-shuffle control. **signer4-L is the exception at 36% vs a 10% baseline with shuffle controls at 4–10%** — so the pseudo-labels *can* carry real handshape information; the representation is what fails to transfer. Dropping the 6 `constraint_dominated` clips lifts signer1-F 0.44×→0.71× and signer3-F 0.78×→0.87×, but nothing except signer4-L crosses 1.0× — removing known-bad labels sharpens the picture without changing it, which is itself evidence that label noise is not the binding constraint.
   - **Frame-level random splits report 74.6% — a 10.9× inflation** over the honest occurrence-level, signer-grouped 6.8%. Frames inside one hold are near-duplicates. This is exactly the illusion the group's third paper is about, reproduced inside the continuous pipeline.

   *Diagnosis:* the bottleneck is **representation, not segmentation**. 2D MediaPipe landmarks, even canonicalized for position/scale/in-plane rotation, do not isolate handshape — finger configuration and contact project ambiguously into 2D, differently per signer and camera angle.

   *Negative control worth keeping:* `mean_pick_depth` looks like a great label-quality gate (deepest quartile 25.9% vs shallowest 9.0%) — but the effect **vanishes entirely** when signer4-L is excluded (4.7% → 8.6%, no trend), because it supplies 184 of 270 occurrences in that quartile. Do not use it as a filter on the strength of the pooled table.

9. **Stage-1 demo app built: live segmentation in the browser** (`sofi_streaming.py`, `streaming_segmentation.ipynb`, `web/`). Algorithm 2 is offline — centred smoothing, a ±`run` minimum test, and `min_sep` of lookahead before a minimum can be confirmed — so it cannot be ported to a live loop naively. Two findings, both of which would have quietly broken a demo:
   - **Causal port costs almost nothing.** Running the same algorithm on a fixed 15-frame (300ms @50fps) delay matches the offline version on 87% of clips exactly, with 97% of matched holds landing on the identical frame, and segment-count MAE 1.23 vs 1.19. **Real-time is not what limits this pipeline.**
   - **The parameters are frame counts tuned at 50fps, and a webcam is 30fps.** Reusing them as frame counts degrades MAE to **2.61 at 30fps** (exact-count 32% → 2.8%) and **4.56 at 15fps — worse than the Algorithm 1 baseline we're trying to beat**. Expressing them as durations and converting at runtime holds MAE at ~1.2 across 50/30/25fps. This is a unit bug that would have looked like "the demo just isn't very good."

   Architecture is browser-side: MediaPipe Hands in the tab → keypoints → segmenter in JS → only cropped hold frames go to a backend. No video leaves the client. The JS port is verified byte-identical to Python across 50 cases / 9,958 frames (`web/test_parity.mjs`, re-run automatically by the notebook). The page deliberately does **not** show letter predictions, since there is no model that honestly generalizes yet.

10. **Live testing exposed three failures the dataset could not.** First run on a real camera gave **47 holds in one session at 5fps**, thumbnails showing a resting hand. Root cause is that dataset clips are pre-trimmed to active fingerspelling, while a camera also sees an empty frame, a hand arriving, and a hand idling — the unmodified algorithm reports holds for all three.
    - **Degenerate params at low fps.** At 5fps the tuned durations round to `smooth=1, run=1, min_sep=2` — no smoothing, and a window so short almost any frame qualifies. Added hard floors and a `viable` flag; below 14fps the app declines to report. The 5fps itself was MediaPipe running on the full 960×720 capture every tick — now downscaled to 320×240, with the GPU/CPU delegate reported rather than silently falling back.
    - **Hand entering/leaving manufactures holds.** Absence was fed in as zero motion; the spikes on either side leave a trough that looks like a hold. The obvious fix (reject any window containing an undetected frame) was *too* blunt — it cost MAE 1.23 → 1.45, because the clips it hit were signer5-R mid-signing dropouts that offline interpolation handles correctly. Now gated on gap *length*: sustained absence suppresses, brief blinks don't.
    - **An idle hand still produces holds** (~1.5/sec). Nothing in Algorithm 2 asks whether signing is happening. Added a scale-free contrast gate — the live counterpart of `mean_pick_depth` — requiring peak motion in the window to exceed the minimum by ≥1.10×. Measured over the 218 clips vs simulated idle jitter: **keeps 98.5% of real holds, rejects 93% of idle ones**, dropping false positives from 1.50/sec to 0.20/sec at no cost to count accuracy (MAE 1.23 → 1.25).

    Worth noting for the writeup: none of these were findable offline. The streaming version is now deliberately *stricter* than the batch one rather than a pure port, which is the correct relationship — it sees input the offline algorithm was never given.

11. **Per-letter reference sheets — and they resolve the fork** (`letter_reference.ipynb`, `letter_sheets/`). The per-clip spot-check sheets asked "is segment *i* really letter *i*?", which needs PSL expertise nobody on this project has. The per-letter sheets ask a question anyone can answer: *do these images show the same handshape?* Each sheet collects every occurrence of one letter across different words and different signers.

    **Verdict: the pseudo-labels are good.** `چ` shows 8 occurrences across 4 signers and 2 different words, at different positions within those words — all visibly the same handshape. This was the open question blocking everything: whether `letter_classifier.ipynb`'s failure meant bad labels or bad features. It's the features.

    The cleanest evidence is `پ`, which scores **worst** on feature-space self-consistency (ratio 0.55) yet is visually one of the most consistent sheets — 5 signers, same pose. A label problem cannot look like that; only a representation problem can. Caveat: `پ` occurs in one word only, so it tests cross-signer but not cross-word generalization; `چ` tests both.

    Also visible in `_alphabet.png`: a large family of letters that are near-identical fist variants in 2D, differing only in thumb position and finger contact. That is a direct visual explanation for why 2D landmark geometry cannot separate them. Two exemplars are bad picks (`ت` catches motion blur, `ج` catches a face) — the dataset's "cropped" videos are not uniformly hand-cropped.

**Immediate next actions (in progress or queued):**
- [ ] **Train a small CNN on the hand crops at the segmented hold frames** — the crops already exist, the segmenter already says which frames to use, and the result above says appearance is where the missing signal must be. This is the direct next experiment.
- [ ] Human spot-check of `spot_check/*.png` (needs someone who reads Urdu + knows PSL handshapes) to put a number on pseudo-label noise. Two sheets inspected informally look clean — distinct, settled handshapes matching the captions.
- [ ] Measure signer-invariance explicitly (the 97% signer accuracy is a ready-made metric); consider gradient-reversal / adversarial signer removal.
- [ ] Still chasing Kamran for the letter-level annotated frames **and** the noun→letter-sequence mapping for the 22 acronym words (~30% of vocabulary, and they disproportionately contain letters that are rare elsewhere).

---

## 🗂️ Repo Map

Run the notebooks in this order; each depends on the artifacts of the previous one.

| File | What it is |
|---|---|
| `canonicalize_words.py` | Word normalization. **`canonicalize()` for identity, `spelling_for_labels()` for letter labels — they are not interchangeable** (see Progress Log #5) |
| `decompose_letters.py` | Builds `word_letter_mapping.json` (54 plain words → ordered letter sequences). Run it first |
| `sofi_dynamic.py` | SOFI Algorithm 2: OFI signal, local-minima detection, segmentation |
| `pseudo_label.py` | Constrained k-minima DP alignment → frame-level labels + quality diagnostics |
| `letter_features.py` | Normalized-keypoint and robust `d_` feature construction; occurrence-level aggregation |
| `sofi_segmentation.ipynb` | *(earlier)* Algorithm 1, fixed threshold — kept as the brittle baseline |
| `word_baseline.ipynb` | *(earlier)* word-level RandomForest, near chance by construction |
| `sofi_algorithm2.ipynb` | Algorithm 2 validation: strict vs tolerant, parameter sweep, vs Algorithm 1, LOSO |
| `pseudo_labeling.ipynb` | Builds the pseudo-labels, diagnostics, exports, and the spot-check contact sheets |
| `letter_classifier.ipynb` | Letter classification + the negative result and its controls |
| `letter_reference.ipynb` | Per-letter sheets + the alphabet reference; the label-verification step |
| `letter_sheets/` | One sheet per letter, plus `_alphabet.png` (one exemplar each) |
| `sofi_streaming.py` | **Causal** Algorithm 2 for live input. Parameters in *seconds*, not frames |
| `streaming_segmentation.ipynb` | Streaming-vs-offline parity, framerate transfer, browser parity test |
| `web/` | Stage-1 demo app: live segmentation in the browser (`web/README.md` to run it) |
| `pseudo_labels.csv` | Frame → letter / `<TRANS>` / null. `frame_id` indexes the video directly |
| `pseudo_label_clips.csv` | Per-clip diagnostics (`alignment_confidence`, `mean_pick_depth`, `constraint_dominated`, quality score) |
| `pseudo_label_segments.csv` | Per-letter segment bounds and chosen minimum frame |
| `spot_check/` | Contact sheets + `review.csv` template awaiting human verdicts |

**Environment note:** needs `opencv-python-headless` (contact sheets) in addition to pandas/numpy/scikit-learn/matplotlib. The videos live at `PSL_Fingerspelling_Video_Dataset_Cropped/PSL_Fingerspelling_Video_Dataset_Cropped/<signer>/<word>.MP4` (the folder is nested one level deeper than you'd expect).

---

## What the Existing Project Currently Is

Our research group (Kamran, Baig, Awais, et al.) already has a working, published pipeline. This DRP is **not starting from scratch** — it's about taking this existing system and making it meaningfully better. The current pipeline, across the group's papers, looks like this:

**Dataset:** PSL-CFRT — 350 short video clips, 5 signers (3F/2M), 3 camera views (front, +45°, -45°, but each signer only recorded from *one* fixed angle), 38 Urdu letters, continuous (unsegmented) fingerspelling of common nouns.

**Pipeline (current):**
1. **Video segmentation** (heuristic, hand-engineered): three methods — Wrist-Near-to-Shoulder (WNSS, fixed 250px threshold), Hand Region Segmentation (HRS, via MediaPipe crop), and Optical-Flow-Intensity segmentation (SOFI, fixed threshold of 2.5, Farneback optical flow).
2. **Sign detection/recognition**: YOLO (v8x, v11x) as the best performer, with ResNet-50 and VGG as image-classification alternatives.
3. **Post-processing / translation**: two heuristic algorithms — either require a letter to be detected 10+ consecutive times, or take the statistical mode of detections within a segment — to convert frame-level predictions into a letter sequence, then a translated Urdu string.
4. **Evaluation**: Character Error Rate (CER) → Accuracy = 1 − CER.

**Best current results:** ~63–68% translation accuracy on front-facing video, dropping to ~48% on tilted (35–45°) angles. Sequence-to-sequence/RNN-style models were deliberately avoided for speed reasons, in favor of frame-level detection + heuristic post-processing.

This README is about **what's wrong with this pipeline and how we fix it** — not a re-summary of the papers.

---

## Diagnosed Weaknesses in the Current System (What We're Fixing)

1. **Heuristic, hand-tuned segmentation** — WNSS (250px) and SOFI (2.5 threshold) are fixed, manually-tuned constants calibrated on this one small dataset. They will not transfer to new signers, cameras, or signing speeds. This is a brittle, non-learned component sitting in the middle of an otherwise deep-learning pipeline. **(Confirmed empirically — see Progress Log #2 above: a fixed SOFI-style threshold does not generalize across words in our own re-implementation.)**

2. **Severe viewpoint sensitivity** — accuracy drops ~20 points (67.84% → 48.07%) just from a 35–45° camera angle change. Since only 2 of 5 signers were recorded off-angle, the model has barely seen angle variation at training time. This is a **generalization gap problem**, structurally identical to the Illusion-to-Reality Gap found in the isolated-letter study — but no one has measured it properly here yet (no LODO-style, no cross-signer, no cross-angle held-out evaluation has been run on the continuous pipeline). **(Confirmed a related, earlier-stage symptom of this ourselves — see Progress Log #1: MediaPipe hand detection itself degrades to ~50% for the off-angle signer, before any recognition model is even involved.)**

3. **No cross-dataset / cross-signer generalization testing at all** — unlike the isolated-letter benchmark, the continuous pipeline has only ever been evaluated on splits of the *same* 5-signer, single dataset. We don't actually know how it performs on an unseen signer, camera, or environment.

4. **Naive post-processing heuristics** — "10 consecutive identical detections" and "statistical mode per segment" are brittle rules that explicitly fail when signing speed is high (an observation the authors themselves noted: accuracy is inversely proportional to signing speed). A learned temporal model (CTC, a lightweight sequence decoder, or even a simple BiLSTM/temporal transformer head on top of frame features) would remove this fragility — the seq2seq option was rejected for speed, but wasn't the only sequence-modeling option available.

5. **Dynamic/motion letters are entirely excluded** — signs with inherent motion (e.g., Jeem, Alifmad) are not modeled at all. This means the system only handles a subset of the real PSL alphabet, which is a real deployment blocker.

6. **Small, low-diversity dataset** — 350 clips, 5 signers total, uniform green background, studio lighting. No dataset-level solution to this is proposed yet (e.g., synthetic augmentation, sim-to-real, or merging with the isolated-letter datasets from the companion project). **(Our own word-level baseline experiment put a number on this: ~3-4 samples per word class, which is why that baseline landed near chance — see Progress Log #3.)**

7. **Confusable sign pairs** — visually similar letters (Daal/Zaal, Zwad/Zo'ain) are a known, unaddressed failure mode. No fine-grained/hard-negative-mining strategy has been tried.

8. **Object-detection framing requires expensive bounding-box annotation** — every training frame needs a hand bounding box (LabelImg). A classification-only or landmark-based (MediaPipe keypoints) approach could reduce annotation cost significantly and may also reduce background dependence (an issue explicitly linked to poor generalization in the isolated-letter study).

9. **Multiple mutually-exclusive design choices never systematically compared** — the paper states WNSS/NS and the two SOFI algorithms are mutually exclusive by design, and only a few hand-picked combinations were reported. No full ablation grid or statistically validated comparison exists.

10. **Single-run, no confidence intervals anywhere** — same weakness as the isolated-letter study; all numbers are one run, no variance reported.

---

## 🎯 DRP Goal

> Turn the existing heuristic, viewpoint-fragile, single-dataset continuous PSL fingerspelling pipeline into a **more robust, better-generalizing, and higher-accuracy system**, validated with a rigorous cross-signer/cross-angle evaluation protocol (borrowing the LODO methodology from our own isolated-letter work), and ship it as a working live demo app.

In short: **replace heuristics with learned components, and prove generalization with proper evaluation — then demo it.**

---

## 🔴 Highest Priority — Do This First

1. **Get access to the PSL-CFRT raw videos + annotations** and reproduce the current best baseline (YOLOv8x + WNSS + HRS + Algo-1, ~63–68% accuracy) ourselves. Confirm we can match the published numbers before changing anything. **(Partial: we have the raw videos + keypoint CSV. Still missing the letter-level annotated frame set needed to actually train/reproduce YOLOv8x — following up with Kamran. In the meantime, prototyping our own segmentation/labeling approach in parallel — see Progress Log.)**

2. **Build a proper cross-signer / cross-angle held-out evaluation** — right now results are only reported per-angle on ad hoc splits. Set up a leave-one-signer-out (or leave-one-angle-out) protocol so we can honestly measure the *real* generalization gap in the continuous pipeline (this doesn't exist yet — it's the single most important missing piece, and it directly extends our isolated-letter LODO methodology). **(Partial: built and ran a leave-one-signer-out protocol for a word-level classification baseline as a proof of concept — not yet the actual continuous-pipeline evaluation this item is about, but the harness/methodology is in place and reusable once letter-level labels exist.)**

3. **Replace at least one heuristic component with a learned one** — the two highest-leverage candidates:
   - Swap the fixed-threshold WNSS/SOFI segmentation for a learned temporal boundary detector (even a simple 1D-CNN/BiLSTM trained to predict segment boundaries from optical-flow + landmark features).
   - Swap the "10 consecutive frames" / "mode" post-processing heuristic for a CTC-based decoder or lightweight temporal transformer head — this removes brittleness to signing speed without paying the full seq2seq inference cost the original authors were trying to avoid.

4. **Multi-seed, statistically reported results** for every experiment going forward (same fix identified for the isolated-letter project — don't repeat this mistake here).

---

## Secondary Priorities (after the above is working)

- Extend the pipeline to handle **dynamic/motion letters** (currently entirely excluded) — likely via short temporal windows around the current frame instead of single-frame classification.
- Try a **landmark-based (MediaPipe keypoints) classifier** instead of raw-pixel YOLO/ResNet/VGG — cheaper to annotate, and likely less prone to background/color confounds (directly relevant to the "hand color matches background" and "hand in front of face" failure modes already observed **and independently confirmed in our own data quality audit, Progress Log #1**).
- Address confusable letter pairs with targeted hard-negative mining or a hierarchical classifier (coarse handshape cluster → fine letter).
- Consider merging with the isolated-letter datasets (PSL Images, UAlpha40) as auxiliary pretraining data or for data augmentation, given both projects share the same alphabet and research group.
- Run a proper ablation grid over segmentation/post-processing combinations instead of the current hand-picked comparisons.

---

## 🖥️ Medium for Building the App

The end deliverable is a **live camera → segmented video → recognized letters → Urdu text** pipeline, so the demo needs real-time webcam input and a responsive UI, not just a static classifier demo.

| Layer | Recommendation | Why |
|---|---|---|
| **Model training** | PyTorch + Ultralytics (for YOLO) alongside a custom PyTorch temporal head (CTC/BiLSTM) for post-processing | Matches existing pipeline components, avoids retooling everything from scratch |
| **Fastest working demo** | **Gradio** with a webcam input component (streaming video/image from webcam) | Can show live segmentation + detection + translated Urdu text with minimal front-end work; good for showing "before" (heuristic pipeline) vs. "after" (learned pipeline) side-by-side |
| **Polished/sharable app** | Web app: React frontend (browser webcam via `getUserMedia`) + FastAPI backend serving the model, with a WebSocket stream for near-real-time frame-by-frame recognition | Needed if a smooth, sub-second-latency live demo is required for a proper defense/demo day; browsers handle continuous video capture much better than repeated file uploads |
| **If true real-time/offline/mobile is required** | Export detection model to **ONNX** or **TensorRT** for speed, or **TensorFlow Lite** for an on-device mobile version | Only worth the added complexity if live, low-latency, or offline deployment is explicitly part of the DRP scope (this pipeline is meant to eventually run in real time, so this is a realistic later step) |

**Superseded — see Progress Log #9.** Streaming video to a Python backend turned out to be the wrong design. MediaPipe Hands runs *in the browser*, so hand tracking and segmentation happen client-side and only the handful of cropped hold frames per word ever need a backend. That removes the WebSocket-video plumbing entirely, keeps latency low, and means no video leaves the user's machine. Stage 1 of this is built and running in `web/`.

---

## ✅ Summary Checklist (in order)

- [~] Get PSL-CFRT videos + annotations, reproduce current baseline — *videos + keypoint CSV in hand; letter-level annotations still pending from Kamran*
- [x] Build leave-one-signer-out / leave-one-angle-out evaluation protocol — *now applied for real: LOSO on segmentation (Algorithm 2) and on letter classification, with label-shuffle controls, occurrence-level grouping, and a demonstration that frame-level splits inflate results 10.9×*
- [~] Replace heuristic segmentation with a learned boundary detector — *Algorithm 2 (dynamic local-minima) built and validated, 3.5× better than Algorithm 1 and LOSO-stable. Still a heuristic, not learned — but it is now a strong baseline for a learned detector to beat*
- [x] Build letter-level labels without the missing annotations — *constrained-DP pseudo-labelling, 14,530 hold frames across 40 classes*
- [ ] **Appearance-based letter model (CNN on hand crops at hold frames)** — *the key open experiment, and no longer a gamble: the per-letter sheets confirm the labels are sound, so a CNN would be training on good data. Keypoint-geometry features are proven insufficient*
- [ ] Replace heuristic post-processing (consecutive-count / mode) with CTC or temporal transformer decoding
- [ ] Multi-seed runs + reported variance for all experiments
- [ ] Extend to dynamic/motion letters
- [ ] Try landmark-based classification to reduce background dependence
- [ ] Address confusable letter pairs
- [~] Build demo app — *Stage 1 done: live browser segmentation in `web/`, streaming port validated against the offline algorithm and across framerates. Stage 2 (letter CNN on hold crops) and Stage 3 (CTC decoding + per-user calibration) pending a working classifier*
- [ ] Write up: how much did we close the generalization gap vs. the original pipeline?

*(`[~]` = in progress / partially done, `[ ]` = not started, `[x]` = done — update as you go)*