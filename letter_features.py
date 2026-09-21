"""
Frame-level features for letter (handshape) classification.

Two feature families, both computed from what the CSV already contains:

1. `normalized_keypoints` -- the 21 MediaPipe landmarks put into a canonical
   hand frame: wrist translated to the origin, scaled by the rigid
   wrist->middle-MCP span, and rotated so that span points "up". This removes
   where the hand is, how big it is, and how it is rotated in the image plane,
   leaving (in principle) only its shape.

2. `robust_distance_features` -- the CSV's 25 engineered `d_` distances, log1p
   and clipped. Raw, they are unusable: they range to 977 because they are
   ratios whose denominator collapses when the palm is seen near edge-on. Those
   blow-ups are concentrated in signer5-R (1256 of 1396 affected rows) and
   signer4-L, i.e. exactly the tilted camera angles. Left raw, a handful of
   frames dominate any distance metric or z-score.

IMPORTANT NEGATIVE RESULT (see letter_classifier.ipynb): neither family, nor
both together, supports letter classification that transfers across signers.
A RandomForest trained on these features predicts *signer identity* at ~97 %
while predicting letters below the majority-class baseline. The normalization
above removes position, scale and in-plane rotation, but not hand morphology or
out-of-plane viewpoint -- and those apparently dominate. The module is kept as
the documented baseline that the appearance-based approach has to beat.
"""
import numpy as np
import pandas as pd

KP_COLS_X = [f"kp{i}_x" for i in range(21)]
KP_COLS_Y = [f"kp{i}_y" for i in range(21)]

WRIST, MIDDLE_MCP = 0, 9
D_CLIP = 20.0


def distance_columns(df: pd.DataFrame):
    return [c for c in df.columns if c.startswith("d_")]


def normalized_keypoints(sub: pd.DataFrame) -> np.ndarray:
    """(n_frames, 40) wrist-centred, scale- and rotation-normalised landmarks.

    kp0 is dropped because it is (0, 0) by construction after centring.
    """
    X = sub[KP_COLS_X].to_numpy(dtype=float)
    Y = sub[KP_COLS_Y].to_numpy(dtype=float)

    X = X - X[:, [WRIST]]
    Y = Y - Y[:, [WRIST]]

    span = np.sqrt(X[:, MIDDLE_MCP] ** 2 + Y[:, MIDDLE_MCP] ** 2)
    span[span < 1e-6] = 1e-6
    X = X / span[:, None]
    Y = Y / span[:, None]

    # rotate the wrist->middle-MCP vector onto +y
    ang = np.arctan2(Y[:, MIDDLE_MCP], X[:, MIDDLE_MCP]) - np.pi / 2
    c, s = np.cos(-ang), np.sin(-ang)
    Xr = c[:, None] * X - s[:, None] * Y
    Yr = s[:, None] * X + c[:, None] * Y
    return np.hstack([Xr[:, 1:], Yr[:, 1:]])


def robust_distance_features(sub: pd.DataFrame, cols=None) -> np.ndarray:
    """log1p of the clipped `d_` distances -- see module docstring."""
    cols = cols or distance_columns(sub)
    return np.log1p(np.clip(sub[cols].to_numpy(dtype=float), 0, D_CLIP))


def build_features(sub: pd.DataFrame, use_keypoints=True, use_distances=True) -> np.ndarray:
    parts = []
    if use_keypoints:
        parts.append(normalized_keypoints(sub))
    if use_distances:
        parts.append(robust_distance_features(sub))
    if not parts:
        raise ValueError("at least one feature family must be enabled")
    return np.hstack(parts)


def feature_names(df: pd.DataFrame, use_keypoints=True, use_distances=True):
    names = []
    if use_keypoints:
        names += [f"nkp{i}_x" for i in range(1, 21)] + [f"nkp{i}_y" for i in range(1, 21)]
    if use_distances:
        names += [f"log_{c}" for c in distance_columns(df)]
    return names


def occurrence_table(frames: pd.DataFrame, df: pd.DataFrame, transition="<TRANS>",
                     use_keypoints=True, use_distances=True):
    """Collapse pseudo-labelled frames into one mean feature vector per
    (signer, clip, letter) occurrence.

    Frames inside one hold are near-duplicates, so treating them as independent
    samples inflates any accuracy estimate enormously. The occurrence is the
    real unit of evidence, and even occurrences must be grouped by clip when
    splitting.
    """
    held = frames[(frames["label"].notna()) & (frames["label"] != transition)]
    merged = held.merge(df, on=["signer_id", "video_id", "frame_id"], how="left")
    merged = merged[merged["hand_detected"] == 1].reset_index(drop=True)

    F = build_features(merged, use_keypoints, use_distances)
    grouped = pd.DataFrame(F).groupby(
        [merged["signer_id"], merged["video_id"], merged["label"]]).mean()
    meta = pd.DataFrame({
        "signer_id": grouped.index.get_level_values("signer_id"),
        "video_id": grouped.index.get_level_values("video_id"),
        "letter": grouped.index.get_level_values("label"),
    })
    return grouped.to_numpy(), meta
