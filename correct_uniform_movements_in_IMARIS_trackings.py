""" 

conda create -n manual_registration python=3.11 mamba -y -c conda-forge
conda activate manual_registration
mamba install -c conda-forge ipywidgets numpy matplotlib scikit-image tifffile napari pyside2 ipykernel xlrd -y

we also need SimpleITK:
mamba install -c conda-forge simpleitk -y
"""
# %% IMPORTS
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from dataclasses import dataclass
import SimpleITK as sitk
from __future__ import annotations
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from typing import Dict, Optional, Tuple, List
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
import networkx as nx
# %% PATHS
IMARIS_xls_files_path = "/Users/husker/Workspace/Joel Imaris Reg Corrections/IMARIS xls files/"
TIF_files_path =      "/Users/husker/Workspace/Joel Imaris Reg Corrections/TIF files/"
RESULTS_path =          "/Users/husker/Workspace/Joel Imaris Reg Corrections/Results/"
# Create results directory if it does not exist
os.makedirs(RESULTS_path, exist_ok=True)
# %% READ DATA
""" 
Read the results xls files exported from an IMARIS tracking analysis. The
data we are interested are on sheet "Position", e.g. columns:

Position								
Position X	Position Y	Position Z	Unit	Category	Collection	Time	TrackID	ID
469,71	384,56	62,50	µm	Spot	Position	1	1000043338	43338
531,92	398,90	70,89	µm	Spot	Position	1	1000043346	43346
519,48	372,82	285,97	µm	Spot	Position	1	1000043353	43353
553,84	494,13	287,50	µm	Spot	Position	1	1000043362	43362
531,03	417,93	192,75	µm	Spot	Position	1	1000043371	43371
494,93	462,04	180,68	µm	Spot	Position	1	1000043381	43381
474,71	485,43	121,95	µm	Spot	Position	1	1000043390	43390

Read the data into a pandas DataFrame for further processing.
"""

# scan the directory for xls files:
xls_files = [f for f in os.listdir(IMARIS_xls_files_path) if f.endswith('.xls') or f.endswith('.xlsx')]
# read each xls file into a pandas DataFrame and store in a dictionary
data_frames = {}
for xls_file in xls_files:
    print(f"Reading file: {xls_file}")
    file_path = os.path.join(IMARIS_xls_files_path, xls_file)
    # sheet name is "Position" and first row needs to be skipped:
    df = pd.read_excel(file_path, sheet_name='Position' , skiprows=1)
    data_frames[xls_file] = df
    
    print(f"  Unique timepoints in this file: {data_frames[xls_file]['Time'].unique()} (total: {len(data_frames[xls_file]['Time'].unique())})")
    print(f"  Number of unique TrackIDs in this file: {data_frames[xls_file]['TrackID'].nunique()}")
    print(f"  Number of unique IDs in this file: {data_frames[xls_file]['ID'].nunique()}")

data_frames[xls_files[0]]

# for testing: From data_frames[xls_files[0]], plot in 3D alle Time=1 points, colored by TrackID
for time in data_frames[xls_files[0]]['Time'].unique():
    print(f"Plotting all cells at timepoint {time} of file {xls_files[0]}")
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    df = data_frames[xls_files[0]]
    timepoint = time
    df_time = df[df['Time'] == timepoint]
    sc = ax.scatter(df_time['Position X'], df_time['Position Y'], df_time['Position Z'], c=df_time['TrackID'], cmap='jet')
    # move the colorbar outside the plot:
    plt.colorbar(sc, ax=ax, fraction=0.02, pad=0.1)
    ax.set_xlabel('X Position (µm)')
    ax.set_ylabel('Y Position (µm)')
    ax.set_zlabel('Z Position (µm)')
    ax.set_xlim(df['Position X'].min(), df['Position X'].max())
    ax.set_ylim(df['Position Y'].min(), df['Position Y'].max())
    ax.set_zlim(df['Position Z'].min(), df['Position Z'].max())
    ax.set_title(f'3D Scatter Plot of Time={timepoint} Points Colored by TrackID')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_path, f'3D_scatter_Time_{timepoint}_File_{xls_files[0]}.png'), dpi=300)
    plt.close(fig)

# plot alle tracks of each zell in one 3D plot:
print(f"Plotting all tracks over time of file {xls_files[0]}")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
df = data_frames[xls_files[0]]
for track_id in df['TrackID'].unique():
    df_track = df[df['TrackID'] == track_id]
    ax.plot(df_track['Position X'], df_track['Position Y'], df_track['Position Z'], label=str(track_id))
ax.set_xlabel('X Position (µm)')
ax.set_ylabel('Y Position (µm)')
ax.set_zlabel('Z Position (µm)')
ax.set_title(f'All Tracks Over Time Colored by TrackID')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_path, f'All_Tracks_3D_File_{xls_files[0]}.png'), dpi=300)
plt.close(fig)

# from data_frames, get for each unique TrackID the number of timepoints it appears in
# and plot in a histogram:
data_frames[xls_files[0]]['TrackID'].value_counts()
plt.figure()
track_lengths = data_frames[xls_files[0]]['TrackID'].value_counts()
plt.hist(track_lengths, bins=np.arange(1, track_lengths.max()+2)-0.5, edgecolor='black')
# annotate number of TrackIDs with length above each bin:
for length, count in track_lengths.value_counts().sort_index().items():
    plt.text(length, count + 0.5, str(count), ha='center')
plt.xlabel('Number of Timepoints per TrackID')
plt.ylabel('Number of TrackIDs')
plt.title(f'Histogram of Track Lengths in File:\n{xls_files[0]}')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_path, f'Histogram_Track_Lengths_File_{xls_files[0]}.png'), dpi=300)
plt.close()

# %% CLUSTER APPROACH

""" 
there are groups of cells, which perform very similar movements over time. and there are also 
some outliers ⟶ the actualy biological movements of cells. let's cluster the tracks based on 
their movement patterns, identify the main cluster(s) representing the uniform movement, first, 
and then we continue.
"""

for xls_file, df in data_frames.items():
    # xls_file = xls_files[0]
    print(f"\nProcessing file (clustering): {xls_file}")

    for col in ["Position X", "Position Y", "Position Z", "Time", "TrackID"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Position X", "Position Y", "Position Z", "Time", "TrackID"]).copy()
    
    # now cluster the XYZ-Tracks of each unique cell (TrackID) based on their movement patterns over time (Time):
    
    track_features = []
    track_ids = []
    for track_id in df["TrackID"].unique():
        df_track = df[df["TrackID"] == track_id].sort_values("Time")
        positions = df_track[["Position X", "Position Y", "Position Z"]].to_numpy(float)
        # compute displacements between consecutive timepoints:
        displacements = np.diff(positions, axis=0)
        # compute summary statistics instead of using raw displacements
        feature_vector = np.concatenate([
            np.mean(displacements, axis=0),
            np.std(displacements, axis=0),
            np.sum(displacements, axis=0)
        ])
        track_features.append(feature_vector)
        track_ids.append(track_id)
    track_features = np.array(track_features)
    track_ids = np.array(track_ids)
    # apply DBSCAN clustering to the track features:
    ## %%
    #clustering = DBSCAN(eps=24.101, min_samples=3).fit(track_features)
    #clustering = AgglomerativeClustering(n_clusters=7).fit(track_features)
    n_clusters = 2
    clustering = KMeans(n_clusters=n_clusters, random_state=42).fit(track_features)
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  Found {n_clusters} clusters of tracks (DBSCAN).")
    
    # plot the cell time-track plot, but now colored by cluster label:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for label in set(labels):
        mask = labels == label
        for track_id in track_ids[mask]:
            df_track = df[df["TrackID"] == track_id].sort_values("Time")
            ax.plot(df_track["Position X"], df_track["Position Y"], df_track["Position Z"],
                    label=f"Cluster {label}" if label != -1 else "Noise", alpha=0.7 if label != -1 else 0.3,
                    color=plt.cm.tab10(label % 10) if label != -1 else "gray")
    ax.set_xlabel('X Position (µm)')
    ax.set_ylabel('Y Position (µm)')
    ax.set_zlabel('Z Position (µm)')
    ax.set_title(f'All Tracks Over Time Colored by Cluster (KMeans, n_clusters={n_clusters})\nFile: {xls_file}')
    plt.tight_layout()
    ## %%
    plt.savefig(os.path.join(RESULTS_path, f'All_Tracks_3D_Clustered_File_{xls_file}.png'), dpi=300)
    plt.close(fig)
    




# %% PROCESS DATA: 3D drift removal
# Two independent correction methods are implemented and kept separate:
# 1) RBF drift field fitted to a reference frame (per timepoint)
# 2) Rigid or z rotation plus translation (per step) via trimmed Procrustes

# ------------------------------------------------------------
# RBF drift field model
# ------------------------------------------------------------

@dataclass
class RBFField:
    centers: np.ndarray   # (M,3)
    sigma: float          # scalar
    W: np.ndarray         # (M,3)
    lam: float            # scalar


def _rbf_kernel(X: np.ndarray, C: np.ndarray, sigma: float) -> np.ndarray:
    # X: (N,3), C: (M,3) -> K: (N,M)
    diff = X[:, None, :] - C[None, :, :]
    r2 = np.sum(diff * diff, axis=2)
    return np.exp(-r2 / (2.0 * sigma * sigma))
def _choose_centers(X: np.ndarray, M: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if X.shape[0] <= M:
        return X.copy()
    idx = rng.choice(X.shape[0], size=M, replace=False)
    return X[idx].copy()
def _default_sigma(C: np.ndarray) -> float:
    M = C.shape[0]
    if M < 2:
        return 1.0
    diff = C[:, None, :] - C[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    nn = np.sqrt(np.min(d2, axis=1))
    s = float(np.median(nn))
    if not np.isfinite(s) or s <= 0:
        finite = d2[np.isfinite(d2)]
        s = float(np.sqrt(np.median(finite))) if finite.size else 1.0
    if not np.isfinite(s) or s <= 0:
        s = 1.0
    return s
def _ridge_solve(K: np.ndarray, D: np.ndarray, lam: float) -> np.ndarray:
    # (K^T K + lam I) W = K^T D
    M = K.shape[1]
    A = K.T @ K
    A.flat[:: M + 1] += lam
    B = K.T @ D
    return np.linalg.solve(A, B)
def robust_fit_rbf_field(
    X: np.ndarray,
    D: np.ndarray,
    M: int = 25,
    keep_fraction: float = 0.7,
    lam: float = 1e-2,
    sigma: float | None = None,
    max_iters: int = 25,
    seed: int = 0,
):
    """
    Fit D ≈ K(X,C) W with trimmed least squares iterations.
    X: (N,3) positions
    D: (N,3) drift vectors to be modeled (to subtract later)
    Returns: field, inlier_mask, residuals
    """
    N = int(X.shape[0])
    if N < 8:
        raise ValueError("Too few correspondences for RBF field fit.")

    keep_fraction = float(np.clip(keep_fraction, 0.1, 0.99))
    k = max(8, int(np.floor(keep_fraction * N)))

    C = _choose_centers(X, M=M, seed=seed)
    if sigma is None:
        sigma = _default_sigma(C)
    sigma = float(sigma)

    K = _rbf_kernel(X, C, sigma)
    W = _ridge_solve(K, D, lam=lam)

    prev = np.inf
    inlier_idx = np.arange(N)

    for _ in range(int(max_iters)):
        Dp = K @ W
        res = np.linalg.norm(Dp - D, axis=1)
        order = np.argsort(res)
        inlier_idx = order[:k]

        Ki = K[inlier_idx]
        Di = D[inlier_idx]
        Wi = _ridge_solve(Ki, Di, lam=lam)

        Dp_i = Ki @ Wi
        cost = float(np.mean(np.sum((Dp_i - Di) ** 2, axis=1)))
        if abs(prev - cost) < 1e-12:
            W = Wi
            break
        W = Wi
        prev = cost

    Dp = K @ W
    res = np.linalg.norm(Dp - D, axis=1)

    inlier_mask = np.zeros(N, dtype=bool)
    inlier_mask[inlier_idx] = True

    field = RBFField(centers=C, sigma=float(sigma), W=W, lam=float(lam))
    return field, inlier_mask, res
def predict_rbf_field(field: RBFField, X: np.ndarray) -> np.ndarray:
    K = _rbf_kernel(X, field.centers, field.sigma)
    return K @ field.W
def estimate_rbf_field_to_reference(
    df: pd.DataFrame,
    t: float,
    tref: float,
    M: int = 25,
    keep_fraction: float = 0.7,
    lam: float = 1e-2,
    sigma: float | None = None,
    min_pairs: int = 15,
    seed: int = 0,
):
    """
    Fit drift field D_t so that D_t(X_t) ≈ X_t - X_ref for matched TrackIDs.
    Corrected positions: X_corr = X_t - D_t_pred(X_t).
    """
    dft = df[df["Time"] == t][["TrackID", "Position X", "Position Y", "Position Z"]].copy()
    dfr = df[df["Time"] == tref][["TrackID", "Position X", "Position Y", "Position Z"]].copy()

    m = dft.merge(dfr, on="TrackID", suffixes=("_t", "_ref"))
    n = int(m.shape[0])
    if n < min_pairs:
        info = {"ok": False, "t": float(t), "tref": float(tref), "n_pairs": n, "reason": "Too few pairs"}
        return None, info

    Xt = m[["Position X_t", "Position Y_t", "Position Z_t"]].to_numpy(float)
    Xr = m[["Position X_ref", "Position Y_ref", "Position Z_ref"]].to_numpy(float)
    D = Xt - Xr  # drift to subtract

    field, inlier_mask, res = robust_fit_rbf_field(
        Xt, D, M=M, keep_fraction=keep_fraction, lam=lam, sigma=sigma, seed=seed
    )

    info = {
        "ok": True,
        "t": float(t),
        "tref": float(tref),
        "n_pairs": n,
        "inlier_fraction": float(np.mean(inlier_mask)),
        "res_median": float(np.median(res)),
        "res_p90": float(np.percentile(res, 90)),
        "sigma": float(field.sigma),
        "lam": float(field.lam),
        "M": int(field.centers.shape[0]),
    }
    return field, info
def correct_positions_rbf_to_reference(
    df: pd.DataFrame,
    reference_time: float | None = None,
    M: int = 25,
    keep_fraction: float = 0.7,
    lam: float = 1e-2,
    sigma: float | None = None,
    min_pairs: int = 15,
    seed: int = 0,
):
    """
    Fit one RBF drift field per timepoint t (against reference_time).
    Returns:
      df_corr: df with Position *_corr columns
      tf_table: diagnostics per timepoint
      fields_by_time: dict time -> RBFField (None for reference_time)
      times: sorted times array
      reference_time: float
    """
    times = np.array(sorted(df["Time"].unique()), dtype=float)
    if reference_time is None:
        reference_time = float(times.min())
    reference_time = float(reference_time)

    df_corr = df.copy()
    df_corr["Position X_corr"] = np.nan
    df_corr["Position Y_corr"] = np.nan
    df_corr["Position Z_corr"] = np.nan

    mref = df_corr["Time"] == reference_time
    df_corr.loc[mref, ["Position X_corr", "Position Y_corr", "Position Z_corr"]] = \
        df_corr.loc[mref, ["Position X", "Position Y", "Position Z"]].to_numpy(float)

    tf_rows = []
    fields_by_time: Dict[float, Optional[RBFField]] = {reference_time: None}

    for i, t in enumerate(times):
        t = float(t)
        if t == reference_time:
            tf_rows.append({
                "ok": True, "t": t, "tref": reference_time, "n_pairs": np.nan,
                "inlier_fraction": np.nan, "res_median": 0.0, "res_p90": 0.0,
                "sigma": np.nan, "lam": float(lam), "M": int(M)
            })
            continue

        field, info = estimate_rbf_field_to_reference(
            df_corr, t=t, tref=reference_time,
            M=M, keep_fraction=keep_fraction, lam=lam, sigma=sigma,
            min_pairs=min_pairs, seed=seed + i
        )
        tf_rows.append(info)
        if not info["ok"]:
            continue

        fields_by_time[t] = field

        mask_t = df_corr["Time"] == t
        X = df_corr.loc[mask_t, ["Position X", "Position Y", "Position Z"]].to_numpy(float)
        Dhat = predict_rbf_field(field, X)
        Xc = X - Dhat

        df_corr.loc[mask_t, ["Position X_corr", "Position Y_corr", "Position Z_corr"]] = Xc

    tf_table = pd.DataFrame(tf_rows)
    return df_corr, tf_table, fields_by_time, times, reference_time
def make_rbf_diagnostic_plots(tf_table: pd.DataFrame, out_prefix: str):
    if tf_table is None or tf_table.shape[0] == 0:
        return

    ok = tf_table["ok"] == True if "ok" in tf_table.columns else np.ones(len(tf_table), dtype=bool)
    if not np.any(ok):
        return

    # Normalize x axis name
    if "t1" not in tf_table.columns:
        if "t" in tf_table.columns:
            tf_table = tf_table.copy()
            tf_table["t1"] = tf_table["t"]

    # inlier fraction
    if "inlier_fraction" in tf_table.columns:
        fig = plt.figure()
        plt.plot(tf_table.loc[ok, "t1"], tf_table.loc[ok, "inlier_fraction"], marker="o")
        plt.xlabel("Time (t)")
        plt.ylabel("Inlier fraction")
        plt.title("RBF drift fit: inlier fraction per timepoint")
        plt.tight_layout()
        plt.savefig(out_prefix + "_rbf_inlier_fraction.png", dpi=300)
        plt.close(fig)

    # residual stats
    fig = plt.figure()
    if "res_median" in tf_table.columns:
        plt.plot(tf_table.loc[ok, "t1"], tf_table.loc[ok, "res_median"], marker="o", label="median")
    if "res_p90" in tf_table.columns:
        plt.plot(tf_table.loc[ok, "t1"], tf_table.loc[ok, "res_p90"], marker="o", label="p90")
    plt.xlabel("Time (t)")
    plt.ylabel("Residual norm (µm)")
    plt.title("RBF drift fit: residual statistics per timepoint")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix + "_rbf_residual_stats.png", dpi=300)
    plt.close(fig)

    if "sigma" in tf_table.columns:
        fig = plt.figure()
        plt.plot(tf_table.loc[ok, "t1"], tf_table.loc[ok, "sigma"], marker="o")
        plt.xlabel("Time (t)")
        plt.ylabel("sigma (µm)")
        plt.title("RBF drift fit: sigma per timepoint")
        plt.tight_layout()
        plt.savefig(out_prefix + "_rbf_sigma.png", dpi=300)
        plt.close(fig)

    if "lam" in tf_table.columns:
        fig = plt.figure()
        plt.plot(tf_table.loc[ok, "t1"], tf_table.loc[ok, "lam"], marker="o")
        plt.xlabel("Time (t)")
        plt.ylabel("lambda (ridge)")
        plt.title("RBF drift fit: ridge lambda per timepoint")
        plt.tight_layout()
        plt.savefig(out_prefix + "_rbf_lambda.png", dpi=300)
        plt.close(fig)

    if "M" in tf_table.columns:
        fig = plt.figure()
        plt.plot(tf_table.loc[ok, "t1"], tf_table.loc[ok, "M"], marker="o")
        plt.xlabel("Time (t)")
        plt.ylabel("Number of centers (M)")
        plt.title("RBF drift fit: number of centers per timepoint")
        plt.tight_layout()
        plt.savefig(out_prefix + "_rbf_M.png", dpi=300)
        plt.close(fig)


# ------------------------------------------------------------
# Rigid per step model
# ------------------------------------------------------------

def kabsch_rigid(X, Y):
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    H = Xc.T @ Yc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    b = Y.mean(axis=0) - R @ X.mean(axis=0)
    return R, b
def rotz_only_fit(X, Y):
    x = X[:, :2]
    y = Y[:, :2]
    xc = x - x.mean(axis=0)
    yc = y - y.mean(axis=0)
    H = xc.T @ yc
    U, S, Vt = np.linalg.svd(H)
    R2 = Vt.T @ U.T
    if np.linalg.det(R2) < 0:
        Vt[-1, :] *= -1
        R2 = Vt.T @ U.T
    theta = np.arctan2(R2[1, 0], R2[0, 0])
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=float)
    b = Y.mean(axis=0) - R @ X.mean(axis=0)
    return R, b, float(theta)
def robust_rigid_trimmed(X, Y, keep_fraction=0.7, max_iters=30, mode="rigid"):
    N = int(X.shape[0])
    k = max(6, int(np.floor(float(keep_fraction) * N)))
    idx = np.arange(N)

    theta = None
    for _ in range(int(max_iters)):
        if mode == "rotz":
            R, b, theta = rotz_only_fit(X[idx], Y[idx])
        else:
            R, b = kabsch_rigid(X[idx], Y[idx])

        Yp = (X @ R.T) + b
        res = np.linalg.norm(Yp - Y, axis=1)
        new_idx = np.argsort(res)[:k]
        if np.array_equal(np.sort(new_idx), np.sort(idx)):
            break
        idx = new_idx

    inlier_mask = np.zeros(N, dtype=bool)
    inlier_mask[idx] = True
    Yp = (X @ R.T) + b
    res = np.linalg.norm(Yp - Y, axis=1)
    return R, b, inlier_mask, res, theta
def estimate_step_rigid_from_tracks(df, t0, t1, keep_fraction=0.7, min_pairs=15, mode="rigid"):
    df0 = df[df["Time"] == t0][["TrackID", "Position X", "Position Y", "Position Z"]].copy()
    df1 = df[df["Time"] == t1][["TrackID", "Position X", "Position Y", "Position Z"]].copy()
    m = df0.merge(df1, on="TrackID", suffixes=("_t0", "_t1"))
    n = int(m.shape[0])
    if n < min_pairs:
        return None, None, {"ok": False, "t0": float(t0), "t1": float(t1), "n_pairs": n}

    X = m[["Position X_t0", "Position Y_t0", "Position Z_t0"]].to_numpy(float)
    Y = m[["Position X_t1", "Position Y_t1", "Position Z_t1"]].to_numpy(float)

    R, b, inlier_mask, res, theta = robust_rigid_trimmed(X, Y, keep_fraction=keep_fraction, mode=mode)

    info = {
        "ok": True,
        "t0": float(t0),
        "t1": float(t1),
        "n_pairs": n,
        "keep_fraction": float(keep_fraction),
        "res_median": float(np.median(res)),
        "res_p90": float(np.percentile(res, 90)),
        "rot_deg": float(np.degrees(theta)) if theta is not None else np.nan,
        "b_norm": float(np.linalg.norm(b)),
    }
    return R, b, info
def compose_rb(R2, b2, R1, b1):
    R = R2 @ R1
    b = R2 @ b1 + b2
    return R, b
def invert_rb(R, b):
    Ri = R.T
    bi = -Ri @ b
    return Ri, bi
def build_step_transforms(df, keep_fraction=0.7, min_pairs=15, mode="rigid"):
    times = np.array(sorted(df["Time"].unique()), dtype=float)
    step = {}
    rows = []
    for i in range(len(times) - 1):
        t0 = float(times[i])
        t1 = float(times[i + 1])
        R, b, info = estimate_step_rigid_from_tracks(df, t0, t1, keep_fraction, min_pairs, mode)
        rows.append(info)
        if info["ok"]:
            step[(t0, t1)] = (R, b)
    return step, pd.DataFrame(rows), times
def cumulative_transform_to_ref(step, times, ref_time):
    times = np.array(sorted(times), dtype=float)
    ref_time = float(ref_time)
    ref_idx = int(np.where(times == ref_time)[0][0])

    fwd = {ref_time: (np.eye(3), np.zeros(3))}
    Rcum = np.eye(3)
    bcum = np.zeros(3)

    for i in range(ref_idx, len(times) - 1):
        t0 = float(times[i])
        t1 = float(times[i + 1])
        if (t0, t1) in step:
            Rstep, bstep = step[(t0, t1)]
            Rcum, bcum = compose_rb(Rstep, bstep, Rcum, bcum)
        fwd[t1] = (Rcum.copy(), bcum.copy())

    inv = {}
    for t, (Rft, bft) in fwd.items():
        inv[t] = invert_rb(Rft, bft)
    return inv
def apply_rigid_correction_to_tracks(df: pd.DataFrame, step: dict, times: np.ndarray, ref_time: float):
    df_corr = df.copy()
    inv = cumulative_transform_to_ref(step, times, ref_time)

    X = df_corr[["Position X", "Position Y", "Position Z"]].to_numpy(float)
    T = df_corr["Time"].to_numpy(float)

    Xcorr = np.empty_like(X)
    for i in range(X.shape[0]):
        R, b = inv.get(float(T[i]), (np.eye(3), np.zeros(3)))
        Xcorr[i] = R @ X[i] + b

    df_corr["Position X_corr"] = Xcorr[:, 0]
    df_corr["Position Y_corr"] = Xcorr[:, 1]
    df_corr["Position Z_corr"] = Xcorr[:, 2]
    return df_corr
def sitk_euler_from_Rb(R, b):
    if sitk is None:
        raise RuntimeError("SimpleITK not available. Install SimpleITK or skip image warping.")
    tx = sitk.Euler3DTransform()
    tx.SetMatrix(R.ravel().tolist())
    tx.SetTranslation(tuple(b.tolist()))
    return tx
def warp_stack_rigid_tzcyx(img_tzcyx, step, times, ref_time, spacing_xyz_um, interp=None):
    if sitk is None:
        raise RuntimeError("SimpleITK not available. Install SimpleITK or skip image warping.")
    if interp is None:
        interp = sitk.sitkLinear

    T, Z, C, Y, X = img_tzcyx.shape
    sx, sy, sz = spacing_xyz_um

    times = np.array(sorted(times), dtype=float)
    ref_time = float(ref_time)
    ref_idx = int(np.where(times == ref_time)[0][0])

    fwd = {ref_time: (np.eye(3), np.zeros(3))}
    Rcum = np.eye(3)
    bcum = np.zeros(3)
    for i in range(ref_idx, len(times) - 1):
        t0 = float(times[i]); t1 = float(times[i + 1])
        if (t0, t1) in step:
            Rstep, bstep = step[(t0, t1)]
            Rcum, bcum = compose_rb(Rstep, bstep, Rcum, bcum)
        fwd[t1] = (Rcum.copy(), bcum.copy())

    out = np.empty_like(img_tzcyx)
    out[ref_idx] = img_tzcyx[ref_idx]

    for ti, tval in enumerate(times):
        if ti == ref_idx:
            continue
        Rrt, brt = fwd[float(tval)]  # ref -> t
        tx = sitk_euler_from_Rb(Rrt, brt)

        for ci in range(C):
            vol = img_tzcyx[ti, :, ci, :, :]
            img = sitk.GetImageFromArray(vol)
            img.SetOrigin((0.0, 0.0, 0.0))
            img.SetDirection(np.eye(3).ravel().tolist())
            img.SetSpacing((sx, sy, sz))

            ref_vol = img_tzcyx[ref_idx, :, ci, :, :]
            ref_img = sitk.GetImageFromArray(ref_vol)
            ref_img.SetOrigin((0.0, 0.0, 0.0))
            ref_img.SetDirection(np.eye(3).ravel().tolist())
            ref_img.SetSpacing((sx, sy, sz))

            warped = sitk.Resample(img, ref_img, tx, interp, 0.0, img.GetPixelID())
            out[ti, :, ci, :, :] = sitk.GetArrayFromImage(warped).astype(img_tzcyx.dtype, copy=False)

    return out


# ------------------------------------------------------------
# Runner: fit corrections per file and store results
# Requires that these variables already exist in your notebook:
#   data_frames: dict[str, pd.DataFrame]
#   xls_files: list[str]
#   RESULTS_path: str
# ------------------------------------------------------------

# Parameters
keep_fraction = 0.7
min_pairs = 15
M = 25
lam = 10e-2 # controls smoothness of the RBF field, i.e., larger lam -> smoother field
sigma = None

df_corrs = {}
tf_tables = {}
rbf_fields = {}

for xls_file, df in data_frames.items():
    print(f"\nProcessing file (RBF to reference): {xls_file}")

    df = df.copy()
    for col in ["Position X", "Position Y", "Position Z", "Time", "TrackID"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Position X", "Position Y", "Position Z", "Time", "TrackID"]).copy()

    df_corr, tf_table, fields_by_time, times, tref = correct_positions_rbf_to_reference(
        df,
        reference_time=float(df["Time"].min()),
        M=M,
        keep_fraction=keep_fraction,
        lam=lam,
        sigma=sigma,
        min_pairs=min_pairs,
        seed=0
    )

    df_corrs[xls_file] = df_corr
    tf_tables[xls_file] = tf_table
    rbf_fields[xls_file] = fields_by_time

    base = os.path.splitext(xls_file)[0]
    out_csv = os.path.join(RESULTS_path, f"{base}_positions_corrected_rbf_to_ref.csv")
    out_tf  = os.path.join(RESULTS_path, f"{base}_rbf_to_ref_diagnostics.csv")
    df_corr.to_csv(out_csv, index=False)
    tf_table.to_csv(out_tf, index=False)

    make_rbf_diagnostic_plots(tf_table, os.path.join(RESULTS_path, base + "_rbf_to_ref"))

    print(f"  Saved: {out_csv}")
    print(f"  Saved: {out_tf}")


# ------------------------------------------------------------
# Comparison plot for first file in xls_files
# ------------------------------------------------------------

for xls_file in xls_files:
    # xls_file = xls_files[0]
    print(f"\nPlotting all tracks over time for: {xls_file}")

    df_orig = data_frames[xls_file]
    df_corr = df_corrs[xls_file]

    fig = plt.figure(figsize=(16, 18))

    # Compute limits after df_corr exists
    x_min = min(df_orig["Position X"].min(), df_corr["Position X_corr"].min())
    x_max = max(df_orig["Position X"].max(), df_corr["Position X_corr"].max())
    y_min = min(df_orig["Position Y"].min(), df_corr["Position Y_corr"].min())
    y_max = max(df_orig["Position Y"].max(), df_corr["Position Y_corr"].max())
    z_min = min(df_orig["Position Z"].min(), df_corr["Position Z_corr"].min())
    z_max = max(df_orig["Position Z"].max(), df_corr["Position Z_corr"].max())

    # Original 3D
    ax1 = fig.add_subplot(321, projection="3d")
    for track_id in df_orig["TrackID"].unique():
        df_track = df_orig[df_orig["TrackID"] == track_id]
        ax1.plot(df_track["Position X"], df_track["Position Y"], df_track["Position Z"])
    ax1.set_xlabel("X Position (µm)")
    ax1.set_ylabel("Y Position (µm)")
    ax1.set_zlabel("Z Position (µm)")
    ax1.set_title("Original - All Tracks (3D)")
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_zlim(z_min, z_max)

    # Corrected 3D
    ax2 = fig.add_subplot(322, projection="3d")
    for track_id in df_corr["TrackID"].unique():
        df_track = df_corr[df_corr["TrackID"] == track_id]
        ax2.plot(df_track["Position X_corr"], df_track["Position Y_corr"], df_track["Position Z_corr"])
    ax2.set_xlabel("X Position (µm)")
    ax2.set_ylabel("Y Position (µm)")
    ax2.set_zlabel("Z Position (µm)")
    ax2.set_title("Corrected - All Tracks (3D)")
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_zlim(z_min, z_max)

    # Original XY
    ax3 = fig.add_subplot(323)
    for track_id in df_orig["TrackID"].unique():
        df_track = df_orig[df_orig["TrackID"] == track_id]
        ax3.plot(df_track["Position X"], df_track["Position Y"], alpha=0.7)
    ax3.set_xlabel("X Position (µm)")
    ax3.set_ylabel("Y Position (µm)")
    ax3.set_title("Original - XY Projection")
    ax3.set_aspect("equal", adjustable="box")
    ax3.set_xlim(x_min, x_max)
    ax3.set_ylim(y_min, y_max)

    # Corrected XY
    ax4 = fig.add_subplot(324)
    for track_id in df_corr["TrackID"].unique():
        df_track = df_corr[df_corr["TrackID"] == track_id]
        ax4.plot(df_track["Position X_corr"], df_track["Position Y_corr"], alpha=0.7)
    ax4.set_xlabel("X Position (µm)")
    ax4.set_ylabel("Y Position (µm)")
    ax4.set_title("Corrected - XY Projection")
    ax4.set_aspect("equal", adjustable="box")
    ax4.set_xlim(x_min, x_max)
    ax4.set_ylim(y_min, y_max)

    # Original XZ
    ax5 = fig.add_subplot(325)
    for track_id in df_orig["TrackID"].unique():
        df_track = df_orig[df_orig["TrackID"] == track_id]
        ax5.plot(df_track["Position X"], df_track["Position Z"], alpha=0.7)
    ax5.set_xlabel("X Position (µm)")
    ax5.set_ylabel("Z Position (µm)")
    ax5.set_title("Original - XZ Projection")
    ax5.set_aspect("equal", adjustable="box")
    ax5.set_xlim(x_min, x_max)
    ax5.set_ylim(z_min, z_max)

    # Corrected XZ
    ax6 = fig.add_subplot(326)
    for track_id in df_corr["TrackID"].unique():
        df_track = df_corr[df_corr["TrackID"] == track_id]
        ax6.plot(df_track["Position X_corr"], df_track["Position Z_corr"], alpha=0.7)
    ax6.set_xlabel("X Position (µm)")
    ax6.set_ylabel("Z Position (µm)")
    ax6.set_title("Corrected - XZ Projection")
    ax6.set_aspect("equal", adjustable="box")
    ax6.set_xlim(x_min, x_max)
    ax6.set_ylim(z_min, z_max)

    fig.suptitle(f"3D Tracks Comparison for File: {xls_file}\nkeep_fraction={keep_fraction}, min_pairs={min_pairs}, lam={lam}, M={M}", fontsize=16)
    plt.tight_layout()

    out_plot = os.path.join(RESULTS_path, f"3D_tracks_comparison_File_{xls_file}_kf{keep_fraction}_minpairs{min_pairs}_lam{lam}_M{M}.png")
    plt.savefig(out_plot, dpi=300)
    plt.close(fig)

    print(f"Saved plot: {out_plot}")


# ------------------------------------------------------------
# Optional: build rigid transforms for the same df and save
# This writes a rigid transform table, but does not overwrite your RBF outputs.
# ------------------------------------------------------------

df = df_orig.copy()
for col in ["Position X", "Position Y", "Position Z", "Time", "TrackID"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["Position X", "Position Y", "Position Z", "Time", "TrackID"]).copy()

mode = "rotz"  # or "rigid"
step, tf_rigid, times_rigid = build_step_transforms(df, keep_fraction=keep_fraction, min_pairs=min_pairs, mode=mode)

out_rigid_tf = os.path.join(RESULTS_path, "BEST_rigid_transforms.csv")
tf_rigid.to_csv(out_rigid_tf, index=False)
print(f"Saved rigid transforms: {out_rigid_tf}")


# %% PROCESS DATA: drift removal via correlation graph communities + local drift field
# This implements:
# 1) Build displacement time series per TrackID relative to a reference time
# 2) Build a spatial neighbor graph (kNN or radius)
# 3) Weight edges by displacement similarity (correlation) times spatial proximity
# 4) Community detection on the graph to get local drift groups
# 5) Robust cluster drift traces (component-wise median over tracks)
# 6) Soft spatial interpolation of cluster drift traces to a drift field over space
# 7) Correct every spot by subtracting predicted drift at its location and time



# ------------------------------------------------------------
# Helpers: displacement series extraction
# ------------------------------------------------------------

def _ensure_numeric_tracks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Position X", "Position Y", "Position Z", "Time", "TrackID"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Position X", "Position Y", "Position Z", "Time", "TrackID"]).copy()
    return df
def _pivot_tracks(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
      times: (T,)
      track_ids: (N,)
      X: (N,T,3) positions in microns, NaN for missing
    """
    times = np.array(sorted(df["Time"].unique()), dtype=float)
    track_ids = np.array(sorted(df["TrackID"].unique()), dtype=float)

    tid_to_idx = {float(tid): i for i, tid in enumerate(track_ids)}
    t_to_idx = {float(t): j for j, t in enumerate(times)}

    X = np.full((len(track_ids), len(times), 3), np.nan, dtype=float)

    # use DataFrame access, not tuple attributes
    for _, row in df.iterrows():
        i = tid_to_idx[float(row["TrackID"])]
        j = t_to_idx[float(row["Time"])]

        X[i, j, 0] = float(row["Position X"])
        X[i, j, 1] = float(row["Position Y"])
        X[i, j, 2] = float(row["Position Z"])

    return times, track_ids, X
def _track_positions_at_time(X: np.ndarray, times: np.ndarray, t: float) -> np.ndarray:
    j = int(np.where(times == float(t))[0][0])
    return X[:, j, :]
def _compute_displacements_to_ref(times: np.ndarray,
                                  track_ids: np.ndarray,
                                  X: np.ndarray,
                                  reference_time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Keep only tracks that exist at reference_time and at least 2 timepoints overall.
    Return:
      keep_track_ids: (Nkeep,)
      P0: (Nkeep,3) mean position (for spatial localization)
      D: (Nkeep,T,3) displacement relative to reference_time
    """
    ref_time = float(reference_time)
    ref_idx = int(np.where(times == ref_time)[0][0])

    Xref = X[:, ref_idx, :]  # (N,3), can contain NaN
    has_ref = np.all(np.isfinite(Xref), axis=1)

    # must have at least 2 finite timepoints to correlate
    finite_counts = np.sum(np.all(np.isfinite(X), axis=2), axis=1)
    ok = has_ref & (finite_counts >= 2)

    keep_ids = track_ids[ok]
    Xk = X[ok]  # (Nkeep,T,3)
    Xrefk = Xk[:, ref_idx:ref_idx + 1, :]  # (Nkeep,1,3)

    D = Xk - Xrefk  # displacement relative ref, NaN where Xk is NaN

    # spatial anchor for each track: mean position over timepoints where finite
    P0 = np.nanmean(Xk, axis=1)

    return keep_ids, P0, D

# ------------------------------------------------------------
# Similarity: correlation of displacement time series
# ------------------------------------------------------------

def _corr_1d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Pearson correlation on overlapping finite entries.
    Returns 0 if insufficient overlap or constant signals.
    """
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) < 3:
        return 0.0
    aa = a[m]
    bb = b[m]
    if np.std(aa) <= 1e-12 or np.std(bb) <= 1e-12:
        return 0.0
    r = float(pearsonr(aa, bb)[0])
    if not np.isfinite(r):
        return 0.0
    return r
def displacement_similarity(Di: np.ndarray, Dj: np.ndarray) -> float:
    """
    Di, Dj: (T,3) displacement time series relative to reference_time.
    Similarity is sum of correlations in x, y, z clamped to [0, 3].
    """
    rx = _corr_1d(Di[:, 0], Dj[:, 0])
    ry = _corr_1d(Di[:, 1], Dj[:, 1])
    rz = _corr_1d(Di[:, 2], Dj[:, 2])
    s = max(0.0, rx) + max(0.0, ry) + max(0.0, rz)
    return float(s)
def spatial_weight(pi: np.ndarray, pj: np.ndarray, ell_um: float) -> float:
    d2 = float(np.sum((pi - pj) ** 2))
    if ell_um <= 0:
        return 1.0
    return float(np.exp(-d2 / (2.0 * ell_um * ell_um)))

# ------------------------------------------------------------
# Graph construction and community detection
# ------------------------------------------------------------

@dataclass
class CommunityResult:
    communities: List[List[int]]          # list of lists of node indices
    graph: nx.Graph
    node_index_to_track_id: Dict[int, float]
    track_id_to_node_index: Dict[float, int]

def build_similarity_graph(P0: np.ndarray,
                           D: np.ndarray,
                           track_ids: np.ndarray,
                           k_neighbors: int = 12,
                           radius_um: Optional[float] = None,
                           ell_um: float = 80.0,
                           min_edge_weight: float = 0.25,
                           max_pairs_per_node: Optional[int] = None) -> CommunityResult:
    """
    Nodes are tracks.
    Edges connect spatial neighbors (kNN or radius) weighted by similarity and spatial kernel.
    """
    N = P0.shape[0]
    G = nx.Graph()

    node_index_to_track_id = {i: float(track_ids[i]) for i in range(N)}
    track_id_to_node_index = {float(track_ids[i]): i for i in range(N)}

    for i in range(N):
        G.add_node(i)

    tree = cKDTree(P0)

    if radius_um is not None:
        neighbor_lists = tree.query_ball_point(P0, r=float(radius_um))
        for i in range(N):
            neigh = neighbor_lists[i]
            if i in neigh:
                neigh.remove(i)
            if max_pairs_per_node is not None and len(neigh) > max_pairs_per_node:
                neigh = neigh[:max_pairs_per_node]
            for j in neigh:
                if j <= i:
                    continue
                sim = displacement_similarity(D[i], D[j])
                if sim <= 0:
                    continue
                w = spatial_weight(P0[i], P0[j], ell_um=ell_um) * sim
                if w >= min_edge_weight:
                    G.add_edge(i, j, weight=float(w))
    else:
        k = int(max(1, k_neighbors))
        dists, idxs = tree.query(P0, k=min(k + 1, N))
        for i in range(N):
            neigh = idxs[i].tolist()
            if i in neigh:
                neigh.remove(i)
            if max_pairs_per_node is not None and len(neigh) > max_pairs_per_node:
                neigh = neigh[:max_pairs_per_node]
            for j in neigh:
                if j <= i:
                    continue
                sim = displacement_similarity(D[i], D[j])
                if sim <= 0:
                    continue
                w = spatial_weight(P0[i], P0[j], ell_um=ell_um) * sim
                if w >= min_edge_weight:
                    G.add_edge(i, j, weight=float(w))

    # Community detection with fallback
    communities = None

    # networkx louvain if available
    if hasattr(nx.algorithms.community, "louvain_communities"):
        try:
            communities = list(nx.algorithms.community.louvain_communities(G, weight="weight", resolution=1.0, seed=0))
        except Exception:
            communities = None

    if communities is None:
        try:
            communities = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))
        except Exception:
            # worst case: each node its own community
            communities = [set([i]) for i in range(N)]

    # convert to sorted lists
    comm_lists = [sorted(list(c)) for c in communities if len(c) > 0]
    comm_lists = sorted(comm_lists, key=lambda x: (-len(x), x[0]))

    return CommunityResult(
        communities=comm_lists,
        graph=G,
        node_index_to_track_id=node_index_to_track_id,
        track_id_to_node_index=track_id_to_node_index
    )


# ------------------------------------------------------------
# Robust cluster drift traces and soft field interpolation
# ------------------------------------------------------------

@dataclass
class DriftModel:
    reference_time: float
    times: np.ndarray                 # (T,)
    cluster_centers: np.ndarray       # (K,3)
    cluster_drift: np.ndarray         # (K,T,3)
    ell_um: float                     # spatial mixing lengthscale
    cluster_sizes: np.ndarray         # (K,)
    node_to_cluster: Dict[int, int]   # node index -> cluster index

def cluster_drift_traces(P0: np.ndarray,
                         D: np.ndarray,
                         communities: List[List[int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, int]]:
    """
    For each community c:
      center is mean(P0[node])
      drift trace is componentwise median of D[node,:,:] across nodes, ignoring NaN
    Returns:
      centers: (K,3)
      drift: (K,T,3)
      sizes: (K,)
      node_to_cluster: dict node -> k
    """
    K = len(communities)
    T = D.shape[1]
    centers = np.zeros((K, 3), dtype=float)
    drift = np.zeros((K, T, 3), dtype=float)
    sizes = np.zeros((K,), dtype=int)
    node_to_cluster = {}

    for k, nodes in enumerate(communities):
        nodes = list(nodes)
        sizes[k] = len(nodes)
        for n in nodes:
            node_to_cluster[int(n)] = k

        centers[k] = np.mean(P0[nodes], axis=0)

        # median over nodes, per time, per axis
        for ti in range(T):
            for ax in range(3):
                v = D[nodes, ti, ax]
                vv = v[np.isfinite(v)]
                drift[k, ti, ax] = float(np.median(vv)) if vv.size else 0.0

    return centers, drift, sizes, node_to_cluster
def predict_drift_at_positions(P: np.ndarray,
                               cluster_centers: np.ndarray,
                               cluster_drift_t: np.ndarray,
                               ell_um: float) -> np.ndarray:
    """
    P: (N,3)
    cluster_centers: (K,3)
    cluster_drift_t: (K,3) drift vectors at time t
    Returns: (N,3)
    Soft spatial mixing of cluster drifts via Gaussian kernel.
    """
    N = P.shape[0]
    K = cluster_centers.shape[0]
    out = np.zeros((N, 3), dtype=float)

    if K == 0:
        return out

    if ell_um <= 0:
        # uniform average
        out[:] = np.mean(cluster_drift_t, axis=0)
        return out

    # compute weights (N,K)
    diff = P[:, None, :] - cluster_centers[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    W = np.exp(-d2 / (2.0 * ell_um * ell_um))
    Wsum = np.sum(W, axis=1, keepdims=True)
    Wsum[Wsum <= 1e-12] = 1.0
    Wn = W / Wsum

    out = Wn @ cluster_drift_t
    return out
def build_drift_model(times: np.ndarray,
                      reference_time: float,
                      P0: np.ndarray,
                      D: np.ndarray,
                      comm_res: CommunityResult,
                      ell_um: float) -> DriftModel:
    centers, drift, sizes, node_to_cluster = cluster_drift_traces(P0, D, comm_res.communities)
    return DriftModel(
        reference_time=float(reference_time),
        times=np.array(times, dtype=float),
        cluster_centers=centers,
        cluster_drift=drift,
        ell_um=float(ell_um),
        cluster_sizes=sizes,
        node_to_cluster=node_to_cluster
    )


# ------------------------------------------------------------
# Apply correction to full DataFrame
# ------------------------------------------------------------

def apply_drift_correction(df: pd.DataFrame,
                           model: DriftModel,
                           use_track_mean_position: bool = True) -> pd.DataFrame:
    """
    Correct each row:
      x_corr(t) = x(t) - d_hat(p, t)
    where p is a spatial anchor for that track.
    If use_track_mean_position is True, use per TrackID mean position as p.
    Otherwise use per row position as p (more adaptive but can leak real migration into drift assignment).
    """
    dfc = df.copy()

    times = model.times
    t_to_idx = {float(t): i for i, t in enumerate(times)}

    if use_track_mean_position:
        Ptrack = dfc.groupby("TrackID")[["Position X", "Position Y", "Position Z"]].mean().reset_index()
        track_to_p = {float(r.TrackID): np.array([r["Position X"], r["Position Y"], r["Position Z"]], dtype=float)
                      for _, r in Ptrack.iterrows()}
        P = np.vstack([track_to_p.get(float(tid), np.array([np.nan, np.nan, np.nan]))
                       for tid in dfc["TrackID"].to_numpy(float)])
    else:
        P = dfc[["Position X", "Position Y", "Position Z"]].to_numpy(float)

    X = dfc[["Position X", "Position Y", "Position Z"]].to_numpy(float)
    Trow = dfc["Time"].to_numpy(float)

    Xcorr = np.full_like(X, np.nan, dtype=float)

    for ti, tval in enumerate(times):
        mask = Trow == float(tval)
        if not np.any(mask):
            continue

        idx = int(t_to_idx[float(tval)])
        drift_t = model.cluster_drift[:, idx, :]  # (K,3)

        Pm = P[mask]
        Xm = X[mask]

        drift_pred = predict_drift_at_positions(Pm, model.cluster_centers, drift_t, model.ell_um)
        Xcorr[mask] = Xm - drift_pred

    dfc["Position X_corr"] = Xcorr[:, 0]
    dfc["Position Y_corr"] = Xcorr[:, 1]
    dfc["Position Z_corr"] = Xcorr[:, 2]

    return dfc


# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------

def summarize_communities(P0: np.ndarray,
                          D: np.ndarray,
                          comm_res: CommunityResult,
                          times: np.ndarray,
                          reference_time: float,
                          max_print: int = 12) -> pd.DataFrame:
    """
    For each community:
      size, median within community similarity, spatial spread
    """
    rows = []
    for k, nodes in enumerate(comm_res.communities):
        nodes = list(nodes)
        if len(nodes) < 2:
            med_sim = 0.0
        else:
            sims = []
            for a in range(len(nodes)):
                for b in range(a + 1, len(nodes)):
                    i = nodes[a]
                    j = nodes[b]
                    sims.append(displacement_similarity(D[i], D[j]))
            med_sim = float(np.median(sims)) if len(sims) else 0.0

        Pk = P0[nodes]
        center = np.mean(Pk, axis=0)
        spread = float(np.median(np.linalg.norm(Pk - center[None, :], axis=1)))

        rows.append({
            "cluster": k,
            "size": int(len(nodes)),
            "median_similarity_sumcorr": med_sim,
            "median_spatial_spread_um": spread,
            "center_x": float(center[0]),
            "center_y": float(center[1]),
            "center_z": float(center[2]),
        })

    out = pd.DataFrame(rows).sort_values(["size", "median_similarity_sumcorr"], ascending=[False, False]).reset_index(drop=True)
    if out.shape[0] > max_print:
        print(out.head(max_print))
    else:
        print(out)
    return out
def plot_cluster_drifts(model: DriftModel, out_prefix: str):
    if plt is None:
        return
    K = model.cluster_drift.shape[0]
    T = model.times.shape[0]
    t = model.times

    for ax, name in enumerate(["x", "y", "z"]):
        fig = plt.figure()
        for k in range(K):
            plt.plot(t, model.cluster_drift[k, :, ax], alpha=0.8)
        plt.xlabel("Time")
        plt.ylabel(f"drift {name} (µm)")
        plt.title(f"Cluster drift traces, axis {name}")
        plt.tight_layout()
        plt.savefig(out_prefix + f"_cluster_drifts_{name}.png", dpi=300)
        plt.close(fig)


# ------------------------------------------------------------
# Runner per file
# Requires:
#   data_frames: dict[str, pd.DataFrame]
#   xls_files: list[str]
#   RESULTS_path: str
# ------------------------------------------------------------

def run_local_drift_correction_for_file(df: pd.DataFrame,
                                        out_prefix: str,
                                        reference_time: Optional[float] = None,
                                        k_neighbors: int = 12,
                                        radius_um: Optional[float] = None,
                                        ell_um_graph: float = 80.0,
                                        ell_um_field: float = 120.0,
                                        min_edge_weight: float = 0.25,
                                        max_pairs_per_node: Optional[int] = 50,
                                        use_track_mean_position: bool = True):
    """
    ell_um_graph: spatial lengthscale used in edge weights
    ell_um_field: spatial lengthscale used to mix cluster drift traces into a field
    """
    df = _ensure_numeric_tracks(df)
    times, track_ids, X = _pivot_tracks(df)

    if reference_time is None:
        reference_time = float(times.min())
    reference_time = float(reference_time)

    keep_ids, P0, D = _compute_displacements_to_ref(times, track_ids, X, reference_time=reference_time)

    if keep_ids.size < 5:
        raise RuntimeError("Too few usable tracks with reference_time present. Cannot build drift communities.")

    # Build graph and communities
    comm_res = build_similarity_graph(
        P0=P0,
        D=D,
        track_ids=keep_ids,
        k_neighbors=k_neighbors,
        radius_um=radius_um,
        ell_um=ell_um_graph,
        min_edge_weight=min_edge_weight,
        max_pairs_per_node=max_pairs_per_node
    )

    # Summarize communities
    comm_table = summarize_communities(P0, D, comm_res, times, reference_time)

    # Build drift model (cluster traces and spatial mixing)
    model = build_drift_model(
        times=times,
        reference_time=reference_time,
        P0=P0,
        D=D,
        comm_res=comm_res,
        ell_um=ell_um_field
    )

    # Apply correction to full df
    df_corr = apply_drift_correction(df, model, use_track_mean_position=use_track_mean_position)

    # Save outputs
    df_corr.to_csv(out_prefix + "_positions_corrected_localdrift.csv", index=False)
    comm_table.to_csv(out_prefix + "_localdrift_cluster_summary.csv", index=False)

    # Save model drifts
    # Store cluster drift traces as a long table
    rows = []
    for k in range(model.cluster_drift.shape[0]):
        for ti, tval in enumerate(model.times):
            rows.append({
                "cluster": int(k),
                "time": float(tval),
                "drift_x": float(model.cluster_drift[k, ti, 0]),
                "drift_y": float(model.cluster_drift[k, ti, 1]),
                "drift_z": float(model.cluster_drift[k, ti, 2]),
                "center_x": float(model.cluster_centers[k, 0]),
                "center_y": float(model.cluster_centers[k, 1]),
                "center_z": float(model.cluster_centers[k, 2]),
                "cluster_size": int(model.cluster_sizes[k]),
            })
    drift_table = pd.DataFrame(rows)
    drift_table.to_csv(out_prefix + "_localdrift_cluster_drifts.csv", index=False)

    plot_cluster_drifts(model, out_prefix)

    print("Saved:")
    print(out_prefix + "_positions_corrected_localdrift.csv")
    print(out_prefix + "_localdrift_cluster_summary.csv")
    print(out_prefix + "_localdrift_cluster_drifts.csv")

    return df_corr, model, comm_res, comm_table, drift_table


# ------------------------------------------------------------
# Batch run
# ------------------------------------------------------------

# Tune these first, they matter the most:
# * radius_um or k_neighbors controls locality
# * ell_um_graph controls how strongly spatial distance penalizes edges
# * ell_um_field controls how smoothly cluster drifts are mixed into a field
#
# Practical defaults for microglia tracks often start here:
k_neighbors = 10
radius_um = None              # set e.g. 120.0 to enforce local neighborhoods
ell_um_graph = 80.0
ell_um_field = 120.0
min_edge_weight = 0.35        # increase to make communities cleaner, decrease if graph too sparse

use_track_mean_position = True

df_corrs = {}
models = {}
community_summaries = {}

for xls_file, df in data_frames.items():
    print(f"\nLocal drift correction via communities for: {xls_file}")
    base = os.path.splitext(xls_file)[0]
    out_prefix = os.path.join(RESULTS_path, base)

    df_corr, model, comm_res, comm_table, drift_table = run_local_drift_correction_for_file(
        df=df,
        out_prefix=out_prefix,
        reference_time=float(df["Time"].min()),
        k_neighbors=k_neighbors,
        radius_um=radius_um,
        ell_um_graph=ell_um_graph,
        ell_um_field=ell_um_field,
        min_edge_weight=min_edge_weight,
        max_pairs_per_node=60,
        use_track_mean_position=use_track_mean_position)

    df_corrs[xls_file] = df_corr
    models[xls_file] = model
    community_summaries[xls_file] = comm_table


# ------------------------------------------------------------
# Optional: quick comparison plot for first file
# ------------------------------------------------------------

def plot_tracks_comparison(df_orig: pd.DataFrame,
                           df_corr: pd.DataFrame,
                           out_path: str,
                           title: str = ""):
    if plt is None:
        return
    fig = plt.figure(figsize=(16, 18))

    x_min = min(df_orig["Position X"].min(), df_corr["Position X_corr"].min())
    x_max = max(df_orig["Position X"].max(), df_corr["Position X_corr"].max())
    y_min = min(df_orig["Position Y"].min(), df_corr["Position Y_corr"].min())
    y_max = max(df_orig["Position Y"].max(), df_corr["Position Y_corr"].max())
    z_min = min(df_orig["Position Z"].min(), df_corr["Position Z_corr"].min())
    z_max = max(df_orig["Position Z"].max(), df_corr["Position Z_corr"].max())

    ax1 = fig.add_subplot(321, projection="3d")
    for tid in df_orig["TrackID"].unique():
        d = df_orig[df_orig["TrackID"] == tid]
        ax1.plot(d["Position X"], d["Position Y"], d["Position Z"], alpha=0.9)
    ax1.set_title("Original (3D)")
    ax1.set_xlim(x_min, x_max); ax1.set_ylim(y_min, y_max); ax1.set_zlim(z_min, z_max)
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

    ax2 = fig.add_subplot(322, projection="3d")
    for tid in df_corr["TrackID"].unique():
        d = df_corr[df_corr["TrackID"] == tid]
        ax2.plot(d["Position X_corr"], d["Position Y_corr"], d["Position Z_corr"], alpha=0.9)
    ax2.set_title("Corrected (3D)")
    ax2.set_xlim(x_min, x_max); ax2.set_ylim(y_min, y_max); ax2.set_zlim(z_min, z_max)
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")

    ax3 = fig.add_subplot(323)
    for tid in df_orig["TrackID"].unique():
        d = df_orig[df_orig["TrackID"] == tid]
        ax3.plot(d["Position X"], d["Position Y"], alpha=0.7)
    ax3.set_title("Original XY")
    ax3.set_aspect("equal", adjustable="box")
    ax3.set_xlim(x_min, x_max); ax3.set_ylim(y_min, y_max)
    ax3.set_xlabel("X"); ax3.set_ylabel("Y")

    ax4 = fig.add_subplot(324)
    for tid in df_corr["TrackID"].unique():
        d = df_corr[df_corr["TrackID"] == tid]
        ax4.plot(d["Position X_corr"], d["Position Y_corr"], alpha=0.7)
    ax4.set_title("Corrected XY")
    ax4.set_aspect("equal", adjustable="box")
    ax4.set_xlim(x_min, x_max); ax4.set_ylim(y_min, y_max)
    ax4.set_xlabel("X"); ax4.set_ylabel("Y")

    ax5 = fig.add_subplot(325)
    for tid in df_orig["TrackID"].unique():
        d = df_orig[df_orig["TrackID"] == tid]
        ax5.plot(d["Position X"], d["Position Z"], alpha=0.7)
    ax5.set_title("Original XZ")
    ax5.set_aspect("equal", adjustable="box")
    ax5.set_xlim(x_min, x_max); ax5.set_ylim(z_min, z_max)
    ax5.set_xlabel("X"); ax5.set_ylabel("Z")

    ax6 = fig.add_subplot(326)
    for tid in df_corr["TrackID"].unique():
        d = df_corr[df_corr["TrackID"] == tid]
        ax6.plot(d["Position X_corr"], d["Position Z_corr"], alpha=0.7)
    ax6.set_title("Corrected XZ")
    ax6.set_aspect("equal", adjustable="box")
    ax6.set_xlim(x_min, x_max); ax6.set_ylim(z_min, z_max)
    ax6.set_xlabel("X"); ax6.set_ylabel("Z")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

for xls_file0 in xls_files:
    # xls_file0 = xls_files[0]
    df_orig0 = _ensure_numeric_tracks(data_frames[xls_file0])
    df_corr0 = df_corrs[xls_file0]

    out_plot = os.path.join(
        RESULTS_path,
        f"3D_tracks_comparison_localdrift_{os.path.splitext(xls_file0)[0]}_knn{k_neighbors}_ellg{ell_um_graph}_ellf{ell_um_field}_minw{min_edge_weight}.png"
    )
    plot_tracks_comparison(
        df_orig=df_orig0,
        df_corr=df_corr0,
        out_path=out_plot,
        title=f"Local drift correction via communities: {xls_file0}"
    )
    print(f"Saved plot: {out_plot}")


# VALIDATION: drift removal quality checks
# Assumes you already have:
#   df_orig: pd.DataFrame with columns ["TrackID","Time","Position X","Position Y","Position Z"]
#   df_corr: pd.DataFrame with columns ["TrackID","Time","Position X_corr","Position Y_corr","Position Z_corr"]
# and optionally:
#   RESULTS_path, base (for saving figures)


# ------------------------------------------------------------
# Robust pivot utilities
# ------------------------------------------------------------

def _ensure_numeric_tracks(df: pd.DataFrame,
                           pos_cols=("Position X", "Position Y", "Position Z"),
                           time_col="Time",
                           tid_col="TrackID") -> pd.DataFrame:
    df = df.copy()
    for c in [*pos_cols, time_col, tid_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[*pos_cols, time_col, tid_col]).copy()
    return df
def _pivot_tracks(df: pd.DataFrame,
                 xcol="Position X",
                 ycol="Position Y",
                 zcol="Position Z",
                 time_col="Time",
                 tid_col="TrackID") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
      times: (T,)
      track_ids: (N,)
      X: (N,T,3) positions in microns, NaN for missing
    """
    df = df[[tid_col, time_col, xcol, ycol, zcol]].copy()
    df[tid_col] = pd.to_numeric(df[tid_col], errors="coerce")
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[xcol] = pd.to_numeric(df[xcol], errors="coerce")
    df[ycol] = pd.to_numeric(df[ycol], errors="coerce")
    df[zcol] = pd.to_numeric(df[zcol], errors="coerce")
    df = df.dropna()

    times = np.array(sorted(df[time_col].unique()), dtype=float)
    track_ids = np.array(sorted(df[tid_col].unique()), dtype=float)

    tid_to_idx = {float(tid): i for i, tid in enumerate(track_ids)}
    t_to_idx = {float(t): j for j, t in enumerate(times)}

    X = np.full((len(track_ids), len(times), 3), np.nan, dtype=float)

    # Fast and safe: use itertuples with renamed columns to avoid "Position X" attribute issues
    dfr = df.rename(columns={
        tid_col: "TrackID_",
        time_col: "Time_",
        xcol: "PX_",
        ycol: "PY_",
        zcol: "PZ_",
    })

    for row in dfr.itertuples(index=False):
        i = tid_to_idx[float(row.TrackID_)]
        j = t_to_idx[float(row.Time_)]
        X[i, j, 0] = float(row.PX_)
        X[i, j, 1] = float(row.PY_)
        X[i, j, 2] = float(row.PZ_)

    return times, track_ids, X
def _displacements_relative_to_t0(X: np.ndarray) -> np.ndarray:
    """
    X: (N,T,3) possibly with NaN
    Return D: (N,T,3) where D[:,t] = X[:,t] - X[:,t0] if both present else NaN
    Uses per track first valid timepoint as anchor.
    """
    N, T, _ = X.shape
    D = np.full_like(X, np.nan, dtype=float)
    for i in range(N):
        xi = X[i]
        valid = np.all(np.isfinite(xi), axis=1)
        if not np.any(valid):
            continue
        t0 = int(np.where(valid)[0][0])
        x0 = xi[t0]
        for t in range(T):
            if np.all(np.isfinite(xi[t])):
                D[i, t] = xi[t] - x0
    return D


# ------------------------------------------------------------
# Metric 1: pairwise trace correlation (common mode drift)
# ------------------------------------------------------------

def _pairwise_trace_correlation(D: np.ndarray,
                                min_overlap: int = 4) -> np.ndarray:
    """
    D: (N,T,3) displacement traces
    Returns corr values for all i<j with sufficient overlap.
    We correlate the scalar time series ||D(t)||, not x/y/z separately.
    """
    N, T, _ = D.shape
    s = np.linalg.norm(D, axis=2)  # (N,T)
    corrs = []
    for i in range(N):
        si = s[i]
        vi = np.isfinite(si)
        for j in range(i + 1, N):
            sj = s[j]
            vj = np.isfinite(sj)
            vv = vi & vj
            if int(np.sum(vv)) < int(min_overlap):
                continue
            a = si[vv]
            b = sj[vv]
            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                continue
            c = float(np.corrcoef(a, b)[0, 1])
            if np.isfinite(c):
                corrs.append(c)
    return np.array(corrs, dtype=float)


# ------------------------------------------------------------
# Metric 2: local neighbor distance invariance
# ------------------------------------------------------------

def _mean_position_per_track(X: np.ndarray) -> np.ndarray:
    """
    Mean position per track, ignoring NaNs.
    """
    return np.nanmean(X, axis=1)  # (N,3)
def _knn_from_points(P: np.ndarray, k: int = 8) -> np.ndarray:
    """
    P: (N,3)
    Returns neighbor indices array nbrs: (N,k) excluding self.
    NaN rows in P are ignored by assigning empty neighbors.
    """
    N = P.shape[0]
    nbrs = np.full((N, k), -1, dtype=int)

    finite = np.all(np.isfinite(P), axis=1)
    idx = np.where(finite)[0]
    if idx.size < 2:
        return nbrs

    Pf = P[idx]
    # brute force distances
    diff = Pf[:, None, :] - Pf[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)

    for a, ii in enumerate(idx):
        order = np.argsort(d2[a])
        take = order[:k]
        nbrs[ii, :len(take)] = idx[take]
    return nbrs
def _neighbor_distance_stats(X: np.ndarray,
                             k_neighbors: int = 8,
                             min_pairs: int = 20) -> Dict[str, float]:
    """
    X: (N,T,3)
    Measures how stable neighbor distances are over time.
    Returns summary stats of relative distance change.
    """
    P = _mean_position_per_track(X)
    nbrs = _knn_from_points(P, k=k_neighbors)

    N, T, _ = X.shape
    rel_changes = []

    for i in range(N):
        for nn in nbrs[i]:
            if nn < 0:
                continue
            # distance time series
            xi = X[i]
            xj = X[nn]
            valid = np.all(np.isfinite(xi), axis=1) & np.all(np.isfinite(xj), axis=1)
            if np.sum(valid) < 4:
                continue
            d = np.linalg.norm(xi[valid] - xj[valid], axis=1)
            if np.nanmedian(d) < 1e-9:
                continue
            # relative deviation around its median
            rel = (d - np.nanmedian(d)) / np.nanmedian(d)
            rel_changes.append(np.nanstd(rel))

    rel_changes = np.array(rel_changes, dtype=float)
    rel_changes = rel_changes[np.isfinite(rel_changes)]
    if rel_changes.size < min_pairs:
        return {
            "ok": False,
            "n_pairs": int(rel_changes.size),
            "rel_std_median": np.nan,
            "rel_std_p90": np.nan,
        }

    return {
        "ok": True,
        "n_pairs": int(rel_changes.size),
        "rel_std_median": float(np.nanmedian(rel_changes)),
        "rel_std_p90": float(np.nanpercentile(rel_changes, 90)),
    }


# ------------------------------------------------------------
# Metric 3: MSD comparison (do not destroy genuine motion)
# ------------------------------------------------------------

def _msd_per_track(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    X: (N,T,3)
    Returns:
      tau: (T-1,)
      msd: (N,T-1) where msd[i,tau] uses all valid pairs separated by tau
    """
    N, T, _ = X.shape
    tau = np.arange(1, T, dtype=int)
    msd = np.full((N, T - 1), np.nan, dtype=float)

    for i in range(N):
        xi = X[i]
        valid = np.all(np.isfinite(xi), axis=1)
        if np.sum(valid) < 3:
            continue
        for dt in tau:
            # pairs t and t+dt
            v = valid[:-dt] & valid[dt:]
            if np.sum(v) < 2:
                continue
            dx = xi[dt:][v] - xi[:-dt][v]
            msd[i, dt - 1] = float(np.mean(np.sum(dx * dx, axis=1)))
    return tau.astype(float), msd
def _msd_summary(msd: np.ndarray) -> Dict[str, float]:
    """
    msd: (N,T-1)
    Summarize by median across tracks for a few lags.
    """
    if msd.size == 0:
        return {"ok": False}
    med = np.nanmedian(msd, axis=0)
    return {
        "ok": True,
        "msd_tau1_median": float(med[0]) if med.size >= 1 else np.nan,
        "msd_tau2_median": float(med[1]) if med.size >= 2 else np.nan,
        "msd_tau3_median": float(med[2]) if med.size >= 3 else np.nan,
        "msd_last_median": float(med[-1]) if med.size >= 1 else np.nan,
    }


# ------------------------------------------------------------
# Full validation runner with plots
# ------------------------------------------------------------

@dataclass
class DriftValidationResult:
    corr_mean_orig: float
    corr_median_orig: float
    corr_mean_corr: float
    corr_median_corr: float
    neighbor_relstd_median_orig: float
    neighbor_relstd_p90_orig: float
    neighbor_relstd_median_corr: float
    neighbor_relstd_p90_corr: float
    msd_tau: np.ndarray
    msd_median_orig: np.ndarray
    msd_median_corr: np.ndarray

def validate_drift_correction(df_orig: pd.DataFrame,
                              df_corr: pd.DataFrame,
                              out_prefix: Optional[str] = None,
                              k_neighbors: int = 8,
                              min_overlap_corr: int = 4) -> DriftValidationResult:
    """
    Produces:
      * pairwise trace correlation histograms
      * neighbor distance stability comparison
      * MSD median curves before and after
    """
    df_orig = _ensure_numeric_tracks(df_orig, pos_cols=("Position X", "Position Y", "Position Z"))
    df_corr = _ensure_numeric_tracks(df_corr, pos_cols=("Position X_corr", "Position Y_corr", "Position Z_corr"))

    # Pivot
    times_o, tids_o, Xo = _pivot_tracks(df_orig, "Position X", "Position Y", "Position Z")
    times_c, tids_c, Xc = _pivot_tracks(df_corr, "Position X_corr", "Position Y_corr", "Position Z_corr")

    # Ensure same ordering: restrict to common track ids and times
    common_t = np.intersect1d(times_o, times_c)
    common_id = np.intersect1d(tids_o, tids_c)

    def _subselect(times, tids, X, common_t, common_id):
        tid_to_i = {float(t): i for i, t in enumerate(tids)}
        t_to_j = {float(t): j for j, t in enumerate(times)}
        I = np.array([tid_to_i[float(tid)] for tid in common_id], dtype=int)
        J = np.array([t_to_j[float(t)] for t in common_t], dtype=int)
        return X[I][:, J], common_t, common_id

    Xo2, times2, tids2 = _subselect(times_o, tids_o, Xo, common_t, common_id)
    Xc2, _, _ = _subselect(times_c, tids_c, Xc, common_t, common_id)

    # Pairwise correlation on displacement norms
    Do = _displacements_relative_to_t0(Xo2)
    Dc = _displacements_relative_to_t0(Xc2)

    co = _pairwise_trace_correlation(Do, min_overlap=min_overlap_corr)
    cc = _pairwise_trace_correlation(Dc, min_overlap=min_overlap_corr)

    corr_mean_orig = float(np.nanmean(co)) if co.size else np.nan
    corr_median_orig = float(np.nanmedian(co)) if co.size else np.nan
    corr_mean_corr = float(np.nanmean(cc)) if cc.size else np.nan
    corr_median_corr = float(np.nanmedian(cc)) if cc.size else np.nan

    # Neighbor distance invariance
    ns_o = _neighbor_distance_stats(Xo2, k_neighbors=k_neighbors)
    ns_c = _neighbor_distance_stats(Xc2, k_neighbors=k_neighbors)

    # MSD
    tau, msd_o = _msd_per_track(Xo2)
    _, msd_c = _msd_per_track(Xc2)

    msd_med_o = np.nanmedian(msd_o, axis=0)
    msd_med_c = np.nanmedian(msd_c, axis=0)

    # Plots
    if out_prefix is not None:
        # Correlation hist
        fig = plt.figure()
        if co.size:
            plt.hist(co, bins=40, alpha=0.6, label="original")
        if cc.size:
            plt.hist(cc, bins=40, alpha=0.6, label="corrected")
        plt.xlabel("pairwise corr of ||displacement(t)||")
        plt.ylabel("count")
        plt.title("Common mode drift proxy: pairwise correlations")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_prefix + "_validate_pairwise_corr_hist.png", dpi=300)
        plt.close(fig)

        # MSD median curves
        fig = plt.figure()
        plt.plot(tau, msd_med_o, marker="o", label="original")
        plt.plot(tau, msd_med_c, marker="o", label="corrected")
        plt.xlabel("lag (frames)")
        plt.ylabel("MSD (µm^2)")
        plt.title("MSD median across tracks")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_prefix + "_validate_msd_median.png", dpi=300)
        plt.close(fig)

        # Neighbor distance stats bar style
        fig = plt.figure()
        labels = ["median rel std", "p90 rel std"]
        vals_o = [ns_o.get("rel_std_median", np.nan), ns_o.get("rel_std_p90", np.nan)]
        vals_c = [ns_c.get("rel_std_median", np.nan), ns_c.get("rel_std_p90", np.nan)]
        x = np.arange(len(labels))
        w = 0.35
        plt.bar(x - w/2, vals_o, width=w, label="original")
        plt.bar(x + w/2, vals_c, width=w, label="corrected")
        plt.xticks(x, labels, rotation=0)
        plt.ylabel("relative distance variability")
        plt.title(f"Local geometry stability (k={k_neighbors})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_prefix + "_validate_neighbor_distance_stats.png", dpi=300)
        plt.close(fig)

    res = DriftValidationResult(
        corr_mean_orig=corr_mean_orig,
        corr_median_orig=corr_median_orig,
        corr_mean_corr=corr_mean_corr,
        corr_median_corr=corr_median_corr,
        neighbor_relstd_median_orig=float(ns_o.get("rel_std_median", np.nan)),
        neighbor_relstd_p90_orig=float(ns_o.get("rel_std_p90", np.nan)),
        neighbor_relstd_median_corr=float(ns_c.get("rel_std_median", np.nan)),
        neighbor_relstd_p90_corr=float(ns_c.get("rel_std_p90", np.nan)),
        msd_tau=tau,
        msd_median_orig=msd_med_o,
        msd_median_corr=msd_med_c,
    )
    return res
def print_validation_summary(res: DriftValidationResult):
    print("Validation summary")
    print(f"Pairwise corr mean:   original {res.corr_mean_orig:.3f}   corrected {res.corr_mean_corr:.3f}")
    print(f"Pairwise corr median: original {res.corr_median_orig:.3f}   corrected {res.corr_median_corr:.3f}")
    print(f"Neighbor rel std median: original {res.neighbor_relstd_median_orig:.4f}   corrected {res.neighbor_relstd_median_corr:.4f}")
    print(f"Neighbor rel std p90:    original {res.neighbor_relstd_p90_orig:.4f}   corrected {res.neighbor_relstd_p90_corr:.4f}")
    if res.msd_median_orig.size:
        print(f"MSD median tau=1: original {res.msd_median_orig[0]:.3f}   corrected {res.msd_median_corr[0]:.3f}")
        print(f"MSD median last:  original {res.msd_median_orig[-1]:.3f}   corrected {res.msd_median_corr[-1]:.3f}")


# ------------------------------------------------------------
# Example usage inside your notebook
# ------------------------------------------------------------

# Pick one file you already processed
for xls_file in xls_files:
    # xls_file = xls_files[0]
    df_orig = data_frames[xls_file]
    df_corr = df_corrs[xls_file]

    base = os.path.splitext(xls_file)[0]
    out_prefix = os.path.join(RESULTS_path, base)

    res = validate_drift_correction(
        df_orig=df_orig,
        df_corr=df_corr,
        out_prefix=out_prefix,
        k_neighbors=10,
        min_overlap_corr=4
    )
    print_validation_summary(res)
    print("Saved validation plots with prefix:", out_prefix)

# %% END