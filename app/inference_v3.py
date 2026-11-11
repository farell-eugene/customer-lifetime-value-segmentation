"""Inferensi Customer Value Profile v3 untuk prototipe Streamlit.

Pemakaian:
    import joblib, pandas as pd
    from inference_v3 import predict_profile
    bundle = joblib.load('hasil_skripsi_v3/<hash>_standard_v3/models/stacking_bundle.joblib')
    profile = predict_profile(bundle, snapshot_df)   # snapshot_df: kolom fitur panel pada tanggal acuan

Fungsi di sini sama dengan fungsi di Skripsi_Multiyear_FULL_v3.ipynb (bagian 3.1, 4.2, dan 5.6).
Seluruh angka berasal dari data sintetis.
"""
import numpy as np
import pandas as pd


def _numeric_inputs(frame, cols, algorithm):
    a = frame[cols].to_numpy(dtype=float)
    a[~np.isfinite(a)] = np.nan
    if algorithm == 'MLP':
        a = np.sign(a) * np.log1p(np.abs(a))
    return a


def predict_base(bundle, frame):
    x = bundle['imputer'].transform(_numeric_inputs(frame, bundle['columns'], bundle['algorithm']))
    if bundle['scaler'] is not None:
        x = bundle['scaler'].transform(x)
    p = bundle['model'].predict(x) * bundle['target_std'] + bundle['target_mean']
    if bundle['cfg']['target_mode'] == 'residual':
        p = p * 1e6 + frame['Baseline_12M_USD'].to_numpy()
    else:
        p = p * 1e6
    return np.maximum(p, 0.)


def _meta_design(predictions, frame, extra_columns):
    x = np.asarray(predictions, dtype=float) / 1e6
    if extra_columns:
        extra = frame[list(extra_columns)].to_numpy(dtype=float)
        x = np.column_stack([x, np.sign(extra) * np.log1p(np.abs(extra))])
    x[~np.isfinite(x)] = np.nan
    return x


def predict_meta(meta, predictions, frame):
    x = _meta_design(predictions, frame, meta['extra_columns'])
    if meta['spec']['kind'] in ('Convex', 'ConvexNoInt', 'NNLS'):
        p = x @ meta['weights'] + meta['intercept']
    else:
        p = meta['model'].predict(meta['scaler'].transform(meta['imputer'].transform(x)))
    return np.maximum(p * 1e6, 0.)


def cluster_labels(cluster_bundle, frame):
    pre = cluster_bundle['preprocessor']
    a = pre['imputer'].transform(frame[pre['columns']])
    if pre['method'] == 'yeo_johnson':
        a = a.copy(); v = pre['variable']; a[:, v] = pre['power'].transform(a[:, v])
    else:
        a = np.sign(a) * np.log1p(np.abs(a))
    return cluster_bundle['model'].predict(pre['scaler'].transform(a))


def predict_profile(bundle, frame):
    """Segmen K-Means + prediksi proksi CLV enam bulan (stacking) untuk setiap snapshot."""
    order = bundle['base_order']
    base = {a: predict_base(bundle['base_models'][a], frame) for a in order}
    stack = predict_meta(bundle['meta'], np.column_stack([base[a] for a in order]), frame)
    out = frame[[c for c in ['Snapshot_ID', 'Customer_ID', 'As_Of_Date'] if c in frame]].copy()
    out['Cluster'] = cluster_labels(bundle['kmeans'], frame)
    for a in order:
        out['Pred_' + a] = base[a]
    out['Predicted_6M_CLV_Proxy_USD'] = stack
    return out
