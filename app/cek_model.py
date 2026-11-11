"""Cek cepat: memuat model, memprediksi contoh snapshot, dan membandingkan dengan hasil notebook."""
import traceback
import warnings
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas
import sklearn
import xgboost

import xgb_compat  # noqa: F401
from sk_compat import perbaiki_bundle
import inference_v3 as inf

BASE = Path(__file__).parent
print("versi:", "sklearn", sklearn.__version__, "xgboost", xgboost.__version__, "lightgbm", lightgbm.__version__,
      "numpy", np.__version__, "pandas", pandas.__version__)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    b = perbaiki_bundle(joblib.load(BASE / "models" / "stacking_bundle.joblib"))
print("bundle OK; peringatan saat memuat:", len(w))
snap = pandas.read_csv(BASE / "data" / "contoh_snapshot.csv", parse_dates=["As_Of_Date"])
gagal = False
base = {}
for a in b["base_order"]:
    try:
        base[a] = inf.predict_base(b["base_models"][a], snap); print("base", a, "OK")
    except Exception:
        gagal = True; print("base", a, "GAGAL"); traceback.print_exc(limit=-3)
try:
    cl = inf.cluster_labels(b["kmeans"], snap); print("kmeans OK")
except Exception:
    gagal = True; print("kmeans GAGAL"); traceback.print_exc(limit=-3)
if not gagal:
    stack = inf.predict_meta(b["meta"], np.column_stack([base[a] for a in b["base_order"]]), snap)
    ref = pandas.read_csv(BASE / "data" / "01_latest_profile_with_context.csv").set_index("Snapshot_ID").loc[snap["Snapshot_ID"]]
    print("selisih prediksi maks (USD):", float(np.abs(stack - ref["Predicted_6M_CLV_Proxy_USD"].to_numpy()).max()))
    print("cluster sama:", bool((cl == ref["Cluster"].to_numpy()).all()))
    print("CEK SELESAI")
else:
    raise SystemExit(1)
