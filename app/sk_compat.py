"""Kompatibilitas objek scikit-learn yang disimpan dengan versi 1.6.1 saat dimuat di versi lebih baru.

Atribut internal yang belum ada pada versi lama dilengkapi dari atribut yang setara.
Hasil prediksi diperiksa ulang terhadap keluaran notebook melalui cek_model.py.
"""
from sklearn.impute import SimpleImputer


def _perbaiki_objek(o):
    if isinstance(o, SimpleImputer) and not hasattr(o, "_fill_dtype"):
        o._fill_dtype = getattr(o, "_fit_dtype", o.statistics_.dtype)


def perbaiki_bundle(obj, _seen=None):
    _seen = _seen if _seen is not None else set()
    if id(obj) in _seen:
        return obj
    _seen.add(id(obj))
    _perbaiki_objek(obj)
    if isinstance(obj, dict):
        for v in obj.values():
            perbaiki_bundle(v, _seen)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            perbaiki_bundle(v, _seen)
    elif hasattr(obj, "__dict__") and type(obj).__module__.startswith("sklearn"):
        for v in vars(obj).values():
            if isinstance(v, (dict, list, tuple)) or (hasattr(v, "__dict__") and type(v).__module__.startswith("sklearn")):
                perbaiki_bundle(v, _seen)
    return obj
