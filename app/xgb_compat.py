"""Kompatibilitas pemuatan model XGBoost lintas sistem operasi.

Model dilatih di Google Colab (Linux). Pada sebagian instalasi Windows, XGBoost menolak
memuat konfigurasi pelatihan yang tersimpan di dalam pickle ("input stream corrupted").
Bagian model (pohon dan parameter prediksi) tetap utuh, sehingga bila pemuatan biasa gagal,
hanya bagian "Model" yang dimuat. Hasil prediksi tidak berubah.
"""
from xgboost import core

_ASLI = core.Booster.__setstate__
_KUNCI = b"L" + (5).to_bytes(8, "big") + b"Model"


def _bagian_model(buf):
    b = bytes(buf)
    i = b.find(_KUNCI)
    if i < 0:
        raise ValueError("Bagian Model tidak ditemukan pada model XGBoost")
    return bytearray(b[i + len(_KUNCI):-1])


def _setstate(self, state):
    try:
        _ASLI(self, state)
    except core.XGBoostError:
        core.Booster.__init__(self, model_file=_bagian_model(state["handle"]))


core.Booster.__setstate__ = _setstate
