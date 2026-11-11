# app.py - Prototipe Customer Value Profile (Streamlit)
# Jalankan: streamlit run app.py
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import xgb_compat  # noqa: F401  (pemuatan XGBoost di Windows)
from inference_v3 import predict_profile
from sk_compat import perbaiki_bundle

BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "stacking_bundle.joblib"
PROFILE_PATH = BASE / "data" / "01_latest_profile_with_context.csv"
CONTOH_PATH = BASE / "data" / "contoh_snapshot.csv"

# Nama segmen mengikuti Bab IV. Amber hanya dipakai untuk segmen bernilai tinggi.
SEGMEN = {0: "Kontrak aktif bernilai tinggi", 1: "Pembeli jarang"}
WARNA = {SEGMEN[0]: "#B7791F", SEGMEN[1]: "#6B7A8F"}
BADGE = {0: "orange", 1: "gray"}
URAIAN = {
    0: "Pelanggan dengan kontrak Term aktif yang bertransaksi rutin dan bernilai besar.",
    1: "Pelanggan yang bertransaksi lebih jarang dengan nilai pembelian lebih kecil.",
}
# R² stacking per segmen pada data uji (Bab IV).
R2_UJI = {0: "0,5669", 1: "0,2178"}
FONT = "Segoe UI"

st.set_page_config(page_title="Customer Value Profile", page_icon=":material/insights:", layout="wide")


# ---------- data dan model ----------
@st.cache_resource
def load_bundle():
    return perbaiki_bundle(joblib.load(MODEL_PATH))


@st.cache_data
def load_profile():
    df = pd.read_csv(PROFILE_PATH, parse_dates=["As_Of_Date"])
    df["Segmen"] = df["Cluster"].map(SEGMEN)
    df["Label"] = df["Customer_ID"] + " · " + df["Customer_Name"]
    return df


def kolom_wajib(bundle):
    cols = set()
    for a in bundle["base_order"]:
        b = bundle["base_models"][a]
        cols |= set(b["columns"])
        if b["cfg"].get("target_mode") == "residual":
            cols.add("Baseline_12M_USD")
    cols |= set(bundle["kmeans"]["preprocessor"]["columns"])
    cols |= set(bundle["meta"].get("extra_columns") or [])
    return sorted(cols)


# ---------- format angka (gaya Indonesia) ----------
def angka(x, des=0):
    s = f"{x:,.{des}f}"
    return s.replace(",", "#").replace(".", ",").replace("#", ".")


def usd(x):
    return f"USD {angka(x)}"


def usd_juta(x, des=1):
    return f"USD {angka(x / 1e6, des)} juta"


def persen(x, des=0):
    return f"{angka(x * 100, des)}%"


BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]


def tanggal(d):
    return f"{d.day} {BULAN[d.month - 1]} {d.year}"


def bulan_tahun(d):
    return f"{BULAN[d.month - 1]} {d.year}"


def kepala(judul, keterangan):
    st.caption("CUSTOMER VALUE PROFILE · DATA SINTETIS PENJUALAN BATU BARA B2B")
    st.title(judul)
    st.write(keterangan)


def gaya_grafik(chart):
    return (
        chart.configure(font=FONT)
        .configure_view(stroke=None)
        .configure_axis(labelFontSize=12, titleFontSize=12, titleFontWeight="normal", labelColor="#4A524D",
                        titleColor="#4A524D", gridColor="#E3E6E1", domainColor="#C9CEC8", tickColor="#C9CEC8")
    )


LABEL_JUTA = "replace(format(datum.value / 1000000, '.2~f'), '.', ',') + ' jt'"


# ---------- halaman 1: ringkasan segmen ----------
def halaman_ringkasan():
    df = load_profile()
    total = df["Predicted_6M_CLV_Proxy_USD"].sum()
    porsi = df.groupby("Cluster")["Predicted_6M_CLV_Proxy_USD"].sum() / total
    kepala("Ringkasan segmen",
           "Perbandingan dua segmen pelanggan pada tanggal acuan terbaru setiap pelanggan. "
           "Nilai prediksi adalah proksi CLV, yaitu kontribusi terdiskonto enam bulan setelah tanggal acuan. "
           f"Tanggal acuan berkisar {tanggal(df['As_Of_Date'].min())} sampai {tanggal(df['As_Of_Date'].max())}.")

    k1, k2, k3 = st.columns(3)
    k1.metric("Pelanggan", angka(len(df)), border=True)
    k2.metric("Total prediksi 6 bulan", usd_juta(total), border=True)
    k3.metric("Porsi nilai segmen bernilai tinggi", persen(porsi.get(0, 0)), border=True,
              help="Bagian total prediksi yang berasal dari segmen kontrak aktif bernilai tinggi.")

    st.subheader("Profil tiap segmen")
    kiri, kanan = st.columns(2)
    for kol, c in zip((kiri, kanan), (0, 1)):
        with kol:
            st.badge(SEGMEN[c], color=BADGE[c])
            st.caption(URAIAN[c])

    def ringkas(s):
        return [angka(len(s)), usd_juta(s["Predicted_6M_CLV_Proxy_USD"].sum()),
                usd_juta(s["Predicted_6M_CLV_Proxy_USD"].median(), 2), angka(s["Frequency_12M"].median()),
                f"{angka(s['Total_Quantity_12M_Tonnes'].median())} t", persen(s["Term_Transaction_Share"].median())]

    tabel = pd.DataFrame({
        "Ukuran": ["Jumlah pelanggan", "Total prediksi 6 bulan", "Median prediksi per pelanggan",
                   "Median transaksi 12 bulan", "Median tonase 12 bulan", "Median porsi transaksi Term"],
        SEGMEN[0]: ringkas(df[df["Cluster"] == 0]),
        SEGMEN[1]: ringkas(df[df["Cluster"] == 1]),
    })
    st.dataframe(tabel, hide_index=True, width="stretch")
    st.caption(f"Keandalan prediksi pada data uji (R²): {R2_UJI[0]} untuk segmen bernilai tinggi dan "
               f"{R2_UJI[1]} untuk segmen pembeli jarang.")

    st.subheader("Sebaran prediksi per pelanggan")
    st.caption("Setiap garis kecil mewakili satu pelanggan. Sumbu mendatar memakai skala logaritma; "
               "garis tebal menandai median segmen.")
    plot = df[df["Predicted_6M_CLV_Proxy_USD"] > 0]
    warna = alt.Color("Segmen:N", scale=alt.Scale(domain=list(WARNA), range=list(WARNA.values())), legend=None)
    sumbu_x = alt.X("Predicted_6M_CLV_Proxy_USD:Q", title="Prediksi proksi CLV 6 bulan (USD, skala log)",
                    scale=alt.Scale(type="log"), axis=alt.Axis(labelExpr=LABEL_JUTA, values=[1e5, 3e5, 1e6, 3e6, 1e7]))
    sumbu_y = alt.Y("Segmen:N", title=None, sort=list(WARNA), axis=alt.Axis(labelLimit=260))
    titik = alt.Chart(plot).mark_tick(thickness=1.5, size=26, opacity=0.55).encode(
        x=sumbu_x, y=sumbu_y, color=warna,
        tooltip=[alt.Tooltip("Customer_ID:N", title="ID"), alt.Tooltip("Customer_Name:N", title="Pelanggan"),
                 alt.Tooltip("Predicted_6M_CLV_Proxy_USD:Q", title="Prediksi (USD)", format=",.0f")])
    median = alt.Chart(plot).mark_tick(thickness=4, size=40, color="#1F2421").encode(
        x=alt.X("median(Predicted_6M_CLV_Proxy_USD):Q"), y=sumbu_y)
    st.altair_chart(gaya_grafik((titik + median).properties(height=170)), width="stretch")

    st.subheader("Sepuluh pelanggan dengan prediksi tertinggi")
    top = df.nlargest(10, "Predicted_6M_CLV_Proxy_USD")
    st.dataframe(
        pd.DataFrame({
            "Pelanggan": top["Customer_Name"], "ID": top["Customer_ID"], "Negara": top["Country"],
            "Segmen": top["Segmen"],
            "Prediksi 6 bulan (USD)": top["Predicted_6M_CLV_Proxy_USD"].map(angka),
            "Kontribusi 12 bln (USD)": top["Historical_Contribution_12M_USD"].map(angka),
            "Transaksi 12 bln": top["Frequency_12M"].astype(int),
        }),
        hide_index=True, width="stretch")
    st.caption("Model: stacking ensemble (Random Forest, XGBoost, LightGBM, MLP) dengan meta-learner Ridge. "
               "Segmentasi: K-Means dua cluster. Seluruh data bersifat sintetis.")


# ---------- halaman 2: profil pelanggan ----------
def halaman_profil():
    df = load_profile().sort_values("Predicted_6M_CLV_Proxy_USD", ascending=False)
    kepala("Profil pelanggan", "Pilih pelanggan untuk melihat segmen, prediksi, dan perilaku transaksi 12 bulan terakhir.")
    pilihan = st.selectbox("Pelanggan", df["Label"], index=0, placeholder="Ketik ID atau nama pelanggan")
    row = df.loc[df["Label"] == pilihan].iloc[0]
    c = int(row["Cluster"])
    seg = df[df["Cluster"] == c]

    st.header(row["Customer_Name"])
    st.badge(SEGMEN[c], color=BADGE[c])
    st.caption(f"{row['Customer_ID']} · {row['Country']} · tanggal acuan {tanggal(row['As_Of_Date'])}")

    posisi = (seg["Predicted_6M_CLV_Proxy_USD"] < row["Predicted_6M_CLV_Proxy_USD"]).mean()
    a, b, d = st.columns(3)
    a.metric("Prediksi proksi CLV 6 bulan", usd(row["Predicted_6M_CLV_Proxy_USD"]), border=True)
    b.metric("Posisi dalam segmen", f"di atas {persen(posisi)}", border=True,
             help="Persentase pelanggan dalam segmen yang sama dengan prediksi lebih rendah.")
    d.metric("Kontribusi 12 bulan terakhir", usd(row["Historical_Contribution_12M_USD"]), border=True)

    if c == 1:
        st.info(f"Prediksi pada segmen pembeli jarang kurang pasti (R² data uji {R2_UJI[1]}). "
                "Baca angka prediksi bersama riwayat transaksi di bawah.", icon=":material/info:")

    st.subheader("Dibandingkan dengan median segmen")
    banding = pd.DataFrame({
        "Ukuran": ["Transaksi 12 bulan", "Tonase 12 bulan (t)", "Kontrak 12 bulan", "Porsi transaksi Term",
                   "Kontribusi 12 bulan (USD)", "Prediksi 6 bulan (USD)"],
        "Pelanggan ini": [angka(row["Frequency_12M"]), angka(row["Total_Quantity_12M_Tonnes"]),
                          angka(row["Contract_Count_12M"]), persen(row["Term_Transaction_Share"]),
                          angka(row["Historical_Contribution_12M_USD"]), angka(row["Predicted_6M_CLV_Proxy_USD"])],
        "Median segmen": [angka(seg["Frequency_12M"].median()), angka(seg["Total_Quantity_12M_Tonnes"].median()),
                          angka(seg["Contract_Count_12M"].median()), persen(seg["Term_Transaction_Share"].median()),
                          angka(seg["Historical_Contribution_12M_USD"].median()),
                          angka(seg["Predicted_6M_CLV_Proxy_USD"].median())],
    })
    st.dataframe(banding, hide_index=True, width="stretch")


# ---------- halaman 3: prediksi dari file ----------
def halaman_prediksi():
    bundle = load_bundle()
    wajib = kolom_wajib(bundle)
    kepala("Prediksi dari file snapshot",
           "Unggah ringkasan transaksi pelanggan per tanggal acuan (snapshot) untuk memperoleh segmen dan "
           "prediksi proksi CLV enam bulan.")

    st.subheader("1. Siapkan file")
    st.write(f"File CSV berisi satu baris per pelanggan per tanggal acuan dengan {len(wajib)} kolom wajib "
             "yang sama seperti panel penelitian. Contoh file dapat diunduh di bawah.")
    st.download_button("Unduh contoh file", CONTOH_PATH.read_bytes(), "contoh_snapshot.csv", "text/csv",
                       icon=":material/download:")

    st.subheader("2. Unggah file")
    up = st.file_uploader("File snapshot (.csv)", type="csv")
    if up is None:
        return
    frame = pd.read_csv(up)
    hilang = [k for k in wajib if k not in frame.columns]
    if hilang:
        st.error(f"{len(hilang)} kolom wajib tidak ditemukan: {', '.join(hilang[:12])}"
                 + (" dan lainnya." if len(hilang) > 12 else "."), icon=":material/error:")
        return
    if "As_Of_Date" in frame:
        frame["As_Of_Date"] = pd.to_datetime(frame["As_Of_Date"])
    st.success(f"{angka(len(frame))} baris dan seluruh {len(wajib)} kolom wajib ditemukan.", icon=":material/check:")

    st.subheader("3. Hasil prediksi")
    hasil = predict_profile(bundle, frame)
    hasil["Segmen"] = hasil["Cluster"].map(SEGMEN)
    a, b, d = st.columns(3)
    a.metric("Snapshot diprediksi", angka(len(hasil)), border=True)
    b.metric("Total prediksi 6 bulan", usd_juta(hasil["Predicted_6M_CLV_Proxy_USD"].sum()), border=True)
    d.metric("Masuk segmen bernilai tinggi", angka((hasil["Cluster"] == 0).sum()), border=True)
    tampil = pd.DataFrame({
        "ID": hasil.get("Customer_ID", pd.Series(range(len(hasil)))),
        "Tanggal acuan": hasil["As_Of_Date"].map(tanggal) if "As_Of_Date" in hasil else "",
        "Segmen": hasil["Segmen"],
        "Prediksi 6 bulan (USD)": hasil["Predicted_6M_CLV_Proxy_USD"].map(angka),
    })
    st.dataframe(tampil, hide_index=True, width="stretch", height=360)
    st.download_button("Unduh hasil lengkap (CSV)", hasil.to_csv(index=False), "hasil_prediksi.csv", "text/csv",
                       icon=":material/download:")
    st.caption("File hasil juga memuat prediksi setiap base learner (kolom Pred_*).")


nav = st.navigation(
    [
        st.Page(halaman_ringkasan, title="Ringkasan segmen", icon=":material/donut_small:", url_path="ringkasan", default=True),
        st.Page(halaman_profil, title="Profil pelanggan", icon=":material/person_search:", url_path="profil"),
        st.Page(halaman_prediksi, title="Prediksi dari file", icon=":material/upload_file:", url_path="prediksi"),
    ],
    position="top",
)
nav.run()
