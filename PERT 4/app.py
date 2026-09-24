"""
Aplikasi Segmentasi Nasabah Kartu Kredit (K-Means)
Tahap Deployment - CRISP-DM
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="Segmentasi Nasabah Kartu Kredit",
    page_icon="💳",
    layout="wide",
)


# ---------------------------------------------------------------- load artefak
@st.cache_resource
def load_artifacts():
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    config = joblib.load(MODEL_DIR / "prep_config.joblib")
    kmeans = joblib.load(MODEL_DIR / "kmeans_model.joblib")
    with open(MODEL_DIR / "persona.json", encoding="utf-8") as f:
        persona = json.load(f)
    z_profile = pd.read_csv(MODEL_DIR / "cluster_profile_zscore.csv", index_col=0)
    med_profile = pd.read_csv(MODEL_DIR / "cluster_profile_median.csv", index_col=0)
    return scaler, config, kmeans, persona, z_profile, med_profile


try:
    scaler, config, kmeans, persona, z_profile, med_profile = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "File model tidak ditemukan di folder `models/`. "
        "Pastikan semua file hasil Tahap 3, 4, dan 5 sudah disalin ke sana.\n\n"
        f"Detail: {e}"
    )
    st.stop()

FEATURES = config["features"]
LOG_COLS = config["log_cols"]
IMPUTE = config["impute_values"]
DEFAULTS = med_profile[FEATURES].median()   # nilai awal form = median antar klaster


def cluster_name(c: int) -> str:
    return persona["names"].get(str(c), f"Cluster {c}")


def cluster_reco(c: int) -> str:
    return persona["recommendations"].get(str(c), "-")


# ---------------------------------------------------------------- preprocessing
def preprocess(df_raw: pd.DataFrame) -> np.ndarray:
    """Pipeline sama persis dengan Tahap 3: impute -> log1p -> scaling."""
    X = df_raw[FEATURES].copy().astype(float)
    for col, val in IMPUTE.items():
        X[col] = X[col].fillna(val)
    X = X.fillna(DEFAULTS)                       # pengaman jika ada kolom lain kosong
    X[LOG_COLS] = np.log1p(X[LOG_COLS].clip(lower=0))
    return scaler.transform(X)


def predict(df_raw: pd.DataFrame):
    Xs = preprocess(df_raw)
    return kmeans.predict(Xs), Xs


# ---------------------------------------------------------------- UI
st.title("💳 Segmentasi Nasabah Kartu Kredit")
st.caption(
    "Aplikasi ini memprediksi segmen nasabah berdasarkan perilaku penggunaan kartu kredit "
    "menggunakan model K-Means."
)

tab1, tab2, tab3 = st.tabs(
    ["🔍 Prediksi Satu Nasabah", "📂 Prediksi Banyak Nasabah (CSV)", "📊 Profil Klaster"]
)

# ---- Tab 1: satu nasabah
with tab1:
    st.subheader("Masukkan data perilaku nasabah")
    with st.form("form_nasabah"):
        cols = st.columns(3)
        inputs = {}
        for i, feat in enumerate(FEATURES):
            with cols[i % 3]:
                default = float(DEFAULTS[feat])
                if "FREQUENCY" in feat or feat == "PRC_FULL_PAYMENT":
                    inputs[feat] = st.slider(
                        feat, 0.0, 1.0, float(np.clip(default, 0, 1)), 0.01, key=f"in_{feat}"
                    )
                elif feat == "TENURE":
                    inputs[feat] = st.number_input(
                        feat, min_value=0, max_value=12,
                        value=int(np.clip(round(default), 0, 12)), step=1, key=f"in_{feat}"
                    )
                elif feat.endswith("_TRX"):
                    inputs[feat] = st.number_input(
                        feat, min_value=0, value=int(round(default)), step=1, key=f"in_{feat}"
                    )
                else:
                    inputs[feat] = st.number_input(
                        feat, min_value=0.0, value=default, step=10.0, key=f"in_{feat}"
                    )
        submitted = st.form_submit_button("Prediksi Segmen", type="primary")

    if submitted:
        row = pd.DataFrame([inputs])
        pred, Xs = predict(row)
        c = int(pred[0])

        st.success(f"Segmen nasabah: **{cluster_name(c)}** (Cluster {c})")
        st.markdown(f"**Rekomendasi strategi:** {cluster_reco(c)}")

        st.markdown("**Profil nasabah vs rata-rata klaster** (z-score, 0 = rata-rata seluruh nasabah)")
        cmp_df = pd.DataFrame(
            {
                "Nasabah ini": Xs[0],
                f"Rata-rata Cluster {c}": z_profile.loc[c, FEATURES].values,
            },
            index=FEATURES,
        )
        st.bar_chart(cmp_df)

# ---- Tab 2: batch
with tab2:
    st.subheader("Upload file CSV nasabah")
    st.write(
        "File harus memuat kolom fitur berikut (kolom `CUST_ID` opsional):"
    )
    st.code(", ".join(FEATURES), language=None)

    up = st.file_uploader("Pilih file CSV", type=["csv"])
    if up is not None:
        data = pd.read_csv(up)
        missing_cols = [c for c in FEATURES if c not in data.columns]
        if missing_cols:
            st.error(f"Kolom berikut tidak ditemukan di file: {missing_cols}")
        else:
            pred, _ = predict(data)
            out = data.copy()
            out["Cluster"] = pred
            out["Persona"] = [cluster_name(int(p)) for p in pred]

            st.success(f"{len(out)} nasabah berhasil diprediksi.")
            st.dataframe(out.head(200))

            st.markdown("**Distribusi segmen**")
            st.bar_chart(out["Persona"].value_counts())

            st.download_button(
                "⬇️ Download hasil (CSV)",
                out.to_csv(index=False).encode("utf-8"),
                file_name="hasil_segmentasi.csv",
                mime="text/csv",
            )

# ---- Tab 3: profil klaster
with tab3:
    st.subheader("Ringkasan tiap klaster")
    for c in sorted(z_profile.index):
        with st.expander(f"Cluster {c}: {cluster_name(int(c))}", expanded=False):
            st.markdown(f"**Rekomendasi:** {cluster_reco(int(c))}")
            top = z_profile.loc[c, FEATURES].sort_values()
            st.markdown(
                "**Tertinggi:** " + ", ".join(f"{k} ({v:+.2f})" for k, v in top.tail(3)[::-1].items())
            )
            st.markdown(
                "**Terendah:** " + ", ".join(f"{k} ({v:+.2f})" for k, v in top.head(3).items())
            )

    st.subheader("Heatmap profil klaster (z-score)")
    st.dataframe(
        z_profile[FEATURES].T.style.background_gradient(cmap="coolwarm", axis=None).format("{:.2f}")
    )

    st.subheader("Median tiap fitur per klaster (skala asli)")
    st.dataframe(med_profile[FEATURES].T.round(2))
