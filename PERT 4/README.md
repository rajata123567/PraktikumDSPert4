# Segmentasi Nasabah Kartu Kredit (K-Means)

Proyek Data Science dengan metodologi CRISP-DM. Aplikasi Streamlit ini memprediksi
segmen nasabah kartu kredit berdasarkan perilaku penggunaannya.

## Struktur
```
app.py
requirements.txt
models/
  scaler.joblib
  prep_config.joblib
  kmeans_model.joblib
  persona.json
  cluster_profile_zscore.csv
  cluster_profile_median.csv
```

## Menjalankan lokal
```
pip install -r requirements.txt
streamlit run app.py
```
