import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


# =========================================================
# 1. KONFIGURASI DASHBOARD
# =========================================================

st.set_page_config(
    page_title="Rain Tomorrow Prediction",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = Path("best_rain_prediction_model.pkl")
IMAGE_PATH = Path("sunflower_rain.jpg")


# =========================================================
# 2. HEADER DASHBOARD
# =========================================================

def show_header():
    # Membagi halaman menjadi bagian teks dan gambar
    col_text, col_image = st.columns(
        [3, 2],
        gap="large",
        vertical_alignment="center"
    )

    with col_text:
        st.title("Welcome to my first machine learning dashboard")

        st.subheader(
            "This dashboard created by Jesi Pebralia"
        )

        st.write(
            "This application predicts whether it will rain "
            "tomorrow based on today's weather observations."
        )

        st.caption(
            "Machine Learning | Weather Forecasting | Indonesia Climate"
        )

    with col_image:
        if IMAGE_PATH.exists():
            st.image(
                Image.open(IMAGE_PATH),
                use_container_width=True
            )
        else:
            st.warning(
                "Gambar tidak ditemukan. Simpan gambar sebagai "
                "sunflower_rain.jpg di folder aplikasi."
            )

    st.divider()


# =========================================================
# 3. MEMUAT MODEL PICKLE
# =========================================================

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        model_export = pickle.load(file)

    pipeline = model_export["pipeline"]
    threshold = float(model_export["best_threshold"])
    features = model_export["features"]

    return pipeline, threshold, features


# =========================================================
# 4. INFORMASI APLIKASI
# =========================================================

def rain():
    st.header("🌧️ Rain Tomorrow Prediction")

    st.write(
        """
        Aplikasi ini digunakan untuk memprediksi kondisi hujan
        atau tidak hujan pada hari berikutnya berdasarkan data
        meteorologi hari ini.

        Pengguna dapat memasukkan data cuaca secara manual
        melalui slider atau mengunggah file CSV. Model machine
        learning yang telah dilatih sebelumnya kemudian digunakan
        untuk menghasilkan prediksi.

        Dataset:
        [Indonesia Climate - Kaggle]
        (https://www.kaggle.com/datasets/greegtitan/indonesia-climate)
        """
    )


# =========================================================
# 5. INPUT DATA MENGGUNAKAN SLIDER
# =========================================================

def manual_input():
    st.subheader("Input Data Cuaca Hari Ini")

    data = {}

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Suhu dan Kelembapan")

        data["Tn"] = st.slider(
            "Suhu minimum (°C)",
            -10.0, 35.0, 23.0, 0.1
        )

        data["Tx"] = st.slider(
            "Suhu maksimum (°C)",
            -5.0, 45.0, 32.0, 0.1
        )

        data["Tavg"] = st.slider(
            "Suhu rata-rata (°C)",
            -10.0, 40.0, 27.0, 0.1
        )

        data["RH_avg"] = st.slider(
            "Kelembapan rata-rata (%)",
            0.0, 100.0, 80.0, 1.0
        )

        data["RR"] = st.slider(
            "Curah hujan hari ini (mm)",
            0.0, 300.0, 0.0, 0.1
        )

    with col2:
        st.markdown("#### Penyinaran dan Angin")

        data["ss"] = st.slider(
            "Durasi penyinaran matahari (jam)",
            0.0, 12.0, 6.0, 0.1
        )

        data["ff_x"] = st.slider(
            "Kecepatan angin maksimum (m/s)",
            0.0, 40.0, 5.0, 0.1
        )

        data["ff_avg"] = st.slider(
            "Kecepatan angin rata-rata (m/s)",
            0.0, 30.0, 2.0, 0.1
        )

        data["month"] = st.slider(
            "Bulan dalam setahun",
            1, 12, 1, 1
        )

        data["dayofyear"] = st.slider(
            "Hari dalam setahun",
            1, 366, 1, 1
        )

    return pd.DataFrame([data])


# =========================================================
# 6. FUNGSI PREDIKSI
# =========================================================

def predict_rain(data, pipeline, threshold, features):
    missing = [
        feature for feature in features
        if feature not in data.columns
    ]

    if missing:
        raise ValueError(
            "Kolom fitur tidak lengkap: "
            + ", ".join(missing)
        )

    X = data[features].copy()

    # Pastikan semua fitur numerik
    for column in features:
        X[column] = pd.to_numeric(
            X[column], errors="raise"
        )

    if X.isnull().any().any():
        raise ValueError(
            "Data mengandung nilai kosong (NaN)."
        )

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(
            "Data mengandung nilai yang tidak valid."
        )

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X)

        # Asumsi: kelas positif adalah kelas hujan.
        # Pastikan urutan kelas sesuai dengan target training.
        positive_probability = probabilities[:, 1]

        predictions = (
            positive_probability >= threshold
        ).astype(int)

        return predictions, positive_probability

    predictions = pipeline.predict(X)

    return predictions, None


# =========================================================
# 7. MENAMPILKAN HASIL PREDIKSI
# =========================================================

def show_results(data, pipeline, threshold, features):
    try:
        predictions, probabilities = predict_rain(
            data,
            pipeline,
            threshold,
            features
        )

        st.subheader("Hasil Prediksi untuk Besok")

        if probabilities is not None:
            # Hasil ini mengasumsikan label 1 = hujan
            # dan label 0 = tidak hujan.
            for i, prediction in enumerate(predictions):
                if prediction == 1:
                    st.error(
                        f"🌧️ Data ke-{i + 1}: "
                        "Diprediksi HUJAN besok"
                    )
                elif prediction == 0:
                    st.success(
                        f"☀️ Data ke-{i + 1}: "
                        "Diprediksi TIDAK HUJAN besok"
                    )
                else:
                    st.warning(
                        f"Label prediksi tidak dikenali: {prediction}"
                    )

                st.metric(
                    f"Probabilitas hujan — data ke-{i + 1}",
                    f"{probabilities[i] * 100:.2f}%"
                )

            st.caption(
                f"Threshold model: {threshold:.3f}"
            )

            results = data.copy()
            results["Prediksi_kode"] = predictions
            results["Prediksi"] = [
                "Hujan" if p == 1 else "Tidak Hujan"
                for p in predictions
            ]
            results["Probabilitas_hujan_persen"] = (
                probabilities * 100
            ).round(2)

        else:
            results = data.copy()
            results["Prediksi_kode"] = predictions

            st.info(
                "Model tidak menyediakan probabilitas. "
                "Hasil berikut berasal dari predict()."
            )

        st.dataframe(
            results,
            use_container_width=True
        )

        st.download_button(
            label="Unduh Hasil Prediksi (CSV)",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="rain_prediction_results.csv",
            mime="text/csv"
        )

    except (ValueError, KeyError) as error:
        st.error(f"Prediksi gagal: {error}")


# =========================================================
# 8. FUNGSI UTAMA APLIKASI
# =========================================================

def main():

    # PENTING: kedua tulisan pembuka ditampilkan paling atas
    show_header()

    # Informasi aplikasi prediksi
    rain()

    st.divider()

    # Memuat model tersimpan
    try:
        pipeline, threshold, features = load_model()

    except FileNotFoundError:
        st.error(
            "File model tidak ditemukan. Pastikan "
            "best_rain_prediction_model.pkl berada "
            "di folder yang sama dengan app.py."
        )
        st.stop()

    except (KeyError, pickle.UnpicklingError, EOFError) as error:
        st.error(f"Model gagal dimuat: {error}")
        st.stop()

    # Pilihan metode input
    st.sidebar.title("🌦️ Menu Prediksi")

    input_method = st.sidebar.radio(
        "Pilih metode input data:",
        [
            "Input Manual (Slider)",
            "Upload File CSV"
        ]
    )

    # -----------------------------------------------------
    # INPUT MANUAL
    # -----------------------------------------------------

    if input_method == "Input Manual (Slider)":

        input_data = manual_input()

        if st.button(
            "🔮 Prediksi Hujan Besok",
            type="primary"
        ):
            show_results(
                input_data,
                pipeline,
                threshold,
                features
            )

    # -----------------------------------------------------
    # INPUT CSV
    # -----------------------------------------------------

    else:
        st.subheader("Upload File CSV")

        uploaded_file = st.file_uploader(
            "Pilih file CSV berisi data cuaca hari ini",
            type=["csv"]
        )

        if uploaded_file is not None:
            try:
                input_data = pd.read_csv(uploaded_file)

                st.write("Pratinjau data:")
                st.dataframe(
                    input_data.head(),
                    use_container_width=True
                )

                missing = [
                    f for f in features
                    if f not in input_data.columns
                ]

                if missing:
                    st.error(
                        "Kolom berikut belum tersedia: "
                        + ", ".join(missing)
                    )

                elif st.button(
                    "🔮 Prediksi Semua Data CSV",
                    type="primary"
                ):
                    show_results(
                        input_data,
                        pipeline,
                        threshold,
                        features
                    )

            except (
                pd.errors.ParserError,
                UnicodeDecodeError
            ) as error:
                st.error(
                    f"File CSV tidak dapat dibaca: {error}"
                )

    # Footer
    st.divider()
    st.caption(
        "Rain Tomorrow Prediction | Created by Jesi Pebralia"
    )


# =========================================================
# 9. MENJALANKAN APLIKASI
# =========================================================

if __name__ == "__main__":
    main()
