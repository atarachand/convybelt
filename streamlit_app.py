# Placeholder Streamlit app. Replace with your full app code.
# ✅ Streamlit App with Live Sensor Feed + Dashboard Summary Tab

import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import os

st.set_page_config(page_title="Conveyor Belt Monitor", layout="wide")

# Sidebar Navigation
page = st.sidebar.selectbox("📂 Select View", ["📡 Live Monitor", "📊 Dashboard Summary"])

# Common Setup
if not os.path.exists("rul_predictor.h5") or not os.path.exists("anomaly_detector.h5"):
    np.random.seed(42)
    n_samples = 5000
    time_data = np.arange(n_samples)
    temp = 60 + np.random.normal(0, 0.5, n_samples).cumsum() / 50
    vib = 0.2 + np.random.normal(0, 0.02, n_samples).cumsum() / 20
    spd = 1.0 + np.random.normal(0, 0.05, n_samples)
    load = 100 + 5 * np.sin(time_data / 200) + np.random.normal(0, 2, n_samples)
    rul = np.flip(np.linspace(0, 100, n_samples))
    anomaly = np.zeros(n_samples)
    anomaly[np.random.choice(n_samples, size=50, replace=False)] = 1
    vib[anomaly == 1] += np.random.uniform(0.3, 0.7, size=(anomaly == 1).sum())
    temp[anomaly == 1] += np.random.uniform(5, 10, size=(anomaly == 1).sum())

    df = pd.DataFrame({
        'temperature': temp,
        'vibration': vib,
        'speed': spd,
        'load': load,
        'rul': rul,
        'anomaly': anomaly
    })

    features = df[['temperature', 'vibration', 'speed', 'load']]
    target_rul = df['rul']
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target_rul, test_size=0.2, random_state=42)

    rul_model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(4,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    rul_model.compile(optimizer='adam', loss='mse')
    rul_model.fit(X_train, y_train, epochs=10, verbose=0)
    rul_model.save("rul_predictor.h5")

    normal_data = features_scaled[df['anomaly'] == 0]
    input_layer = tf.keras.Input(shape=(4,))
    encoded = layers.Dense(32, activation='relu')(input_layer)
    encoded = layers.Dense(16, activation='relu')(encoded)
    decoded = layers.Dense(32, activation='relu')(encoded)
    decoded = layers.Dense(4, activation='linear')(decoded)

    autoencoder = models.Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(normal_data, normal_data, epochs=10, verbose=0)
    autoencoder.save("anomaly_detector.h5")

    pd.DataFrame(scaler.data_max_, index=features.columns).to_csv("scaler_max.csv")
    pd.DataFrame(scaler.data_min_, index=features.columns).to_csv("scaler_min.csv")

# Load models and scalers
rul_model = tf.keras.models.load_model("rul_predictor.h5", compile=False)
anomaly_model = tf.keras.models.load_model("anomaly_detector.h5", compile=False)
max_vals = pd.read_csv("scaler_max.csv", index_col=0).values.flatten()
min_vals = pd.read_csv("scaler_min.csv", index_col=0).values.flatten()

# Shared prediction history
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["timestamp", "temperature", "vibration", "speed", "load", "rul", "anomaly_score"])

# 📡 Live Monitor Page
if page == "📡 Live Monitor":
    st.title("📡 Live Conveyor Belt Health Monitoring")
    col1, col2 = st.columns(2)
    rul_plot = st.empty()
    anom_plot = st.empty()

    run_live = st.sidebar.checkbox("▶️ Start Live Sensor Feed")

    if run_live:
        st.sidebar.success("✅ Live mode ON. Refresh or uncheck to stop.")
        for _ in range(200):
            temp = np.random.uniform(60, 100)
            vib = np.random.uniform(0.2, 1.0)
            spd = np.random.uniform(0.5, 2.0)
            load = np.random.uniform(70, 140)

            x = np.array([[temp, vib, spd, load]])
            x_scaled = (x - min_vals) / (max_vals - min_vals + 1e-10)

            rul = rul_model.predict(x_scaled)[0][0]
            recon = anomaly_model.predict(x_scaled)
            loss = np.mean(np.square(x_scaled - recon))

            ts = pd.Timestamp.now()
            new_row = pd.DataFrame([[ts, temp, vib, spd, load, rul, loss]],
                                   columns=st.session_state.history.columns)
            st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)

            col1.metric("📉 Predicted RUL", f"{rul:.2f} hours")
            col2.metric("🔎 Anomaly Score", f"{loss:.6f}")

            if rul < 10:
                st.error("🚨 Critical RUL below 10 hours!")
            if loss > 0.01:
                st.warning("⚠️ High Anomaly Detected!")

            if len(st.session_state.history) > 1:
                df_sorted = st.session_state.history.sort_values("timestamp")
                rul_plot.line_chart(df_sorted.set_index("timestamp")["rul"])
                anom_plot.line_chart(df_sorted.set_index("timestamp")["anomaly_score"])

            time.sleep(1)

        st.success("✅ Stream ended. Displaying summary log...")
        st.dataframe(st.session_state.history.tail(10))
        st.download_button("📥 Download Log CSV", data=st.session_state.history.to_csv(index=False), file_name="sensor_log.csv")
    else:
        st.info("🔴 Live mode is OFF. Use the sidebar to start.")

# 📊 Dashboard Page
elif page == "📊 Dashboard Summary":
    st.title("📊 Summary of Predictions")

    if len(st.session_state.history) < 1:
        st.info("📭 No prediction data available. Run the live simulation first.")
    else:
        df = st.session_state.history.copy()

        col1, col2, col3 = st.columns(3)
        col1.metric("Average RUL", f"{df['rul'].mean():.2f} hours")
        col2.metric("Max Anomaly Score", f"{df['anomaly_score'].max():.6f}")
        col3.metric("Critical Events", f"{(df['rul'] < 10).sum()} detected")

        st.line_chart(df.set_index("timestamp")["rul"])
        st.line_chart(df.set_index("timestamp")["anomaly_score"])
        st.dataframe(df.tail(20))
        st.download_button("📥 Download Full Log", data=df.to_csv(index=False), file_name="dashboard_log.csv")
