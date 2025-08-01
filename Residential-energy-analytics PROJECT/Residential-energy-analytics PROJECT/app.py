import os
import sys
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# ✅ Set base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "energy_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "energy_forecast_model.pkl")

# ✅ Force add scripts directory to Python path
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# ✅ Import recommendations directly (no package prefix needed)
try:
    from recommendations import generate_recommendations
except ImportError as e:
    st.error(f"❌ Failed to import recommendations module: {e}")
    st.stop()

# ✅ Load trained model
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model not found! Run train_model.py first to generate the model.")
    st.stop()

model = joblib.load(MODEL_PATH)

# ✅ Load dataset
df = None
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
else:
    st.warning("⚠️ No default dataset found. Please upload your energy data CSV.")

# ✅ Streamlit UI
st.set_page_config(page_title="Residential Energy Analytics", layout="wide")
st.title("🏡 Residential Energy Analytics Dashboard")
st.write("Monitor energy usage, detect inefficiencies, and get AI-powered recommendations.")

# ✅ File uploader for custom data
uploaded_file = st.file_uploader("Upload your energy data CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])

# ✅ Display chart and predictions
if df is not None and not df.empty:
    st.subheader("📊 Energy Consumption Over Time")
    st.line_chart(df.set_index('timestamp')['consumption'])

    # ✅ Predict next hour consumption
    latest = df.tail(1)
    hour = latest['timestamp'].dt.hour.values[0]
    day = latest['timestamp'].dt.dayofweek.values[0]
    prediction = model.predict([[hour, day]])[0]
    st.metric("🔮 Predicted Next Hour Consumption (kWh)", round(prediction, 2))

    # ✅ Show recommendations
    st.subheader("💡 Energy-Saving Recommendations")
    for tip in generate_recommendations(hour):
        st.write("✅", tip)
else:
    st.info("ℹ️ Upload a CSV file or run the data generator to populate energy data.")
