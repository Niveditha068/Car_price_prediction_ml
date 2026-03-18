import streamlit as st
import pandas as pd
import joblib

# ======================================
# Page Config
# ======================================
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# ======================================
# Load Model & Data
# ======================================
model = joblib.load("final_random_forest_model.pkl")
df = pd.read_csv("cars24_20221210.csv")

# ======================================
# Title & Header
# ======================================
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>🚗 Used Car Price Predictor</h1>
    <p style='text-align: center;'>Estimate your car resale value instantly</p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ======================================
# Dropdown Values
# ======================================
brands = sorted(df["make"].dropna().unique())
fueltypes = sorted(df["fueltype"].dropna().unique())
bodytypes = sorted(df["bodytype"].dropna().unique())
states = sorted(df["registrationstate"].dropna().unique())

# ======================================
# Layout
# ======================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🚘 Car Details")

    make = st.selectbox("Brand", brands, key="make")
    fueltype = st.selectbox("Fuel Type", fueltypes, key="fuel")
    bodytype = st.selectbox("Body Type", bodytypes, key="body")
    transmission = st.selectbox(
        "Transmission", ["Manual", "Automatic"], key="transmission"
    )

with col2:
    st.subheader("📊 Usage Details")

    kilometerdriven = st.number_input(
        "Kilometers Driven", min_value=0, max_value=300000, value=30000
    )
    year = st.number_input(
    "Manufacturing Year", min_value=2000, max_value=2026, value=2020
    )
    ownernumber = st.selectbox(
        "Owner Number", [1, 2, 3, 4, 5], key="owner"
    )
    isc24assured = st.selectbox(
        "C24 Assured", ["Yes", "No"], key="assured"
    )

    registrationstate = st.selectbox(
        "Registration State", states, key="state"
    )

    # Filter cities based on state
    filtered_cities = sorted(
        df[df["registrationstate"] == registrationstate]["city"]
        .dropna()
        .unique()
    )

    city = st.selectbox("City", filtered_cities, key="city")

# ======================================
# Predict Button
# ======================================
st.markdown("<br>", unsafe_allow_html=True)

if st.button("💰 Predict Price", use_container_width=True):

    # Convert Yes/No to 1/0
    isc24_value = 1 if isc24assured == "Yes" else 0

    # Create input dataframe
    input_df = pd.DataFrame({
    'kilometerdriven': [kilometerdriven],
    'ownernumber': [ownernumber],
    'registrationstate': [registrationstate],
    'isc24assured': [isc24_value],
    'bodytype': [bodytype],
    'city': [city],
    'make': [make],
    'fueltype': [fueltype],
    'transmission': [transmission],   
    'year': [year],                 
})

    # Prediction
    predicted_price = model.predict(input_df)[0]

    # ======================================
    # Output UI
    # ======================================
    st.markdown("<hr>", unsafe_allow_html=True)

    st.success(f"💰 Estimated Car Price: ₹ {predicted_price:,.2f}")

    # Price Range
    lower = predicted_price * 0.9
    upper = predicted_price * 1.1

    st.info(f"📊 Expected Price Range: ₹ {lower:,.0f} - ₹ {upper:,.0f}")

    # Fun Visual Indicator
    st.progress(min(int(predicted_price / 1000000 * 100), 100))

# ======================================
# Footer
# ======================================
st.markdown(
    """
    <hr>
    <p style='text-align: center; font-size: 12px;'>
    Built with ❤️ using Streamlit | Data Science Project
    </p>
    """,
    unsafe_allow_html=True
)