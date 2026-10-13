import streamlit as st
import pandas as pd
import numpy as np
import joblib

CURRENT_YEAR = 2026   # same year used to create car_age during cleaning

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="wide")


# ---------- Load saved files ----------
@st.cache_resource
def load_model():
    return joblib.load("car_price_model.pkl"), joblib.load("model_meta.pkl")


@st.cache_data
def load_tables():
    fi = pd.read_csv("data/feature_importance.csv", index_col=0)
    res = pd.read_csv("data/model_results.csv")
    return fi, res


model, meta = load_model()
importance, results = load_tables()

st.title("🚗 Used Car Price Predictor")
st.caption("Machine learning project: predicts the resale price of a used car in India (CarDekho data).")

tab1, tab2, tab3 = st.tabs(["Predict price", "What affects price", "Model comparison"])


# ---------- Tab 1: prediction form ----------
with tab1:
    st.subheader("Enter the car details")
    c1, c2, c3 = st.columns(3)

    with c1:
        brand = st.selectbox("Brand", meta["brands"])
        year = st.number_input("Manufacturing year", min_value=1995, max_value=CURRENT_YEAR, value=2018)
        km = st.number_input("Kilometres driven", min_value=0, max_value=500000,
                             value=meta["defaults"]["km_driven"], step=1000)
    with c2:
        fuel = st.selectbox("Fuel type", meta["fuels"])
        transmission = st.selectbox("Transmission", meta["transmissions"])
        owner = st.selectbox("Owner", meta["owners"])
    with c3:
        seller = st.selectbox("Seller type", meta["seller_types"])
        seats_options = meta["seats"]
        seats = st.selectbox("Seats", seats_options,
                             index=seats_options.index(5.0) if 5.0 in seats_options else 0)
        mileage = st.number_input("Mileage (kmpl)", min_value=0.0, max_value=50.0,
                                  value=float(meta["defaults"]["mileage"]), step=0.5)

    c4, c5 = st.columns(2)
    with c4:
        engine = st.number_input("Engine (CC)", min_value=500, max_value=6000,
                                 value=int(meta["defaults"]["engine"]), step=50)
    with c5:
        power = st.number_input("Max power (bhp)", min_value=30.0, max_value=500.0,
                                value=float(meta["defaults"]["max_power"]), step=1.0)

    if st.button("Predict price", type="primary"):
        row = pd.DataFrame([{
            "km_driven": km, "mileage": mileage, "engine": engine, "max_power": power,
            "seats": seats, "car_age": CURRENT_YEAR - year,
            "fuel": fuel, "seller_type": seller, "transmission": transmission,
            "owner": owner, "brand": brand,
        }])

        row = pd.get_dummies(row).reindex(columns=meta["columns"], fill_value=0)
        price = float(np.expm1(model.predict(row))[0])
        mae = float(results.iloc[0]["MAE (₹)"])

        st.success(f"Estimated resale price: ₹{price:,.0f}  (about {price / 100000:.2f} lakh)")
        st.info(f"On test data the model's typical error was about ₹{mae:,.0f}, "
                "so treat this as an estimate, not an exact price.")


# ---------- Tab 2: feature importance ----------
with tab2:
    st.subheader("Which factors affect the price the most?")
    st.bar_chart(importance["importance"].sort_values(ascending=False))
    st.caption("Higher means the model relied on that feature more when predicting price.")


# ---------- Tab 3: model comparison ----------
with tab3:
    st.subheader("Model comparison on unseen test data")
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.bar_chart(results.set_index("Model")["R²"])
    st.caption("R² closer to 1 is better. Lower MAE and RMSE are better.")