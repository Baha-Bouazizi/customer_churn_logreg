import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go

with open("modele_churn_balanced.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler_balanced.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Customer Churn Prediction")
st.write("Enter the customer information to predict whether they will churn or not.")

tenure = st.number_input("Subscription Duration (tenure)", min_value=0.0, max_value=100.0, value=33.0)
age = st.number_input("Customer Age", min_value=18.0, max_value=100.0, value=33.0)
address = st.number_input("Years at Current Address", min_value=0.0, max_value=50.0, value=12.0)
income = st.number_input("Customer Income", min_value=0.0, max_value=1000.0, value=33.0)
ed = st.number_input("Education Level", min_value=0.0, max_value=10.0, value=2.0)
employ = st.number_input("Years of Employment", min_value=0.0, max_value=50.0, value=0.0)

if st.button("Predict Churn"):
    features = np.array([[tenure, age, address, income, ed, employ]])
    features_scaled = scaler.transform(features)
    
    proba_churn = model.predict_proba(features_scaled)[0][1]
    proba_stay = 1 - proba_churn
    
    if proba_churn >= 0.5:
        st.warning(f"The customer is likely to churn with a probability of {proba_churn:.2f}")
    else:
        st.success(f"The customer is likely to stay with a probability of {proba_stay:.2f}")
    
    fig = go.Figure(go.Bar(
        x=["Stay", "Churn"],
        y=[proba_stay, proba_churn],
        marker_color=["green", "red"]
    ))
    fig.update_layout(
        title="Probability of Stay vs Churn",
        yaxis=dict(range=[0,1]),
        xaxis_title="Customer Status",
        yaxis_title="Probability"
    )
    st.plotly_chart(fig)

