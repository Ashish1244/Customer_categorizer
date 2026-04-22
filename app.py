import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import joblib # or 'import pickle' if that's what you used to save the model

# --- LOAD MODEL ---
# Note: Ensure you have saved your 'best_model' from the notebook using:
# joblib.dump(best_model, 'catboost_model.pkl')
@st.cache_resource
def load_model():
    try:
        return joblib.load('catboost_model.pkl')
    except:
        st.error("Model file 'catboost_model.pkl' not found. Please export your model from the notebook.")
        return None

model = load_model()

# --- PREPROCESSING FUNCTION ---
def preprocess_input(data):
    df = pd.DataFrame([data])
    
    # Feature Extraction (Based on your EDA/Feature Engineering notebooks)
    df['Age'] = 2022 - df['Year_Birth']
    
    # Encoding Education
    edu_map = {"Basic": 0, "2n Cycle": 1, "Graduation": 2, "Master": 3, "PhD": 4}
    df["Education"] = df["Education"].map(edu_map)
    
    # Encoding Marital Status
    marital_map = {"Married": 1, "Together": 1, "Absurd": 0, "Widow": 0, 
                   "YOLO": 0, "Divorced": 0, "Single": 0, "Alone": 0}
    df['Marital Status'] = df['Marital_Status'].map(marital_map)
    
    # Children and Family
    df['Children'] = df['Kidhome'] + df['Teenhome']
    df["Parental Status"] = np.where(df["Children"] > 0, 1, 0)
    
    # Spending and Promos
    df['Total_Spending'] = df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] + \
                           df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
    
    df["Total Promo"] = df["AcceptedCmp1"] + df["AcceptedCmp2"] + df["AcceptedCmp3"] + \
                        df["AcceptedCmp4"] + df["AcceptedCmp5"]
    
    # Tenure
    dt_customer = pd.to_datetime(df['Dt_Customer'])
    df['Days_as_Customer'] = (pd.Timestamp(datetime.today()) - dt_customer).dt.days
    
    # Select final features used in the Classification notebook
    features = ["Age", "Education", "Marital Status", "Parental Status", "Children", 
                "Income", "Total_Spending", "Days_as_Customer", "Recency", "Wines", 
                "Fruits", "Meat", "Fish", "Sweets", "Gold", "Web", "Catalog", 
                "Store", "Discount Purchases", "Total Promo", "NumWebVisitsMonth"]
    
    # Rename columns to match the model's expected input
    df = df.rename(columns={
        "MntWines": "Wines", "MntFruits": "Fruits", "MntMeatProducts": "Meat",
        "MntFishProducts": "Fish", "MntSweetProducts": "Sweets", "MntGoldProds": "Gold",
        "NumWebPurchases": "Web", "NumCatalogPurchases": "Catalog",
        "NumStorePurchases": "Store", "NumDealsPurchases": "Discount Purchases"
    })
    
    return df[features]

# --- STREAMLIT UI ---
st.title("👥 Customer Personality Predictor")
st.markdown("Enter customer details below to predict their segment cluster.")

with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        year_birth = st.number_input("Year of Birth", 1900, 2022, 1980)
        education = st.selectbox("Education", ["Basic", "2n Cycle", "Graduation", "Master", "PhD"])
        income = st.number_input("Annual Income", 0, 700000, 50000)
        dt_cust = st.date_input("Enrollment Date", datetime(2013, 1, 1))
        
    with col2:
        marital = st.selectbox("Marital Status", ["Married", "Together", "Single", "Divorced", "Widow"])
        kidhome = st.number_input("Kids at Home", 0, 5, 0)
        teenhome = st.number_input("Teens at Home", 0, 5, 0)
        recency = st.number_input("Days since last purchase", 0, 100, 50)

    with col3:
        deals = st.number_input("Discount Purchases", 0, 20, 2)
        web_pur = st.number_input("Web Purchases", 0, 20, 5)
        cat_pur = st.number_input("Catalog Purchases", 0, 20, 2)
        store_pur = st.number_input("Store Purchases", 0, 20, 5)
        web_visits = st.number_input("Web Visits/Month", 0, 20, 5)

    st.subheader("Amount Spent on Products")
    c1, c2, c3 = st.columns(3)
    mnt_wines = c1.number_input("Wines", 0, 2000, 100)
    mnt_fruits = c2.number_input("Fruits", 0, 2000, 20)
    mnt_meat = c3.number_input("Meat", 0, 2000, 50)
    mnt_fish = c1.number_input("Fish", 0, 2000, 20)
    mnt_sweet = c2.number_input("Sweets", 0, 2000, 20)
    mnt_gold = c3.number_input("Gold", 0, 2000, 20)

    st.subheader("Campaign Participation")
    camp_cols = st.columns(5)
    cmp1 = camp_cols[0].checkbox("Cmp 1")
    cmp2 = camp_cols[1].checkbox("Cmp 2")
    cmp3 = camp_cols[2].checkbox("Cmp 3")
    cmp4 = camp_cols[3].checkbox("Cmp 4")
    cmp5 = camp_cols[4].checkbox("Cmp 5")

    submit = st.form_submit_button("Predict Personality Cluster")

if submit and model:
    input_data = {
        "Year_Birth": year_birth, "Education": education, "Marital_Status": marital,
        "Income": income, "Kidhome": kidhome, "Teenhome": teenhome,
        "Dt_Customer": dt_cust, "Recency": recency, "MntWines": mnt_wines,
        "MntFruits": mnt_fruits, "MntMeatProducts": mnt_meat, "MntFishProducts": mnt_fish,
        "MntSweetProducts": mnt_sweet, "MntGoldProds": mnt_gold, "NumDealsPurchases": deals,
        "NumWebPurchases": web_pur, "NumCatalogPurchases": cat_pur, "NumStorePurchases": store_pur,
        "NumWebVisitsMonth": web_visits, "AcceptedCmp1": int(cmp1), "AcceptedCmp2": int(cmp2),
        "AcceptedCmp3": int(cmp3), "AcceptedCmp4": int(cmp4), "AcceptedCmp5": int(cmp5)
    }
    
    processed_df = preprocess_input(input_data)
    prediction = model.predict(processed_df)
    
    st.success(f"The Predicted Cluster for this Customer is: **Cluster {prediction[0]}**")