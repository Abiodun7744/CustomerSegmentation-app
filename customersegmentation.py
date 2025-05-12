import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

@st.cache_data
def train_model():
    df = pd.read_csv("CustomerSegmentation.csv")
    df.drop(columns="Unnamed: 0", inplace=True)
    df['Gender'] = LabelEncoder().fit_transform(df['Gender'])

    features = ['CustomerID', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender', 'Years as Customer']
    X = df[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # Supervised learning
    X = df[features]
    y = df['Cluster']
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    return model, scaler

model, scaler = train_model()

st.title("🧠 Customer Segmentation Prediction App")
st.write("Enter customer details below to predict the customer cluster.")

# Form Inputs
customer_id = st.number_input("Customer ID", min_value=1, value=1)
age = st.number_input("Age", 18, 100, 30)
income = st.number_input("Annual Income (k$)", 10, 150, 50)
score = st.number_input("Spending Score (1-100)", 1, 100, 50)
gender = st.selectbox("Gender", ["Male", "Female"])
years = st.number_input("Years as Customer", 0, 50, 3)

# Encode Gender
gender_encoded = 1 if gender == "Male" else 0

# Prediction
if st.button("Predict Cluster"):
    new_input = pd.DataFrame([{
        'CustomerID': customer_id,
        'Age': age,
        'Annual Income (k$)': income,
        'Spending Score (1-100)': score,
        'Gender': gender_encoded,
        'Years as Customer': years
    }])
    prediction = model.predict(new_input)
    st.success(f"Predicted Cluster: {prediction[0]}")
    st.write(new_input.assign(Predicted_Cluster=prediction))
