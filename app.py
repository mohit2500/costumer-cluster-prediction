import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.cluster import KMeans
import streamlit as st


kmeans = joblib.load("Model.pkl")
df = pd.read_csv("Mall_customers.csv")
X = df[["annual income(k$)","spending score(1-100)"]]
X_array = X.values


st.set_page_config(page_title= "customer cluster prediction",layout ="centered")
st.title("customer cluster prediction")
st.write("enter the customer annual income")

#input
annual_income = st.number_input("annual income ",min_value= 0,max_value=400,value = 50)
spending_score = st.slider("spending score between 1-100",1,100,20)


#predict the cluster
if st.button("predict Cluster"):
    input_data = np.array([[annual_income,spending_score]])
    cluster = kmeans.predict(input_data)[0]
    st.success(f"predicted cluster is : {cluster}")