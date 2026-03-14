import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data.csv")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
    
st.title("Linear Regression Streamlit")

x_input = st.slider("Enter X value: ", 0.0, 50.0, 25.0)
input_df = pd.DataFrame({"X": [x_input]})
prediction = model.predict(input_df)[0]

st.write(f"Prediction: {prediction:.2f}")

fig, ax = plt.subplots()
ax.scatter(df["X"], df["Y"], label="Data")

x_range = np.linspace(0, 50, 100)
x_df = pd.DataFrame({"X": x_range})
y_pred = model.predict(x_df)

ax.plot(x_range, y_pred, label="Regression Line")
ax.scatter([x_input], [prediction], s=100, label="Prediction")

ax.legend()
st.pyplot(fig)