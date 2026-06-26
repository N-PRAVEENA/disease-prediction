import streamlit as st
import pickle

# Load models
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

st.title("🩺 Disease Prediction from Symptoms")
st.markdown("Enter your health symptoms below (e.g., *fever, headache, fatigue*)")

user_input = st.text_input("Symptoms")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("❗ Please enter some symptoms.")
    else:
        vector = vectorizer.transform([user_input])
        prediction = model.predict(vector)
        predicted_disease = le.inverse_transform(prediction)[0]
        st.success(f"✅ Predicted Disease: *{predicted_disease}*")