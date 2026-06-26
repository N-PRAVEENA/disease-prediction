import pickle

# Load saved components
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# Get user input
new_input = input("Enter your health issue: ")
new_vector = vectorizer.transform([new_input])

# Predict
prediction = model.predict(new_vector)
predicted_disease = le.inverse_transform(prediction)[0]

print("Predicted Disease:", predicted_disease)