# 🧠 Disease Prediction from Symptoms

A Machine Learning-based web application that predicts possible diseases from user-entered symptoms using Natural Language Processing (TF-IDF) and a trained classification model.

Built with **Python + Scikit-learn + Streamlit**, this project demonstrates an end-to-end ML pipeline from text processing to real-time prediction.

---

## 🚀 Project Objective

Many people experience symptoms but are unsure about the possible disease.  
This project helps users get a **quick preliminary prediction** based on symptoms entered as text.

---

## 🧠 Key Features

- 🔍 Predict disease from free-text symptoms  
- ⚡ Real-time ML prediction  
- 🌐 Interactive Streamlit web app  
- 🧾 TF-IDF based text feature extraction  
- 🧠 Label Encoder for readable output  

---

## 🛠️ Tech Stack

- Python 🐍  
- Scikit-learn 🤖  
- Streamlit 🌐  
- TF-IDF Vectorizer 📊  
- Pickle (.pkl) for model storage  

---

## 📁 Project Structure

- app.py → Streamlit web application  
- test.py → CLI testing script  
- model.pkl → Trained ML model  
- vectorizer.pkl → TF-IDF vectorizer  
- label_encoder.pkl → Encodes disease labels  
- .gitattributes → Git configuration  

---

## ⚙️ How It Works

1. User enters symptoms as text  
2. TF-IDF converts text into numerical features  
3. ML model predicts disease  
4. Label Encoder converts prediction into readable name  
5. Result is displayed in UI  

---

## ▶️ How to Run Locally

### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/disease-prediction.git
cd disease-prediction
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate:

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run Application

```bash
streamlit run app.py
```

---

## 🧪 CLI Test

```bash
python test.py
```

---

## 📊 Example

**Input:**
```
fever, headache, body pain
```

**Output:**
```
Predicted Disease: Viral Infection (example)
```

---

## ⚠️ Important Note

- This is an **educational prototype**
- Not intended for medical diagnosis
- Accuracy depends on training dataset
- Pickle files must be from trusted source only

---

## 📈 Future Enhancements

- Improve dataset size & accuracy  
- Add Deep Learning models (LSTM/Transformer)  
- Deploy on cloud (Streamlit Cloud / AWS)  
- Add doctor recommendation system  
- Improve UI/UX design  

---

## 👨‍💻 Author

Praveena N  
B.Tech IT / AI & Data Science  
J.J. College of Engineering and Technology  

---

## ⭐ Support

If you like this project:
- ⭐ Star the repository  
- 🍴 Fork and improve it  
- 📢 Share with others  

---

## 📌 License

This project is for educational purposes only.
