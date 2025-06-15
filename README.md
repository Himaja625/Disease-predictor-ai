# AI Disease Predictor & Medical Assistant

This is a smart web application that predicts diseases based on selected symptoms using a trained machine learning model. It also includes an AI chatbot to provide general medical and wellness advice, and a doctor directory for location-based consultation.

---

## Problem Statement

People often struggle to identify potential diseases based on symptoms and may delay seeking medical help due to uncertainty or lack of initial guidance.

---

## Proposed Solution

A Disease predictor web app that:
- Predicts the most probable disease based on symptoms
- Provides risk level and remedy suggestions
- Offers basic chatbot health guidance
- Suggests doctors based on location and specialization
- Visualizes disease trends over time

---

## Technologies Used

- **Python**
- **Streamlit** (Web UI)
- **Scikit-learn** (ML Model)
- **Transformers** (Hugging Face AI Chatbot)
- **Matplotlib / Seaborn / Plotly** (Charts)
- **Pandas / NumPy** (Data Handling)
- **FPDF** (PDF Report Generation)
- **SMTP** (Emailing Reports)

---

## Machine Learning Approach

- **Algorithm Used**: Multinomial Naive Bayes  
- **Trained On**: Symptom-to-disease dataset (Training.csv)  
- **Features**: 132 symptom indicators  
- **Labels**: 41 diseases

Model and encoders are stored as `.pkl` files for reuse.

---

## Deployment

- Built as a **Streamlit Web App**
- Can be deployed using **Streamlit Community Cloud**, **GitHub Pages**, or locally
- Requires Python environment and installed libraries

---

## Features

- 🔮 **Disease Prediction** based on selected symptoms  
- 📈 **Interactive Charts** for disease trends  
- 🤖 **AI Chatbot Assistant** (offline model from Hugging Face)  
- 🩺 **Find a Doctor** using location and specialty filters  
- 📄 **PDF Report Generation** + ✉️ Email support


## Conclusion

This project demonstrates how machine learning and natural language processing can be combined to build healthcare tools that empower users to get informed, preliminary health insights and access support instantly.

---

## Future Scope

- Integrate real-time data and API-based doctor database  
- Improve chatbot to use advanced models (GPT, Mistral, etc.)  
- Add user authentication and health history tracking  
- Deploy to mobile platforms

---

## Developed by

**Naga Himaja**  

---

## 📚 References

- [Streamlit Docs](https://docs.streamlit.io/)
- [Scikit-learn Docs](https://scikit-learn.org/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Matplotlib / Plotly](https://plotly.com/python/)
- [Python Email (SMTP)](https://docs.python.org/3/library/email.html)

---

> *Disclaimer: This application is for educational use only and should not be used for real medical decisions.*
