import streamlit as st
st.set_page_config(page_title="Disease Predictor", layout="wide")

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json
from streamlit_lottie import st_lottie
from fpdf import FPDF
from datetime import datetime
import plotly.express as px
import smtplib
from email.message import EmailMessage
from transformers import pipeline

# Load chatbot pipeline (free and clean)
@st.cache_resource
def load_chatbot():
    return pipeline("text2text-generation", model="declare-lab/flan-alpaca-base")

chatbot = load_chatbot()

# Load ML model, encoder, and features
model = joblib.load("disease_model.pkl")
le = joblib.load("label_encoder.pkl")
feature_names = joblib.load("feature_names.pkl")

# Load dataset for charts
df = pd.read_csv("Training.csv")
df.drop(columns=["Unnamed: 133", "fluid_overload.1"], errors='ignore', inplace=True)

# Style
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1588776814546-dced00a5fb49?auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    h1, h2, h3, h4, h5 {
        color: #ffffff;
        text-shadow: 1px 1px 3px #000000;
    }
    .stButton>button {
        background-color: #0072B5;
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Load animation
def load_lottiefile(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

lottie_health = load_lottiefile("health_animation.json")

# Sidebar
page = st.sidebar.radio("📄 Go to", ["Disease Prediction", "Charts", "Doctors", "AI Chatbot 🤖", "About"])

# Disease Prediction
if page == "Disease Prediction":
    st.markdown("<h1 style='text-align: center;'>🩺 Disease Prediction App</h1>", unsafe_allow_html=True)
    st_lottie(lottie_health, height=250, key="health")
    st.subheader("👨‍⚕️ Select Your Symptoms")
    symptoms = feature_names
    selected_symptoms = st.multiselect("Choose symptoms from the list", symptoms)
    input_data = [1 if symptom in selected_symptoms else 0 for symptom in symptoms]

    if st.sidebar.button("🔮 Predict Disease"):
        if sum(input_data) == 0:
            st.warning("⚠️ Please select at least one symptom.")
        else:
            prediction = model.predict([input_data])
            predicted_disease = le.inverse_transform(prediction)[0]
            st.success(f"🌟 Based on your symptoms, the predicted disease is: **{predicted_disease}**")

            with open("disease_info.json", "r", encoding="utf-8") as f:
                disease_info = json.load(f)

            if predicted_disease in disease_info:
                remedy = disease_info[predicted_disease]["remedy"]
                risk = disease_info[predicted_disease]["risk"]
                st.markdown(f"**🩹 Suggested Remedy:** {remedy}")
                st.markdown(f"**📉 Risk Level:** `{risk}`")
            else:
                st.warning("No remedy information found for this disease.")
                remedy = "Not available"
                risk = "Unknown"

            def generate_pdf(disease, symptoms, remedy, risk):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", "B", 16)
                pdf.cell(0, 10, "Disease Prediction Report", ln=True, align="C")
                pdf.ln(10)
                pdf.set_font("Arial", "", 12)
                pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
                pdf.cell(0, 10, f"Predicted Disease: {disease}", ln=True)
                pdf.cell(0, 10, f"Symptoms: {', '.join(symptoms)}", ln=True)
                pdf.cell(0, 10, f"Risk Level: {risk}", ln=True)
                pdf.multi_cell(0, 10, f"Suggested Remedy: {remedy}")
                filename = "disease_report.pdf"
                pdf.output(filename)
                return filename

            if st.button("📄 Download PDF Report"):
                report_file = generate_pdf(predicted_disease, selected_symptoms, remedy, risk)
                with open(report_file, "rb") as f:
                    st.download_button("Download Your Report", f, file_name=report_file)

# Charts
elif page == "Charts":
    st.markdown("<h2 style='color: white;'>📊 Disease Trends Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: white;'>Explore the patterns and monthly trends of various diseases reported in the dataset.</p>", unsafe_allow_html=True)

    # Generate date column if not already
    if 'Month' not in df.columns:
        max_months = min(len(df), 240)
        df = df.iloc[:max_months].copy()
        df["Month"] = pd.date_range(start='2015-01-01', periods=len(df), freq='M')
    if 'prognosis' not in df.columns:
        df['prognosis'] = 'Unknown'

    # Sidebar Filters
    with st.sidebar:
        st.subheader("📅 Filter Data")
        selected_diseases = st.multiselect("Filter by Disease", sorted(df["prognosis"].unique()), default=sorted(df["prognosis"].unique())[:5])
        year_range = st.slider("Select Year Range", 2015, 2030, (2016, 2020))

    # Filtered data
    filtered_df = df[df["prognosis"].isin(selected_diseases)]
    filtered_df["Year"] = filtered_df["Month"].dt.year
    filtered_df = filtered_df[(filtered_df["Year"] >= year_range[0]) & (filtered_df["Year"] <= year_range[1])]

    # Group and plot
    chart_data = filtered_df[["Month", "prognosis"]].copy()
    chart_data["Cases"] = 1
    time_trend = chart_data.groupby([pd.Grouper(key='Month', freq='M'), 'prognosis']).count().reset_index()

    fig = px.line(
        time_trend,
        x="Month",
        y="Cases",
        color="prognosis",
        title="🗓️ Monthly Disease Case Trends",
        template="plotly_white",
        markers=True
    )
    fig.update_layout(
        title_font=dict(size=22, color="darkblue"),
        xaxis_title="Month",
        yaxis_title="Number of Cases",
        legend_title="Disease",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(255,255,255,0.1)',
        font=dict(color="black")
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary
    st.markdown("---")
    st.subheader("📌 Summary")
    st.markdown(f"""
    - Total Diseases Shown: `{len(selected_diseases)}`
    - Date Range: `{year_range[0]} to {year_range[1]}`
    - Total Records Displayed: `{len(filtered_df)}` rows
    """)

# Doctors
elif page == "Doctors":
    st.markdown("<h2>🩺 Find a Doctor</h2>", unsafe_allow_html=True)
    doc_df = pd.read_csv("doctor_contacts.csv")
    location_col = "Location"
    specialty_col = "Specialization"
    if location_col not in doc_df.columns or specialty_col not in doc_df.columns:
        st.error("❌ Required columns not found in doctor_contacts.csv.")
    else:
        location = st.selectbox("📍 Select Location", sorted(doc_df[location_col].dropna().unique()))
        specialty = st.selectbox("🩻 Select Specialty", sorted(doc_df[specialty_col].dropna().unique()))
        filtered_docs = doc_df[(doc_df[location_col] == location) & (doc_df[specialty_col] == specialty)]
        if not filtered_docs.empty:
            for _, row in filtered_docs.iterrows():
                st.markdown(f"""
                    <div style='background-color:#e3f2fd;padding:10px;border-radius:10px;margin:10px 0'>
                        <strong>{row['Name']}</strong><br>
                        📞 {row['Phone']}<br>
                        ✉️ {row['Email']}<br>
                        🏥 {row[location_col]}
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No doctors found matching your filters.")

# Chatbot
elif page == "AI Chatbot 🤖":
    st.markdown("<h2>🤖 Doctor AI Assistant</h2>", unsafe_allow_html=True)
    st.warning("⚠️ This chatbot is for general guidance only. Always consult a certified doctor for medical advice.")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💊 Symptoms Advice"):
            user_input = "What should I do if I have symptoms of common illness?"
        else:
            user_input = None
    with col2:
        if st.button("🥗 Diet Suggestions"):
            user_input = "What is a healthy diet for common illnesses?"
    with col3:
        if st.button("🧘 Lifestyle Tips"):
            user_input = "Give me some lifestyle and wellness advice."

    typed_input = st.chat_input("Ask your question here...")
    if typed_input:
        user_input = typed_input

    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Doctor AI is thinking..."):
            prompt = f"Question: {user_input}\nAnswer:"
            result = chatbot(prompt, max_length=200)[0]
            response = result["generated_text"].replace(prompt, "").strip()
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

elif page == "About":
    st.markdown("<h2>💡 About This Project</h2>", unsafe_allow_html=True)
    st.markdown("""
    Welcome to the **AI Disease Predictor & Medical Assistant**—a smart healthcare tool developed using **Machine Learning** and **Natural Language Processing** to assist users in identifying potential diseases based on symptoms.

    This web application is built as part of the **MS-Edunet Internship Program**, combining core concepts of **AI**, **ML**, and **Data Science** with an intuitive and beautiful UI using **Streamlit**.

    #### ✨ What This App Offers:
    - Disease prediction using a trained machine learning model  
    - Trend visualization with interactive charts  
    - Doctor lookup filtered by location and specialization  
    - AI-powered health chatbot for basic medical guidance  
    - Auto-generated PDF reports and email delivery

    ---
    
    #### 👩‍💻 Developed by:
    **Naga Himaja**  
    3rd Year B.Tech | Data Science 

    ---
    
    _This app is intended for educational purposes only and does not replace professional medical advice._
    """)