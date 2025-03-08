import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64
from time import sleep   
import matplotlib.pyplot as plt
import altair as alt  # Explicit Altair import for better type control

# Install dependencies: pip install matplotlib streamlit PyPDF2 pandas numpy scikit-learn altair

# Custom CSS for styling
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stProgress > div > div > div > div {background-color: #4CAF50;}
    .st-bb {background-color: white;}
    .st-at {background-color: #4CAF50;}
    </style>
    """, unsafe_allow_html=True)

def extract_text_from_pdf(file):
    try:
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
        return text
    except Exception as e:
        st.error(f"Error reading {file.name}: {str(e)}")
        return ""

def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    return (cosine_similarities * 100).round(2)

# Main App
st.title("📄 AI Resume Screening & Candidate Ranking System")
st.markdown("---")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    min_score = st.slider("Minimum Match Score (%)", 0, 100, 50)
    top_n = st.slider("Show Top N Candidates", 1, 20, 5)
    show_details = st.checkbox("Show Resume Details", True)

# Main Content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Job Description")
    job_description = st.text_area("Paste job description here", height=200)

with col2:
    st.subheader("📤 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Select PDF resumes",
        type=["pdf"], 
        accept_multiple_files=True,
        label_visibility="visible"
    )

if uploaded_files and job_description:
    with st.spinner("🔍 Analyzing resumes..."):
        progress_bar = st.progress(0)
        resumes = []
        
        for i, file in enumerate(uploaded_files):
            progress = int((i + 1) / len(uploaded_files) * 100)
            progress_bar.progress(progress)
            text = extract_text_from_pdf(file)
            if text:
                resumes.append(text)
            sleep(0.1)

    if resumes:
        scores = rank_resumes(job_description, resumes)
        
        results = pd.DataFrame({
            "Resume": [file.name for file in uploaded_files],
            "Match Score (%)": scores.astype(float),  # Ensure numeric type
            "Status": np.where(scores >= min_score, "✅ Qualified", "❌ Rejected")
        })
        
        filtered_results = results[results["Match Score (%)"] >= min_score]
        sorted_results = filtered_results.sort_values(by="Match Score (%)", ascending=False).head(top_n)
        
        st.markdown("---")
        st.subheader("📊 Results Overview")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Resumes", len(uploaded_files))
        m2.metric("Qualified Candidates", len(filtered_results))
        m3.metric("Top Score", f"{sorted_results['Match Score (%)'].max():.1f}%")
        
        tab1, tab2 = st.tabs(["📈 Visualization", "📋 Detailed Results"])
        
        with tab1:
            # Fixed Altair chart with explicit data types
            chart = alt.Chart(sorted_results).mark_bar().encode(
                x=alt.X('Resume:N', title='Resume', sort='-y'),
                y=alt.Y('Match Score (%):Q', title='Match Score (%)', scale=alt.Scale(domain=[0, 100])),
                color=alt.value('#4CAF50')
            ).properties(
                width=600,
                height=400
            )
            st.altair_chart(chart, use_container_width=True)
        
        with tab2:
            st.dataframe(
                sorted_results.style.background_gradient(
                    subset=["Match Score (%)"],
                    cmap=plt.cm.Greens,
                    vmin=0,
                    vmax=100
                ), 
                use_container_width=True
            )
            
            csv = sorted_results.to_csv(index=False).encode()
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name="ranking_results.csv",
                mime="text/csv"
            )
        
        if show_details:
            st.markdown("---")
            st.subheader("📄 Resume Details")
            for idx, row in sorted_results.iterrows():
                with st.expander(f"{row['Resume']} - {row['Match Score (%)']}%"):
                    st.write(resumes[idx])
    else:
        st.warning("⚠️ No valid text extracted from uploaded PDFs")

else:
    st.info("👋 Please upload resumes and enter a job description to get started")