import streamlit as st
import pdfplumber
import docx
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Resume Screening App", layout="wide")

st.title("📄 Resume Screening App")
st.markdown("Upload resumes and rank them based on your job description or skills.")

# Text extractor for PDF/DOCX
def extract_text(file):
    text = ""
    if file.type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text()
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + " "
    return text.strip()

# Scoring resumes using TF-IDF
def score_resumes(resume_texts, job_keywords):
    documents = [job_keywords] + resume_texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return scores

# Upload resumes
uploaded_files = st.file_uploader("Upload Resumes (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

# Job description / keyword input
job_keywords = st.text_area("Enter Required Skills (comma-separated)", "python, machine learning, pandas, sql")

if uploaded_files and job_keywords:
    resume_texts = []
    valid_files = []

    st.info("Parsing resumes...")

    for file in uploaded_files:
        text = extract_text(file)
        if text:
            resume_texts.append(text.lower())
            valid_files.append(file)
        else:
            st.warning(f"{file.name} could not be parsed. Skipped.")

    if resume_texts:
        scores = score_resumes(resume_texts, job_keywords.lower())

        results = [{"Filename": f.name, "Score (%)": round(score * 100, 2)}
                   for f, score in zip(valid_files, scores)]
        results = sorted(results, key=lambda x: x['Score (%)'], reverse=True)

        st.success("Ranking Complete ✅")
        st.subheader("📊 Resume Match Results")
        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True)

        # Download CSV
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results as CSV", csv, "resume_ranking.csv", "text/csv")
    else:
        st.error("No valid text extracted from uploaded resumes.")
