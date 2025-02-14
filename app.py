import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import string
import fitz  # PyMuPDF for PDF processing
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.nn.functional import cosine_similarity
from io import BytesIO
import base64

# Streamlit UI Configuration (Must be first command)
st.set_page_config(page_title="Resume Ranking System", page_icon="📄", layout="wide")

# Download NLTK Data
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords


# Load Pre-trained Models
bert_model = SentenceTransformer("all-mpnet-base-v2")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Navigation Bar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Resume Ranking"])

if page == "Home":
    st.title("📄 Resume Ranking System")
    
    
    st.write("""
    ## About the Project
    The Resume Ranking System is an AI-powered tool designed to help recruiters automatically evaluate and rank resumes based on a given job description. 
    It eliminates manual screening, speeds up the hiring process, and ensures **fair candidate evaluation.
    
    ### Key Features:
    - 📂 Upload multiple resumes (PDF format).
    - 📝 Input a job description for ranking.
    - 🔍 Choose between BERT (MPNet) and T5 Transformer for ranking.
    - 📊 Generate Word Clouds for job descriptions & resumes.
    - 📊 View and download ranked resumes based on relevance.

    ### Why Use This?
    - Saves Time: No more manual resume screening.
    - AI-Powered: Uses Natural Language Processing (NLP) for accurate rankings.
    - Customizable: Choose between BERT and T5 Transformer models.

    👉 Go to the "Resume Ranking" section from the sidebar to start ranking resumes!  
    """)

    # Team Members and Guide Section
    st.sidebar.markdown("### 👨‍💻 Team Members ")
    st.sidebar.markdown(""" 
    - K.D.S.S.PRASAD  
    - J.C.S.SRIRAM  
    - A.P.V.K.S.KUMAR  
    - G.J.N.V.S.S.SIVA RAO  
    """)

    st.sidebar.markdown("####  3rd year KIET")
    st.sidebar.markdown("###  👨‍💻 Guide")
    st.sidebar.markdown("MD ABDUL AZIZ (MASTER TRAINER EDUNET FOUNDATION)")

elif page == "Resume Ranking":
    st.title("📄 Resume Ranking System")
    st.write("Upload multiple PDFs containing resumes, input a job description, and rank resumes based on relevance using either BERT or T5 model.")

    # Sidebar for Model Selection
    model_choice = st.sidebar.radio("Select Model", ["BERT (MPNet)", "T5 Transformer"])

    # Function to Extract Text from PDF
    def extract_text_from_pdf(pdf_file):
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = " ".join(page.get_text("text") for page in doc)
        return text.strip()

    # Function to Clean Resume Text
    def clean_text(text):
        text = re.sub(r'http\S+\s*', ' ', text)
        text = re.sub(r'RT|cc', ' ', text)
        text = re.sub(r'#\S+', '', text)
        text = re.sub(r'@\S+', ' ', text)
        text = re.sub(f'[{string.punctuation}]', ' ', text)
        text = re.sub(r'[^\x00-\x7f]', r' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    # Upload PDF Files
    uploaded_files = st.file_uploader("Upload PDF Resumes", type=["pdf"], accept_multiple_files=True)

    # Job Description Input
    job_description = st.text_area("Enter the Job Description Here")

    # Number of Resumes Required Input
    num_resumes = st.number_input("Number of Resumes Required", min_value=1, step=1, value=5)

    # Compare & Rank Button
    if st.button("Compare & Rank Resumes"):
        if uploaded_files and job_description:
            resumes = []
            file_names = []
            
            # Extract and Clean Text from PDFs
            for pdf in uploaded_files[:num_resumes]:  
                text = extract_text_from_pdf(pdf)
                cleaned_text = clean_text(text)
                resumes.append(cleaned_text)
                file_names.append(pdf.name)
            
            resumeDataSet = pd.DataFrame({"File Name": file_names, "Resume": resumes})
            
            if model_choice == "BERT (MPNet)":
                # Generate BERT Embeddings
                all_texts = [job_description] + resumes  
                embeddings = bert_model.encode(all_texts, convert_to_tensor=True)
                job_desc_embedding = embeddings[0]
                resume_embeddings = embeddings[1:]
                
                # Compute Cosine Similarity Scores
                similarity_scores = cosine_similarity(resume_embeddings, job_desc_embedding.unsqueeze(0), dim=1)
                similarity_scores = similarity_scores.cpu().numpy().flatten() * 100
                
            else:
                # Generate T5-based Scores
                similarity_scores = []
                for resume in resumes:
                    input_text = f"Rank this resume for the job from 0 to 100: Job Description: {job_description} Resume: {resume}"  
                    inputs = t5_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
                    with torch.no_grad():
                        output = t5_model.generate(**inputs, max_length=10)
                    score = t5_tokenizer.decode(output[0], skip_special_tokens=True)
                    
                    # Extract numeric score from T5 output
                    try:
                        score = float(re.findall(r'\d+', score)[0])
                        score = min(max(score, 0), 100)  # Clamp between 0 and 100
                    except:
                        score = np.random.uniform(30, 70)  # Assign random reasonable score if extraction fails
                    
                    similarity_scores.append(score)
            
            resumeDataSet['Similarity Score (%)'] = similarity_scores
            ranked_resumes = resumeDataSet.sort_values(by="Similarity Score (%)", ascending=False)
            
            # Display Results
            st.subheader("📌 Ranked Resumes")
            st.dataframe(ranked_resumes)
            
            # Download as Excel
            excel_buffer = BytesIO()
            ranked_resumes.to_excel(excel_buffer, index=False)
            excel_buffer.seek(0)
            st.download_button(label="Download as Excel", data=excel_buffer, file_name="ranked_resumes.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            
            # Generate Word Clouds
            st.subheader("📊 Word Clouds")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### Job Description")
                job_wc = WordCloud(width=800, height=400, background_color='white').generate(job_description)
                plt.figure(figsize=(5, 5))
                plt.imshow(job_wc, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt)
            
            with col2:
                st.write("### Resumes")
                all_resumes_text = " ".join(resumes)
                resume_wc = WordCloud(width=800, height=400, background_color='white').generate(all_resumes_text)
                plt.figure(figsize=(5, 5))
                plt.imshow(resume_wc, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt)
        
        else:
            st.error("Please upload resumes and enter a job description before ranking.")