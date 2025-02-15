# edunet-foundation

# 📄 Resume Ranking System
# 🔹 Project Overview
The Resume Ranking System is an AI-powered application that ranks resumes based on a given job description. It leverages BERT (MPNet) and T5 Transformers to calculate the relevance of each resume and provides ranked results to streamline the hiring process.

# ✅ Features:

Extracts and processes text from PDF resumes.
Uses BERT (MPNet) and T5 Transformers to evaluate resume-job description similarity.
Generates Word Clouds for job descriptions and resumes.
Provides ranked resumes based on cosine similarity or T5 scoring.
Allows downloading ranked results in Excel format.

# 🚀 Technologies Used

Python 3.8+
Streamlit (For UI)
PyMuPDF (pymupdf) (For PDF text extraction)
NLTK (For text preprocessing)
BERT (MPNet) & T5 Transformer (For resume ranking)
Pandas (For data handling)
Matplotlib & WordCloud (For visualization)

# 📌 Installation Guide
# 1️⃣ Clone the Repository
git clone https://github.com/kADALIdurgasivasankarprasad/edunet-project.git
cd edunet-project

# 2️⃣ Create a Virtual Environment (Recommended)

python -m venv venv  # For Windows/Linux

source venv/bin/activate  # For Mac/Linux

venv\Scripts\activate  # For Windows

# 3️⃣ Install Dependencies

pip install -r requirements.txt
If you encounter an error with fitz, install PyMuPDF:

pip uninstall fitz -y
pip install pymupdf
If you get a missing sentencepiece error, install:

pip install sentencepiece
▶️ Running the Application
Once dependencies are installed, run the Streamlit app:

streamlit run app.py

# 🛠 How It Works

1️⃣ Home Page: Explains the project purpose and features.
2️⃣ Upload Resumes: Upload multiple PDF resumes.
3️⃣ Enter Job Description: Input the job description for ranking.
4️⃣ Select Model: Choose between BERT (MPNet) and T5 Transformer.
5️⃣ Compare & Rank: Click the "Compare & Rank Resumes" button.
6️⃣ View Results: See ranked resumes with similarity scores.
7️⃣ Download as Excel: Export the ranked results in Excel format.
8️⃣ Visualize Data: View Word Clouds of job descriptions and resumes.

