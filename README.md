# Resume Ranking System

## 📌 Overview
This is a FastAPI-based **Resume Ranking System** that extracts ranking criteria from a job description and scores multiple resumes based on those criteria. It utilizes **Groq's Llama3-70B** model to:

- **Extract key ranking criteria** from a job description.
- **Analyze and score resumes** based on the extracted criteria.
- **Generate a CSV file** with ranked candidates and their scores.

## 🚀 Features
- 📂 Upload **PDF, DOCX, or TXT** job descriptions and resumes.
- 📊 Extract **key ranking criteria** using an AI model.
- 🏆 Score resumes on a **0-5 scale** based on job criteria.
- 📥 Download results as a **CSV file**.
- 🔥 Supports **CORS** for frontend integration.

## 🛠️ Installation & Setup

### 1️⃣ Clone the repository
```sh
git clone https://github.com/aryanjain762/JD-Resume-ranker.git
cd JD-Resume-ranker
```

### 2️⃣ Install dependencies
Make sure you have Python 3.8+ installed, then run:
```sh
pip install -r requirements.txt
```

### 3️⃣ Set up environment variables
Create a `.env` file and add your **Groq API Key**:

```sh
export GROQ_API_KEY=your-groq-api-key-here 

### 4️⃣ Run the FastAPI server
```sh
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: **http://127.0.0.1:8000**

## 📝 API Endpoints

### **1️⃣ Extract Job Criteria**
**Endpoint:**
```
POST /extract-criteria
```
**Description:** Extracts key ranking criteria from a job description file.

**Request:**
- **file** (PDF/DOCX/TXT) - Job description file.

**Response:**
```json
{
  "criteria": [
    "5+ years experience in Python",
    "Experience with Machine Learning",
    "Strong knowledge of FastAPI"
  ]
}
```

---
### **2️⃣ Score Resumes**
**Endpoint:**
```
POST /score-resumes
```
**Description:** Scores resumes based on provided criteria.

**Request:**
- **criteria** (List) - Criteria extracted from job description.
- **files** (List of PDF/DOCX/TXT) - Resumes to be scored.

**Response:**
- CSV file download containing candidate names, scores per criterion, and total score.


```

Now, the API will be available at **http://localhost:8000**


---
**💡 Made with ❤️ using FastAPI & Llama3-70B**

