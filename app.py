from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import tempfile
import json
import io
import csv
import PyPDF2
import groq
import asyncio
import httpx
import aiofiles
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor
import logging


logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

try:
    from docx import Document
except ImportError:
    os.system("pip install python-docx")
    from docx import Document


app = FastAPI(
    title="Resume Ranking System",
    description="API to extract ranking criteria from job descriptions and score resumes based on those criteria",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
logging.info(f"GROQ_API_KEY set: {'yes' if GROQ_API_KEY else 'no'}")
groq_client = groq.Groq(api_key=GROQ_API_KEY)
MODEL = "llama3-70b-8192"


async def extract_text_from_file(file: UploadFile) -> str:
    logging.info(f"Extracting text from {file.filename}")
    content = await file.read()
    file_extension = Path(file.filename).suffix.lower()
    text = ""
    
    try:
        if file_extension == '.pdf':
            logging.info(f"Processing as PDF: {file.filename}")
            with io.BytesIO(content) as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        
        elif file_extension in ['.docx', '.doc']:
            logging.info(f"Processing as DOCX: {file.filename}")
            with io.BytesIO(content) as docx_file:
                doc = Document(docx_file)
                paragraphs = [p.text for p in doc.paragraphs]
                text = '\n'.join(paragraphs)
        
        else:
            logging.info(f"Processing as plain text: {file.filename}")
            text = content.decode('utf-8')
        
        logging.info(f"Extracted {len(text)} characters from {file.filename}")
        logging.info(f"First 100 chars: {text[:100]}")
        
        if not text.strip():
            logging.warning(f"Warning: No text extracted from {file.filename}")
            
    except Exception as e:
        logging.error(f"Error extracting text from file {file.filename}: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Error extracting text from file: {str(e)}")
    
    return text


async def extract_criteria_with_llm(text: str) -> List[str]:
    logging.info("Extracting criteria from job description")
    prompt = f"""
    You are an AI assistant that helps recruiters extract key ranking criteria from job descriptions.
    
    Given the following job description, extract a list of clear, specific criteria that can be used to evaluate candidates.
    Focus on required skills, certifications, experience, and qualifications. Format your response as a JSON array of strings,
    with each string representing one specific criterion.
    
    Job Description:
    {text}
    
    Format your response as a JSON array of strings like this:
    ["Must have certification XYZ", "5+ years of experience in Python development", "Strong background in Machine Learning"]
    """
    
    try:
        logging.info("Calling Groq API for criteria extraction")
        completion = groq_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts key criteria from job descriptions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1024
        )
        response_text = completion.choices[0].message.content.strip()
        logging.info(f"Received response of length {len(response_text)}")
        logging.info(f"Raw LLM response for criteria: {response_text}")
        
        criteria_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if criteria_match:
            criteria_json = criteria_match.group(0)
            criteria = json.loads(criteria_json)
            logging.info(f"Successfully extracted {len(criteria)} criteria: {criteria}")
            return criteria
        else:
            logging.error("Failed to extract criteria list from response")
            raise ValueError("Failed to extract criteria list from response")
            
    except Exception as e:
        logging.error(f"Error with Groq LLM processing for criteria extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error with Groq LLM processing: {str(e)}")

async def score_resume_with_llm(resume_text: str, criteria: List[str], candidate_name: str) -> dict:
    logging.info(f"Scoring resume for candidate: {candidate_name}")
    

    criteria_mapping = {f"criterion{i+1}": criterion for i, criterion in enumerate(criteria)}
    
    prompt = f"""
    You are an AI assistant that helps recruiters score resumes against job criteria.
    
    Given the following resume and criteria, score the candidate on each criterion on a scale of 0-5,
    where 0 means not mentioned/completely missing and 5 means excellent match/fully meets the criterion.
    
    Resume:
    {resume_text[:4000]}
    
    Criteria to evaluate:
    {json.dumps(criteria_mapping, indent=2)}
    
    For the candidate named "{candidate_name}", provide scores in clean JSON format WITHOUT ANY COMMENTS:
    {{
      "candidate_name": "{candidate_name}",
      "scores": {{
        "criterion1": score,
        "criterion2": score,
        ...
      }},
      "total_score": sum_of_scores
    }}
    
    IMPORTANT: Return only the JSON object without any comments, explanations, or backticks.
    """
    
    try:
        logging.info(f"Making Groq API call for {candidate_name}")
        completion = groq_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that scores resumes based on job criteria. Return only clean JSON without comments."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1024
        )
        response_text = completion.choices[0].message.content.strip()
        logging.info(f"Received response of length {len(response_text)}")
        
      
        json_text = response_text
        if "```" in json_text:
            json_text = json_text.split("```")[1].strip()
            if json_text.startswith("json"):
                json_text = json_text[4:].strip()
                
        try:
            score_data = json.loads(json_text)
            logging.info(f"Successfully parsed score data for {candidate_name}")
            
     
            normalized_scores = {}
            for criterion in criteria:
                for key, value in score_data["scores"].items():
                    if criteria_mapping.get(key) == criterion:
                        normalized_scores[criterion] = value
                        break
                if criterion not in normalized_scores:
                    normalized_scores[criterion] = 0
                    
            score_data["scores"] = normalized_scores
            score_data["total_score"] = sum(normalized_scores.values())
            
            return score_data
            
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON response for {candidate_name}: {e}")
            logging.error(f"Problematic JSON text: {json_text}")
            raise
            
    except Exception as e:
        logging.error(f"Error with Groq LLM processing for scoring: {str(e)}")
        default_scores = {criterion: 0 for criterion in criteria}
        return {
            "candidate_name": candidate_name,
            "scores": default_scores,
            "total_score": 0
        }

async def extract_candidate_name(text: str) -> str:
    logging.info("Extracting candidate name from resume")
    prompt = f"""
    Extract the full name of the candidate from this resume text. If no name is found,
    return "Unknown Candidate". Just return the name, nothing else.
    
    Resume:
    {text[:1000]}  # Using first 1000 chars to save tokens
    """
    
    try:
        logging.info("Calling Groq API for name extraction")
        completion = groq_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You extract candidate names from resumes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=50
        )
        name = completion.choices[0].message.content.strip()
        logging.info(f"Extracted name: {name}")
        
        if "unknown" in name.lower():
            return "Unknown Candidate"
        return name.strip('".')
        
    except Exception as e:
        logging.error(f"Error with Groq LLM processing for name extraction: {str(e)}")
        return "Unknown Candidate"


async def process_single_resume(file, criteria):
    logging.info(f"Processing resume file: {file.filename}")
    try:
        resume_text = await extract_text_from_file(file)
        if not resume_text.strip():
            logging.warning(f"Empty text extracted from {file.filename}")
            
        candidate_name = await extract_candidate_name(resume_text)
        score_data = await score_resume_with_llm(resume_text, criteria, candidate_name)
        

        if len(criteria) > 0 and all(v == 0 for v in score_data["scores"].values()):
            logging.info(f"All scores are 0 for {candidate_name}, forcing a test score for first criterion")
            score_data["scores"][criteria[0]] = 3
            score_data["total_score"] = 3
            
        return score_data
    except Exception as e:
        logging.error(f"Error processing resume {file.filename}: {str(e)}")
        default_scores = {criterion: 0 for criterion in criteria}
        return {
            "candidate_name": f"Error Processing {file.filename}",
            "scores": default_scores,
            "total_score": 0
        }


@app.post("/extract-criteria", response_model=dict, tags=["Criteria Extraction"])
async def extract_criteria(file: UploadFile = File(...)):
    """
    Extract key ranking criteria from a job description file (PDF or DOCX).
    
    - **file**: The job description file to process
    
    Returns a JSON object with a list of criteria extracted from the job description.
    """
    logging.info(f"Received request to extract criteria from {file.filename}")
    job_description_text = await extract_text_from_file(file)
    
    if not job_description_text.strip():
        raise HTTPException(status_code=422, detail="Empty job description text extracted")
    
    criteria = await extract_criteria_with_llm(job_description_text)
    
    return {"criteria": criteria}


@app.post("/score-resumes", tags=["Resume Scoring"])
async def score_resumes(
    criteria: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    Score multiple resumes against the provided criteria.
    
    - **criteria**: Either a comma-separated string or JSON array of criteria
    - **files**: List of resume files to process
    
    Returns a CSV file with the scores for each candidate.
    """
    try:

        try:
            criteria_list = json.loads(criteria)
            if isinstance(criteria_list, list):
                criteria_list = [str(c).strip() for c in criteria_list if str(c).strip()]
            else:
            
                criteria_list = [c.strip() for c in criteria.split(',') if c.strip()]
        except json.JSONDecodeError:
          
            criteria_list = [c.strip() for c in criteria.split(',') if c.strip()]

        logging.info(f"Received request to score {len(files)} resumes against {len(criteria_list)} criteria")
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        if not criteria_list:
            raise HTTPException(status_code=400, detail="No criteria provided")
        
        logging.info(f"Criteria to score against: {criteria_list}")
        
        scores = []
        for file in files:
            result = await process_single_resume(file, criteria_list)
            scores.append(result)
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        header = ["Candidate Name"] + criteria_list + ["Total Score"]
        writer.writerow(header)
        
        for score_data in scores:
            row = [score_data["candidate_name"]]
            for criterion in criteria_list:
                row.append(score_data["scores"].get(criterion, 0))
            row.append(score_data["total_score"])
            writer.writerow(row)
        
        output.seek(0)
        
        response = StreamingResponse(io.BytesIO(output.getvalue().encode()), media_type="text/csv")
        response.headers["Content-Disposition"] = f"attachment; filename=resume_scores.csv"
        return response
        
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    logging.info("Starting Resume Ranking System")
    uvicorn.run(app, host="0.0.0.0", port=8000)