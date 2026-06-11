# Resume Scanner

Resume Scanner is an AI-powered Applicant Tracking System (ATS) that helps job seekers evaluate how well their resumes match a job description and enables recruiters to screen candidates efficiently.

## Features

* Resume Category Classification
* Resume–JD Fit Prediction
* ATS Score Prediction (0–100)
* AI-generated Resume Feedback
* Keyword Match Analysis
* Bulk Resume Screening
* Candidate Ranking & CSV Export

## Tech Stack

* Python
* Flask
* Scikit-Learn
* Sentence Transformers
* OpenRouter (Gemma)
* HTML/CSS/JavaScript

## Models

| Task                  | Model                     |
| --------------------- | ------------------------- |
| Resume Classification | LinearSVC + TF-IDF + BERT |
| Resume–JD Matching    | LinearSVC + TF-IDF        |
| ATS Score Prediction  | Ridge Regression          |

## Run Locally

```
pip install flask flask-cors scikit-learn sentence-transformers pymupdf python-docx openai numpy scipy joblib

python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Author

Khánh Nguyễn
Business Administration, Korea University
