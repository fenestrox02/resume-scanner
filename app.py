import os
import io
import json
import re
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from scipy.sparse import hstack, csr_matrix
import joblib
from openai import OpenAI

# DOCX / PDF
try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    import fitz  
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

client = OpenAI(
    api_key= "your_api_key",
    base_url="https://openrouter.ai/api/v1"
)

MODEL = "google/gemma-4-31b-it:free"

app = Flask(__name__)
CORS(app)

# Load models 
clf       = joblib.load('models/classification_model.pkl')
tfidf_clf = joblib.load('models/tfidf.pkl')
bert      = joblib.load('models/sentence_transformer.pkl')
jd_model  = joblib.load('models/jd_fit_model.pkl')
jd_le     = joblib.load('models/jd_fit_label_encoder.pkl')
ats_model = joblib.load('models/ats_model.pkl')
ats_tfidf = joblib.load('models/ats_tfidf.pkl')
ats_ohe   = joblib.load('models/ats_ohe.pkl')

#File 
def extract_text_from_file(file_bytes, filename):
    ext = filename.lower().split('.')[-1]
    if ext == 'txt':
        return file_bytes.decode('utf-8', errors='ignore')
    elif ext == 'docx' and HAS_DOCX:
        doc = DocxDocument(io.BytesIO(file_bytes))
        return '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
    elif ext == 'pdf' and HAS_PDF:
        pdf = fitz.open(stream=file_bytes, filetype='pdf')
        return '\n'.join([page.get_text() for page in pdf])
    else:
        return file_bytes.decode('utf-8', errors='ignore')

def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def scan_resume(resume_text, jd_text):
    resume_clean = clean_text(resume_text)
    combined     = resume_clean + ' ' + clean_text(jd_text)

    tfidf_vec = tfidf_clf.transform([resume_clean])
    bert_vec  = csr_matrix(bert.encode([resume_clean]))
    category  = clf.predict(hstack([tfidf_vec, bert_vec]))[0]

    jd_pred  = jd_model.predict([combined])[0]
    jd_label = jd_le.inverse_transform([jd_pred])[0]

    ats_vec          = ats_tfidf.transform([resume_clean])
    ats_label_mapped = 'Good Fit' if jd_label == 'Fit' else 'No Fit'
    ats_label        = ats_ohe.transform([[ats_label_mapped]])
    ats_score        = np.clip(ats_model.predict(hstack([ats_vec, ats_label]))[0], 0, 100)

    return {
        'category':  category,
        'jd_match':  jd_label,
        'ats_score': round(float(ats_score), 1)
    }


def get_ai_feedback(resume_text, jd_text, category, jd_match, ats_score):
    prompt = f"""You are a senior HR director with 20 years of experience in talent acquisition across Fortune 500 companies. You have reviewed over 50,000 resumes and have deep expertise in ATS systems, hiring trends, and what separates top candidates from average ones.

Your task is to critically evaluate the following resume against the job description with the same sharp eye you would use when shortlisting candidates for a high-stakes role.

--- RESUME ---
{resume_text[:2000]}

--- JOB DESCRIPTION ---
{jd_text[:1000]}

--- AUTOMATED SCREENING RESULTS ---
Detected Job Category: {category}
JD Fit Assessment: {jd_match}
ATS Score: {ats_score}/100

Based on your expert analysis, return ONLY a JSON object with these exact keys (no extra text, no markdown):
{{
  "overall_assessment": "A sharp 2-3 sentence executive summary of this candidate's fit, written as if briefing a hiring manager",
  "keywords_found": ["list of strong matching keywords/skills found in resume"],
  "keywords_missing": ["list of important keywords/skills from JD that are absent in resume"],
  "recommendations": [
    {{"action": "specific, actionable improvement", "impact": "+X%", "section": "which resume section to improve"}}
  ],
  "category_breakdown": {{
    "keyword_match": 0,
    "formatting": 0,
    "experience_relevance": 0
  }},
  "ai_score": 0
}}"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0.3,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Model returned empty response")
        text = content.strip()
        # Strip markdown fences + think tags
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        text = re.sub(r'```json|```', '', text).strip()
        return json.loads(text)
    except Exception as e:
        print(f"AI error: {e}")
        return {
            "overall_assessment": "AI analysis temporarily unavailable. Please try again later.",
            "keywords_found": [],
            "keywords_missing": [],
            "recommendations": [],
            "category_breakdown": {
                "keyword_match": 0,
                "formatting": 0,
                "experience_relevance": 0
            }
        }

#Routes 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/scan', methods=['POST'])
def api_scan():
    # Support both JSON (text) and multipart (file upload)
    if request.content_type and 'multipart/form-data' in request.content_type:
        resume_text = request.form.get('resume', '')
        jd_text     = request.form.get('jd', '')
        file        = request.files.get('resume_file')
        if file and file.filename:
            resume_text = extract_text_from_file(file.read(), file.filename)
    else:
        data        = request.json or {}
        resume_text = data.get('resume', '')
        jd_text     = data.get('jd', '')

    if not resume_text or not jd_text:
        return jsonify({'error': 'Missing resume or job description'}), 400

    ml_result = scan_resume(resume_text, jd_text)
    ai_result = get_ai_feedback(
        resume_text, jd_text,
        ml_result['category'],
        ml_result['jd_match'],
        ml_result['ats_score']
    )

    ai_score = ai_result.get('ai_score', None)
    if ai_score is not None and ai_score != ml_result['ats_score']:
        ml_result['ats_score'] = ai_score

    return jsonify({**ml_result, **ai_result})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
