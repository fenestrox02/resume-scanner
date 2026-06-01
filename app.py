import os
import json
import re
import numpy as np
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from scipy.sparse import hstack, csr_matrix
import joblib
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

app = Flask(__name__)

# Load models
print("Loading models...")
clf       = joblib.load('models/classification_model.pkl')
tfidf_clf = joblib.load('models/tfidf.pkl')
bert      = joblib.load('models/sentence_transformer.pkl')
jd_model  = joblib.load('models/jd_fit_model.pkl')
jd_le     = joblib.load('models/jd_fit_label_encoder.pkl')
ats_model = joblib.load('models/ats_model.pkl')
ats_tfidf = joblib.load('models/ats_tfidf.pkl')
ats_ohe   = joblib.load('models/ats_ohe.pkl')
print("Models loaded!")

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

    # 1. Classification
    tfidf_vec = tfidf_clf.transform([resume_clean])
    bert_vec  = csr_matrix(bert.encode([resume_clean]))
    category  = clf.predict(hstack([tfidf_vec, bert_vec]))[0]

    # 2. JD Matching
    jd_pred  = jd_model.predict([combined])[0]
    jd_label = jd_le.inverse_transform([jd_pred])[0]

    # 3. ATS Scoring
    ats_vec        = ats_tfidf.transform([resume_clean])
    ats_label_mapped = 'Good Fit' if jd_label == 'Fit' else 'No Fit'
    ats_label      = ats_ohe.transform([[ats_label_mapped]])
    ats_score = np.clip(ats_model.predict(hstack([ats_vec, ats_label]))[0], 0, 100)

    return {
        'category':  category,
        'jd_match':  jd_label,
        'ats_score': round(float(ats_score), 1)
    }


def get_gemini_feedback(resume_text, jd_text, category, jd_match, ats_score):
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

Based on your expert analysis, return ONLY a JSON object with these exact keys (no extra text):
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
  }}
}}"""

    try:
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt
        )
        text = response.text.strip().replace('```json', '').replace('```', '').strip()
        return json.loads(text)
    except Exception as e:
        print(f"Gemini error: {e}")
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


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/scan', methods=['POST'])
def api_scan():
    data        = request.json
    resume_text = data.get('resume', '')
    jd_text     = data.get('jd', '')

    if not resume_text or not jd_text:
        return jsonify({'error': 'Missing resume or JD'}), 400

    ml_result = scan_resume(resume_text, jd_text)
    ai_result = get_gemini_feedback(
        resume_text, jd_text,
        ml_result['category'],
        ml_result['jd_match'],
        ml_result['ats_score']
    )

    return jsonify({**ml_result, **ai_result})

if __name__ == '__main__':
    app.run(debug=True, port=5000)