import os
import json
import re
import numpy as np
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from scipy.sparse import hstack, csr_matrix
import joblib
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv('OPENROUTER_API_KEY'),
    base_url="https://openrouter.ai/api/v1"
)

MODEL = "qwen/qwen3-235b-a22b:free"

app = Flask(__name__)
CORS(app)

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
    prompt = f"""You are a senior HR director with 20 years of experience in talent acquisition.
Critically evaluate the resume against the job description.

--- RESUME ---
{resume_text[:2000]}

--- JOB DESCRIPTION ---
{jd_text[:1000]}

--- ML SCREENING RESULTS ---
Detected Category: {category}
JD Fit: {jd_match}
ATS Score: {ats_score}/100

Return ONLY a valid JSON object, no extra text, no markdown:
{{
  "overall_assessment": "2-3 sentence executive summary of candidate fit",
  "keywords_found": ["keyword1", "keyword2"],
  "keywords_missing": ["keyword1", "keyword2"],
  "recommendations": [
    {{"action": "specific improvement", "impact": "+X% score boost", "section": "resume section"}}
  ],
  "category_breakdown": {{
    "keyword_match": 0,
    "formatting": 0,
    "experience_relevance": 0
  }}
}}"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0.3,
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        text = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        text = re.sub(r'```json|```', '', text).strip()
        # Strip <think> tags from Qwen
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        return json.loads(text)
    except Exception as e:
        print(f"AI error: {e}")
        return {
            "overall_assessment": "AI analysis temporarily unavailable.",
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
    resume_text = data.get('resume', '').strip()
    jd_text     = data.get('jd', '').strip()

    if not resume_text or not jd_text:
        return jsonify({'error': 'Missing resume or job description'}), 400

    ml_result = scan_resume(resume_text, jd_text)
    ai_result = get_ai_feedback(
        resume_text, jd_text,
        ml_result['category'],
        ml_result['jd_match'],
        ml_result['ats_score']
    )

    return jsonify({**ml_result, **ai_result})

if __name__ == '__main__':
    app.run(debug=True, port=5000)