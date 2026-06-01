import joblib
import re
import numpy as np
from scipy.sparse import hstack, csr_matrix

clf        = joblib.load('classification_model.pkl')
tfidf_clf  = joblib.load('tfidf.pkl')
bert       = joblib.load('sentence_transformer.pkl')
jd_model   = joblib.load('jd_fit_model.pkl')
jd_le      = joblib.load('jd_fit_label_encoder.pkl')
ats_model  = joblib.load('ats_model.pkl')
ats_tfidf  = joblib.load('ats_tfidf.pkl')
ats_ohe    = joblib.load('ats_ohe.pkl')

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

    # 1. Classification (TF-IDF + BERT)
    tfidf_vec  = tfidf_clf.transform([resume_clean])
    bert_vec   = csr_matrix(bert.encode([resume_clean]))
    X_clf      = hstack([tfidf_vec, bert_vec])
    category   = clf.predict(X_clf)[0]

    # 2. JD Matching (Pipeline)
    jd_pred_enc = jd_model.predict([combined])[0]
    jd_label    = jd_le.inverse_transform([jd_pred_enc])[0]

    # 3. ATS Scoring
    ats_vec          = ats_tfidf.transform([resume_clean])
    ats_label_mapped = 'Good Fit' if jd_label == 'Fit' else 'No Fit'
    ats_label        = ats_ohe.transform([[ats_label_mapped]])
    ats_input = hstack([ats_vec, ats_label])
    ats_score = np.clip(ats_model.predict(ats_input)[0], 0, 100)

    return {
        'category':  category,
        'jd_match':  jd_label,
        'ats_score': round(float(ats_score), 1)
    }

# Test
resume = """
John Doe | john.doe@email.com | LinkedIn: linkedin.com/in/johndoe

SUMMARY
Senior Python Developer with 6 years of experience building scalable machine learning pipelines and data-driven applications. Proficient in TensorFlow, PyTorch, scikit-learn, and cloud deployment on AWS.

SKILLS
Languages: Python, SQL, Bash
ML/AI: TensorFlow, PyTorch, scikit-learn, XGBoost, NLP, Computer Vision
Tools: Docker, Kubernetes, Git, Airflow, Spark
Cloud: AWS (SageMaker, EC2, S3), GCP

EXPERIENCE
Senior ML Engineer — TechCorp (2021–Present)
- Built and deployed 5 production ML models serving 2M+ users daily
- Reduced model inference latency by 40% using model quantization and ONNX
- Led a team of 3 engineers to deliver an NLP-based document classification system

Python Developer — DataStartup (2018–2021)
- Developed REST APIs with FastAPI and Flask for ML model serving
- Automated ETL pipelines using Apache Airflow, reducing manual work by 60%
- Implemented A/B testing framework for model performance evaluation

EDUCATION
B.Sc. Computer Science — HCMUS (2014–2018), GPA: 3.8/4.0
"""

jd = """
We are looking for a Senior Python Developer with strong machine learning experience to join our AI team.

Requirements:
- 5+ years of Python development experience
- Hands-on experience with ML frameworks (TensorFlow, PyTorch, scikit-learn)
- Experience deploying ML models to production
- Familiarity with cloud platforms (AWS or GCP)
- Strong knowledge of REST API development
- Experience with Docker and Kubernetes
"""

result = scan_resume(resume, jd)
print("\n===== RESUME SCANNER RESULT =====")
print(f"Category:  {result['category']}")
print(f"JD Match:  {result['jd_match']}")
print(f"ATS Score: {result['ats_score']}/100")
print("==================================")