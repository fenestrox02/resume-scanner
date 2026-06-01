import pandas as pd
import re
resume = pd.read_csv('Resume.csv')
print(resume['Category'].value_counts())
print(f"\nJob categories: {resume['Category'].nunique()}")

##check null
print(resume.isnull().sum())

def clean_resume(text):
    text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer

resume['clean_text'] = resume['Resume_str'].apply(clean_resume)

X_train, X_test, y_train, y_test = train_test_split(
    resume['clean_text'],
    resume['Category'],
    test_size=0.2,
    random_state=42,
    stratify=resume['Category'])

# TF-IDF features
print("TF-IDF encoding...")
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True, stop_words='english', min_df=2)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

# BERT features
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
X_train_bert = csr_matrix(bert_model.encode(X_train.tolist(), show_progress_bar=True))
X_test_bert  = csr_matrix(bert_model.encode(X_test.tolist(),  show_progress_bar=True))

# Stack TF-IDF + BERT
X_train_combined = hstack([X_train_tfidf, X_train_bert])
X_test_combined  = hstack([X_test_tfidf,  X_test_bert])

# Train
clf = LinearSVC(C=1.0, class_weight='balanced', max_iter=2000)
clf.fit(X_train_combined, y_train)

y_pred = clf.predict(X_test_combined)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

import joblib
joblib.dump(clf, 'classification_model.pkl')
joblib.dump(tfidf, 'tfidf.pkl')
joblib.dump(bert_model, 'sentence_transformer.pkl')