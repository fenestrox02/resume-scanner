import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

train = pd.read_csv('resume & jd fit train.csv')
test  = pd.read_csv('resume & jd fit test.csv')

print(train.shape)
print(train.head(5))
print(train['label'].value_counts())
print(test.shape)
print(train.isnull().sum())

def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

train['combined'] = (train['resume_text'] + ' ' + train['job_description_text']).apply(clean_text)
test['combined']  = (test['resume_text']  + ' ' + test['job_description_text']).apply(clean_text)

# Good Fit + Potential Fit = Fit
train['label_binary'] = train['label'].replace({'Good Fit': 'Fit', 'Potential Fit': 'Fit'})
test['label_binary']  = test['label'].replace({'Good Fit': 'Fit', 'Potential Fit': 'Fit'})
print(train['label_binary'].value_counts())

le = LabelEncoder()
train['label_enc'] = le.fit_transform(train['label_binary'])
test['label_enc']  = le.transform(test['label_binary'])
print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )),
    ('clf', LinearSVC(
        C=5.0,
        max_iter=2000,
        class_weight='balanced',
    ))
])

pipeline.fit(train['combined'], train['label_enc'])
y_pred = pipeline.predict(test['combined'])
print(f"\nAccuracy: {accuracy_score(test['label_enc'], y_pred):.2f}")
print(classification_report(test['label_enc'], y_pred, target_names=le.classes_))

joblib.dump(pipeline, 'jd_fit_model.pkl')
joblib.dump(le, 'jd_fit_label_encoder.pkl')
