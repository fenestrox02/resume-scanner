import pandas as pd
import re
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from scipy.sparse import hstack

train = pd.read_csv('ats score train.csv')
test  = pd.read_csv('ats scorevalidation.csv')
print(train.shape, test.shape)
print(train.isnull().sum())

def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

train['clean'] = train['text'].apply(clean_text)
test['clean']  = test['text'].apply(clean_text)

# TF-IDF
tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(train['clean'])
X_test_tfidf  = tfidf.transform(test['clean'])

# One-hot encode original_label
ohe = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
X_train_label = ohe.fit_transform(train[['original_label']])
X_test_label  = ohe.transform(test[['original_label']])

# Stack TF-IDF + label
X_train = hstack([X_train_tfidf, X_train_label])
X_test  = hstack([X_test_tfidf,  X_test_label])

y_train = train['ats_score']
y_test  = test['ats_score']

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.4f}")

joblib.dump(model, 'ats_model.pkl')
joblib.dump(tfidf, 'ats_tfidf.pkl')
joblib.dump(ohe,   'ats_ohe.pkl')
