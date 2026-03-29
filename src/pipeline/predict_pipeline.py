import os
import sys
import pandas as pd
import joblib
from src.exception import CustomException
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        self.model_path        = os.path.join('artifacts', 'model.pkl')
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

    def predict(self, features: pd.DataFrame):
        try:
            model        = joblib.load(self.model_path)
            preprocessor = joblib.load(self.preprocessor_path)

            data_scaled = preprocessor.transform(features)
            prediction  = model.predict(data_scaled)

            label_map = {0: 'Excellent', 1: 'Fail', 2: 'Good', 3: 'Outstanding'}
            return label_map.get(int(prediction[0]), str(prediction[0]))

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course):
        self.gender                      = gender
        self.race_ethnicity              = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch                       = lunch
        self.test_preparation_course     = test_preparation_course

    def get_data_as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([{
            'gender':                      self.gender,
            'race_ethnicity':              self.race_ethnicity,
            'parental_level_of_education': self.parental_level_of_education,
            'lunch':                       self.lunch,
            'test_preparation_course':     self.test_preparation_course
        }])


# ── Predictive Maintenance ────────────────────────────────────────────────────

class MaintenancePredictPipeline:
    def __init__(self):
        self.model_path = os.path.join('artifacts', 'maintenance_model.pkl')

    def predict(self, features: pd.DataFrame):
        try:
            model = joblib.load(self.model_path)
            prediction  = model.predict(features)
            proba       = model.predict_proba(features)          # shape (1, 2)
            failure_pct = round(float(proba[0][1]) * 100, 1)    # probability of class 1
            label = 'FAILURE PREDICTED' if int(prediction[0]) == 1 else 'NORMAL OPERATION'
            return label, failure_pct
        except Exception as e:
            raise CustomException(e, sys)


class MaintenanceData:
    def __init__(self, footfall, tempMode, AQ, USS, CS, VOC, RP, IP, Temperature):
        self.footfall    = footfall
        self.tempMode    = tempMode
        self.AQ          = AQ
        self.USS         = USS
        self.CS          = CS
        self.VOC         = VOC
        self.RP          = RP
        self.IP          = IP
        self.Temperature = Temperature

    def get_data_as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([{
            'footfall':    float(self.footfall),
            'tempMode':    float(self.tempMode),
            'AQ':          float(self.AQ),
            'USS':         float(self.USS),
            'CS':          float(self.CS),
            'VOC':         float(self.VOC),
            'RP':          float(self.RP),
            'IP':          float(self.IP),
            'Temperature': float(self.Temperature),
        }])


# ── Spam / Ham ────────────────────────────────────────────────────────────────

import re
import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Ensure NLTK data is available (downloads are no-ops if already present)
for _pkg in ('punkt', 'punkt_tab', 'wordnet', 'omw-1.4',
             'stopwords', 'averaged_perceptron_tagger_eng'):
    nltk.download(_pkg, quiet=True)

_stop       = set(stopwords.words('english'))
_lemmatizer = WordNetLemmatizer()


def _get_wordnet_pos(tag: str):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def _preprocess_message(text: str) -> str:
    """Reproduce the notebook's sentence_to_lemmatized_sentence for a single message."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    sentences = sent_tokenize(text)
    processed = []
    for sent in sentences:
        words = word_tokenize(sent)
        words = [w for w in words if w not in _stop and w not in string.punctuation]
        tagged = pos_tag(words)
        words  = [_lemmatizer.lemmatize(w, _get_wordnet_pos(t)) for w, t in tagged]
        processed.append(' '.join(words))
    return ' '.join(processed)


class SpamHamPredictPipeline:
    def __init__(self):
        self.model_path     = os.path.join('artifacts', 'spam_model.pkl')
        self.vectorizer_path = os.path.join('artifacts', 'spam_tfidf.pkl')

    def predict(self, message: str):
        try:
            model      = joblib.load(self.model_path)
            vectorizer = joblib.load(self.vectorizer_path)

            cleaned   = _preprocess_message(message)
            features  = vectorizer.transform([cleaned])
            proba     = model.predict_proba(features)       # shape (1, 2)
            spam_pct  = round(float(proba[0][1]) * 100, 1)
            # Use probability threshold instead of model label
            label     = 'SPAM' if spam_pct > 50 else 'HAM'
            return label, spam_pct

        except Exception as e:
            raise CustomException(e, sys)


class SpamHamData:
    def __init__(self, message: str):
        self.message = message


# ── Kindle Review Sentiment ───────────────────────────────────────────────────

import subprocess

def _download_if_missing(file_id: str, destination: str):
    """Downloads a file from Google Drive if it doesn't already exist."""
    if os.path.exists(destination):
        return
    try:
        import gdown
    except ImportError:
        logging.info("Installing gdown for model download...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
    
    url = f"https://drive.google.com/uc?id={file_id}"
    logging.info(f"Downloading {destination} from Google Drive...")
    gdown.download(url, destination, quiet=False)


def _preprocess_kindle_text(text: str) -> str:
    """Reproduce the notebook's preprocessing_the_text for a single review."""
    text = text.lower()
    # Clean but keep punctuation for sent_tokenize as in notebook
    text = re.sub(r"[^a-zA-Z\s.,!?;:\-]", " ", text)
    sentences = sent_tokenize(text)
    
    processed_doc = []
    for sentence in sentences:
        # Tokenize and filter stops
        words = [w for w in word_tokenize(sentence) if w not in _stop]
        
        lemmatized_sentence = []
        for w in words:
            tag = pos_tag([w])[0][1]
            wn_pos = _get_wordnet_pos(tag)
            lemmatized_sentence.append(_lemmatizer.lemmatize(w, wn_pos))
        
        processed_doc.append(" ".join(lemmatized_sentence))
    
    return " ".join(processed_doc)


class KindlePredictPipeline:
    def __init__(self):
        # The user's model is in the notebook folder, but we should probably 
        # look for it in artifacts or the source path.
        # User said: "kept only logistic_regression_model.joblib back where ever it was"
        self.model_path = os.path.join('notebook', 'Notebook', 'Kindle review sentiment analysis', 'logistic_regression_model.joblib')
        self.vectorizer_path = os.path.join('artifacts', 'kindle_vectorizer.joblib')
        self.vectorizer_id = "1Xjz353ARfeicN1MDJuR5y0frW7BuEjg8"

    def predict(self, review_text: str):
        try:
            # Ensure vectorizer exists (from Google Drive as per notebook)
            _download_if_missing(self.vectorizer_id, self.vectorizer_path)
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}")

            model = joblib.load(self.model_path)
            vectorizer = joblib.load(self.vectorizer_path)

            cleaned = _preprocess_kindle_text(review_text)
            features = vectorizer.transform([cleaned])
            prediction = model.predict(features)

            mapping = {0: "Positive", 1: "Neutral", 2: "Negative"}
            return mapping.get(int(prediction[0]), "Unknown")

        except Exception as e:
            raise CustomException(e, sys)


class KindleData:
    def __init__(self, review: str):
        self.review = review

    def get_raw_text(self) -> str:
        return self.review
