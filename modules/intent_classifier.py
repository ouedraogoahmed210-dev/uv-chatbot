"""
Module de classification d'intentions pour le chatbot FAQ UV-BF.

Étape 3 du Projet 8 : Entraînement d'un modèle pour identifier le type de
question posée par l'étudiant (intention).

Approche : TF-IDF (n-grammes de mots + caractères pour robustesse au français
et aux fautes de frappe) + SVM linéaire calibré pour obtenir des probabilités
de confiance, avec validation croisée pour l'évaluation.
"""
import os
import re
import unicodedata
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report

def normalize_text(text: str) -> str:
    """Nettoie et normalise le texte : minuscules, accents conservés (français),
    suppression de la ponctuation superflue."""
    text = text.strip().lower()
    text = re.sub(r"[^\w\sàâäéèêëïîôöùûüÿçœæ']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class IntentClassifier:
    """Classifieur d'intentions basé sur TF-IDF + SVM linéaire calibré."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            preprocessor=normalize_text,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self.model = CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced"), cv=3
        )
        self.classes_ = None

    def train(self, csv_path: str, evaluate: bool = True):
        df = pd.read_csv(csv_path)
        X_text, y = df["text"].tolist(), df["intent"].tolist()

        X_vec = self.vectorizer.fit_transform(X_text)

        if evaluate:
            X_train, X_test, y_train, y_test = train_test_split(
                X_vec, y, test_size=0.25, random_state=42, stratify=y
            )
            eval_model = CalibratedClassifierCV(
                LinearSVC(C=1.0, class_weight="balanced"), cv=3
            )
            eval_model.fit(X_train, y_train)
            preds = eval_model.predict(X_test)
            report = classification_report(y_test, preds, zero_division=0)
            scores = cross_val_score(self.model, X_vec, y, cv=3)
            print("=== Rapport d'évaluation (hold-out 25%) ===")
            print(report)
            print(f"Score moyen en validation croisée (3-fold) : {scores.mean():.3f}")

        # Entraînement final sur toutes les données disponibles
        self.model.fit(X_vec, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, text: str):
        """Retourne (intention_predite, confiance, top_3_intentions)"""
        X_vec = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X_vec)[0]
        classes = self.model.classes_
        ranked = sorted(zip(classes, proba), key=lambda x: -x[1])
        intent, confidence = ranked[0]
        return intent, float(confidence), ranked[:3]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "model": self.model,
                     "classes": self.classes_}, path)

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        clf = cls()
        clf.vectorizer = data["vectorizer"]
        clf.model = data["model"]
        clf.classes_ = data["classes"]
        return clf


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clf = IntentClassifier()
    clf.train(os.path.join(base, "data", "intents_training.csv"))
    clf.save(os.path.join(base, "models", "intent_classifier.joblib"))

    tests = [
        "Je veux savoir comment obtenir mon diplome",
        "mot de passe oublie aide moi",
        "quand debute la rentree",
        "le site plante tout le temps",
    ]
    for t in tests:
        intent, conf, top3 = clf.predict(t)
        print(f"'{t}' -> {intent} (confiance={conf:.2f}) | top3={top3}")
