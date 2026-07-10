"""
Script d'entraînement du chatbot FAQ UV-BF.
À exécuter une première fois (et après tout ajout de données d'entraînement) :

    python train_models.py
"""
import os
from modules.intent_classifier import IntentClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print("Entraînement du classifieur d'intentions...")
    clf = IntentClassifier()
    clf.train(os.path.join(BASE_DIR, "data", "intents_training.csv"))
    model_path = os.path.join(BASE_DIR, "models", "intent_classifier.joblib")
    clf.save(model_path)
    print(f"Modèle sauvegardé : {model_path}")
    print("\nTerminé. Vous pouvez maintenant lancer : streamlit run app.py")
