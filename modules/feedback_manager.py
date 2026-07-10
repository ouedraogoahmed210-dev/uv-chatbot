"""
Etape 6 du projet : apprentissage continu via un mecanisme de feedback.

Chaque interaction avec le chatbot est enregistree dans un fichier CSV
(question posee, intention predite, confiance, FAQ trouvee, similarite,
entites detectees). L'etudiant peut ensuite noter la reponse comme utile
ou pas utile. Les interactions mal notees ou a faible confiance sont
signalees comme candidates a un futur reentrainement du modele.
"""
import os
import csv
import json
from datetime import datetime

LOG_COLUMNS = [
    "timestamp", "user_question", "intent", "intent_confidence",
    "matched_faq_id", "matched_question", "similarity", "low_confidence",
    "entities_detected", "feedback",
]

FEEDBACK_POSITIF = "utile"
FEEDBACK_NEGATIF = "pas_utile"


class FeedbackManager:
    def __init__(self, log_path):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        if not os.path.exists(log_path):
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(LOG_COLUMNS)

    def log_interaction(self, user_question, bot_result, feedback=""):
        """Enregistre une interaction et renvoie son identifiant de ligne."""
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                user_question,
                bot_result.get("intent"),
                bot_result.get("intent_confidence"),
                bot_result.get("matched_faq_id"),
                bot_result.get("matched_question"),
                bot_result.get("similarity"),
                bot_result.get("low_confidence"),
                json.dumps(bot_result.get("entities_detected", {}), ensure_ascii=False),
                feedback,
            ])
        return self._count_rows() - 1

    def update_feedback(self, row_id, feedback):
        """Met a jour la colonne feedback d'une interaction deja enregistree."""
        rows = self._read_all()
        if 0 <= row_id < len(rows):
            rows[row_id]["feedback"] = feedback
            self._write_all(rows)

    def _read_all(self):
        with open(self.log_path, encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _write_all(self, rows):
        with open(self.log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def _count_rows(self):
        with open(self.log_path, encoding="utf-8") as f:
            return sum(1 for _ in f)

    def get_stats(self):
        """Calcule les statistiques utilisees par le tableau de bord."""
        rows = self._read_all()
        total = len(rows)
        if total == 0:
            return {"total": 0}

        intents_count = {}
        negative_feedback = []
        low_conf_count = 0
        positive, negative = 0, 0

        for r in rows:
            intents_count[r["intent"]] = intents_count.get(r["intent"], 0) + 1
            if r["low_confidence"] == "True":
                low_conf_count += 1
            if r["feedback"] == FEEDBACK_POSITIF:
                positive += 1
            elif r["feedback"] == FEEDBACK_NEGATIF:
                negative += 1
                negative_feedback.append(r["user_question"])

        return {
            "total": total,
            "intents_distribution": intents_count,
            "low_confidence_count": low_conf_count,
            "positive_feedback": positive,
            "negative_feedback": negative,
            "questions_to_review": negative_feedback,
        }

    def export_review_candidates(self, out_path):
        """Exporte les questions mal comprises pour un futur reentrainement."""
        rows = self._read_all()
        candidates = [r for r in rows if r["feedback"] == FEEDBACK_NEGATIF or r["low_confidence"] == "True"]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            writer.writeheader()
            writer.writerows(candidates)
        return len(candidates)
