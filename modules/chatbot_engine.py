"""
Moteur principal du chatbot FAQ UV-BF.

Ce module gere deux etapes du projet :
- etape 4, le matching intelligent : trouver la question de la FAQ la plus
  proche de la question posee par l'etudiant, en s'appuyant sur l'intention
  detectee ;
- etape 5, la generation de reponse : remplir le template de reponse trouve
  avec les entites de la FAQ, puis le personnaliser avec les entites
  detectees dans la question de l'utilisateur (matricule, email, etc).

Chaque FAQ peut avoir plusieurs reformulations (variantes) en plus de sa
question principale. Le moteur vectorise toutes ces reformulations, ce qui
rend le matching beaucoup plus robuste face aux differentes facons de poser
la meme question, plutot que de se limiter a une seule phrase par FAQ.
"""
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from modules.intent_classifier import IntentClassifier, normalize_text
from modules.ner_module import CustomNER

INTENT_CONFIDENCE_THRESHOLD = 0.28
SIMILARITY_THRESHOLD = 0.15
SIMILARITY_HIGH_THRESHOLD = 0.35

# Numero d'assistance propose quand le chatbot ne comprend pas la question
NUMERO_ASSISTANCE = "50 54 35 35"


class ChatbotEngine:
    def __init__(self, faq_path, intent_model_path, spacy_base_model="fr_core_news_sm"):
        with open(faq_path, encoding="utf-8") as f:
            self.faq = json.load(f)

        self.intent_classifier = IntentClassifier.load(intent_model_path)
        self.ner = CustomNER(base_model=spacy_base_model)

        # on construit un corpus etendu : question principale + variantes,
        # chaque ligne du corpus pointant vers l'index de sa FAQ d'origine
        corpus_texts = []
        self.corpus_faq_index = []
        for i, item in enumerate(self.faq):
            corpus_texts.append(item["question"])
            self.corpus_faq_index.append(i)
            for variante in item.get("variantes", []):
                corpus_texts.append(variante)
                self.corpus_faq_index.append(i)

        self.vectorizer = TfidfVectorizer(preprocessor=normalize_text, ngram_range=(1, 2))
        self.corpus_matrix = self.vectorizer.fit_transform(corpus_texts)

    def _match_faq(self, user_text, intent):
        """Cherche la question de la FAQ la plus proche du texte de
        l'utilisateur, en comparant a la question principale ET a toutes
        ses variantes. On garde la meilleure similarite obtenue pour chaque
        FAQ. Les FAQ dont l'intention correspond a celle detectee recoivent
        un bonus de score pour etre favorisees en cas d'egalite."""
        user_vec = self.vectorizer.transform([user_text])
        sims_par_ligne = cosine_similarity(user_vec, self.corpus_matrix)[0]

        best_sim_par_faq = [0.0] * len(self.faq)
        for ligne_idx, faq_idx in enumerate(self.corpus_faq_index):
            if sims_par_ligne[ligne_idx] > best_sim_par_faq[faq_idx]:
                best_sim_par_faq[faq_idx] = sims_par_ligne[ligne_idx]

        boosted = list(best_sim_par_faq)
        for i, item in enumerate(self.faq):
            if item["intent"] == intent:
                boosted[i] += 0.20

        best_idx = max(range(len(self.faq)), key=lambda i: boosted[i])
        return self.faq[best_idx], float(best_sim_par_faq[best_idx]), boosted

    @staticmethod
    def _fill_template(template, entities):
        """Remplace les {PLACEHOLDER} du template par la valeur correspondante
        dans le dictionnaire d'entites de la FAQ."""
        def replacer(match):
            key = match.group(1)
            return entities.get(key, match.group(0))
        return re.sub(r"\{([A-Z0-9_]+)\}", replacer, template)

    def _personalize_with_user_entities(self, response, user_entities):
        """Ajoute une phrase de personnalisation si l'utilisateur a fourni
        une entite exploitable dans sa question (matricule, email)."""
        notes = []
        if "MATRICULE" in user_entities:
            notes.append("J'ai bien note votre matricule " + user_entities["MATRICULE"][0] + ".")
        if "EMAIL" in user_entities:
            notes.append("Une confirmation peut vous etre envoyee a " + user_entities["EMAIL"][0] + ".")
        if notes:
            response += "\n\n" + " ".join(notes)
        return response

    def get_response(self, user_text):
        """Point d'entree principal : prend le texte de l'utilisateur et
        renvoie un dictionnaire avec la reponse et les details de traitement
        (intention, entites, score de confiance, FAQ correspondante)."""
        intent, intent_confidence, top3_intents = self.intent_classifier.predict(user_text)
        matched_faq, similarity, _ = self._match_faq(user_text, intent)
        user_entities = self.ner.extract_entities(user_text)

        # La similarite avec la FAQ est le signal le plus fiable : si elle est
        # deja tres bonne, on fait confiance au matching meme si le
        # classifieur d'intention hesite un peu (l'intention n'est qu'un
        # coup de pouce, pas la source de verite). On ne declenche le doute
        # que si la similarite est franchement faible, ou si elle est
        # moyenne ET que l'intention est peu sure en meme temps.
        low_confidence = (
            similarity < SIMILARITY_THRESHOLD
            or (similarity < SIMILARITY_HIGH_THRESHOLD and intent_confidence < INTENT_CONFIDENCE_THRESHOLD)
        )

        if low_confidence:
            suggestions = [item["question"] for item in self.faq if item["intent"] == intent][:3]
            response_text = (
                "Je ne suis pas certain d'avoir bien compris votre question. "
                "Pourriez-vous la reformuler ? Voici quelques questions frequentes "
                "qui pourraient vous interesser :\n- " + "\n- ".join(suggestions)
                + "\n\nSi vous ne trouvez pas votre reponse, contactez l'assistance "
                "de l'UV-BF au " + NUMERO_ASSISTANCE + "."
            )
        else:
            response_text = self._fill_template(matched_faq["reponse"], matched_faq["entities"])
            response_text = self._personalize_with_user_entities(response_text, user_entities)

        return {
            "response": response_text,
            "intent": intent,
            "intent_confidence": round(intent_confidence, 3),
            "matched_faq_id": matched_faq["id"],
            "matched_question": matched_faq["question"],
            "similarity": round(similarity, 3),
            "entities_detected": user_entities,
            "low_confidence": low_confidence,
            "top3_intents": [(i, round(c, 3)) for i, c in top3_intents],
        }
