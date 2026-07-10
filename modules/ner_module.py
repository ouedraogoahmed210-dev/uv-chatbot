"""
Module NER personnalise du chatbot FAQ UV-BF (etape 2 du projet).

On definit ici des entites propres au domaine de l'UV-BF, en plus des
entites classiques que spaCy sait deja reconnaitre (personne, lieu, etc).

Deux approches sont combinees :
1. Un EntityRuler spaCy, qui repose sur des listes de mots ou d'expressions
   du vocabulaire metier (plateformes, services, documents, filieres...).
2. Un composant regex fait maison, pour les entites qui suivent un format
   precis : matricule etudiant, email, periode, montant en FCFA.

Le vocabulaire ci-dessous a ete verifie avec des informations reelles sur
l'UV-BF (site officiel uv.bf, foire aux questions officielle).
"""
import re
import spacy
from spacy.language import Language

# Vocabulaire "plateformes numeriques" de l'UV-BF
PLATEFORME_TERMS = [
    "plateforme de formation", "plateforme d'evaluation", "portail etudiant",
    "portail de candidature", "portail d'inscription", "espace numerique de travail",
    "ent", "e-learning", "site officiel", "espace etudiant", "espace de cours",
]

# Vocabulaire "services administratifs et pedagogiques"
SERVICE_TERMS = [
    "inscription", "reinscription", "frais d'inscription", "frais de scolarite",
    "bourse", "bourse d'etudes", "session de rattrapage", "support technique",
    "service de la scolarite", "agence comptable", "emploi du temps",
    "unite d'enseignement", "mot de passe", "matricule etudiant",
    "quotite", "ine", "subvention", "assistance technique",
]

# Vocabulaire "documents administratifs"
DOCUMENT_TERMS = [
    "attestation de scolarite", "certificat de scolarite", "diplome",
    "releve de notes", "piece d'identite", "certificat medical",
    "justificatif", "dossier de candidature", "recu de paiement", "quittance",
]

# Filieres reellement proposees par l'UV-BF (BTS, licence, master)
FILIERE_TERMS = [
    "fdia", "fouille de donnees et intelligence artificielle",
    "e-agriculture", "forensic", "geomatique", "robotique et automatismes",
    "sante numerique", "science numerique", "sciences fondamentales", "tic",
    "genie logiciel", "master", "licence", "bts",
]

# Lieux lies a l'UV-BF (siege et espaces numeriques ouverts en region)
LIEU_TERMS = [
    "ouagadougou", "ouaga 2000", "bobo-dioulasso", "campus",
    "espace numerique ouvert", "centre d'examen",
]

# Organismes officiels lies a la reconnaissance des diplomes ou au financement
ORGANISME_TERMS = [
    "cames", "uv-bf", "citadel", "boad", "ministere de l'enseignement superieur",
]

# Moyens de paiement mobile courants au Burkina Faso
PAIEMENT_TERMS = [
    "orange money", "moov money", "mobile money", "tresor money",
]

# Programmes et dispositifs specifiques aux etudiants (ordinateur, SIM, communautes)
PROGRAMME_TERMS = [
    "1e1o", "un etudiant un ordinateur", "un etudiant, un ordinateur",
    "campus faso", "carte sim", "communaute virtuelle", "point focal",
    "espaces numeriques ouverts", "groupe whatsapp",
]


def _make_patterns(label, terms):
    """Transforme une liste de termes en patterns spaCy pour l'EntityRuler.
    Chaque terme est decoupe en mots, spaCy comparera token par token."""
    patterns = []
    for term in terms:
        patterns.append({"label": label, "pattern": [{"LOWER": tok} for tok in term.split()]})
    return patterns


ENTITY_RULER_PATTERNS = (
    _make_patterns("PLATEFORME", PLATEFORME_TERMS)
    + _make_patterns("SERVICE", SERVICE_TERMS)
    + _make_patterns("DOCUMENT", DOCUMENT_TERMS)
    + _make_patterns("FILIERE", FILIERE_TERMS)
    + _make_patterns("LIEU", LIEU_TERMS)
    + _make_patterns("ORGANISME", ORGANISME_TERMS)
    + _make_patterns("PAIEMENT", PAIEMENT_TERMS)
    + _make_patterns("PROGRAMME", PROGRAMME_TERMS)
)

# Entites structurees, reconnues par expression reguliere plutot que par mot-cle
REGEX_PATTERNS = {
    "MATRICULE": re.compile(r"\b[A-Za-z]{2,6}[-/]?\d{4}[-/]?\d{3,6}\b"),
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PERIODE": re.compile(
        r"\b(\d{1,2}\s+)?(janvier|fevrier|mars|avril|mai|juin|juillet|aout|"
        r"septembre|octobre|novembre|decembre)(\s+\d{4})?\b"
        r"|\b(premier|deuxieme|1er|2e)\s+semestre\b"
        r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        flags=re.IGNORECASE,
    ),
    "MONTANT": re.compile(r"\b\d{1,3}(?:[ .]?\d{3})*\s?(?:fcfa|f\s?cfa)\b", re.IGNORECASE),
}


@Language.component("regex_entity_matcher")
def regex_entity_matcher(doc):
    """Composant spaCy personnalise. Cherche les entites structurees par
    regex et les ajoute a celles deja trouvees par l'EntityRuler, sans
    creer de chevauchement (priorite aux entites deja posees)."""
    new_ents = list(doc.ents)
    taken_char_spans = [(e.start_char, e.end_char) for e in doc.ents]

    for label, pattern in REGEX_PATTERNS.items():
        for match in pattern.finditer(doc.text):
            start_char, end_char = match.start(), match.end()
            overlap = any(not (end_char <= s or start_char >= e) for s, e in taken_char_spans)
            if overlap:
                continue
            span = doc.char_span(start_char, end_char, label=label, alignment_mode="expand")
            if span is not None:
                new_ents.append(span)
                taken_char_spans.append((start_char, end_char))

    try:
        doc.ents = sorted(new_ents, key=lambda s: s.start_char)
    except ValueError:
        # en cas de chevauchement residuel malgre la verification ci-dessus
        doc.ents = spacy.util.filter_spans(new_ents)
    return doc


def build_custom_nlp(base_model="fr_core_news_sm"):
    """Construit le pipeline spaCy complet : modele francais de base,
    puis EntityRuler (vocabulaire metier), puis composant regex."""
    try:
        nlp = spacy.load(base_model)
    except OSError:
        # si le modele francais n'est pas installe, on utilise un pipeline
        # vide (moins precis pour la tokenisation, mais fonctionnel)
        # installation recommandee : python -m spacy download fr_core_news_sm
        nlp = spacy.blank("fr")

    if "entity_ruler" not in nlp.pipe_names:
        if "ner" in nlp.pipe_names:
            ruler = nlp.add_pipe("entity_ruler", before="ner")
        else:
            ruler = nlp.add_pipe("entity_ruler")
        ruler.add_patterns(ENTITY_RULER_PATTERNS)

    if "regex_entity_matcher" not in nlp.pipe_names:
        nlp.add_pipe("regex_entity_matcher", last=True)

    return nlp


class CustomNER:
    """Classe utilisee par le chatbot pour extraire les entites d'un texte."""

    def __init__(self, base_model="fr_core_news_sm"):
        self.nlp = build_custom_nlp(base_model)

    def extract_entities(self, text):
        """Retourne un dictionnaire {LABEL: [valeurs trouvees]}."""
        doc = self.nlp(text)
        entities = {}
        for ent in doc.ents:
            entities.setdefault(ent.label_, []).append(ent.text)
        return entities

    def extract_entities_list(self, text):
        """Retourne une liste de tuples (texte, label), pratique pour l'affichage."""
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]


if __name__ == "__main__":
    ner = CustomNER()
    samples = [
        "Comment obtenir mon attestation de scolarite sur le portail etudiant ?",
        "Mon matricule est UVBF-2024-00123, mon compte sur l'ENT est bloque.",
        "Je veux m'inscrire au master FDIA a Ouagadougou avant le 15 octobre 2026.",
        "Contactez-moi a jean.dupont@uvbf.bf pour la bourse de 150000 FCFA.",
        "J'ai paye mes frais de scolarite avec Orange Money, le diplome est reconnu par le CAMES ?",
    ]
    for s in samples:
        print(s)
        print(" ->", ner.extract_entities_list(s))
        print()
