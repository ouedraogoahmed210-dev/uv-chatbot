# Chatbot FAQ intelligent - UV-BF (Projet No 8)

Chatbot qui repond aux questions frequentes des etudiants de l'UV-BF. Il
combine un NER personnalise (extraction d'entites du domaine), une
classification d'intentions, un systeme de matching intelligent, une
generation de reponses par templates, et un mecanisme de feedback pour
l'apprentissage continu. Le tout est servi via une interface Streamlit
avec un tableau de bord.

## Correspondance avec les etapes du sujet

| Etape du sujet | Fichier | Description |
|---|---|---|
| 1. Base de FAQ | data/faq.json | 64 questions et reponses reparties en 16 intentions, chacune avec plusieurs variantes de formulation pour un matching plus robuste |
| 2. NER personnalise | modules/ner_module.py | EntityRuler spaCy (PLATEFORME, SERVICE, DOCUMENT, FILIERE, LIEU, ORGANISME, PAIEMENT, PROGRAMME) et composant regex (MATRICULE, EMAIL, PERIODE, MONTANT) |
| 3. Classification d'intentions | modules/intent_classifier.py | TF-IDF (1 a 2 grammes) et SVM lineaire calibre, pres de 190 exemples d'entrainement pour 16 intentions |
| 4. Matching intelligent | modules/chatbot_engine.py | Similarite cosinus TF-IDF entre la question posee et la FAQ, avec bonus pour la coherence d'intention |
| 5. Generation de reponses | modules/chatbot_engine.py | Templates avec placeholders remplis par les entites de la FAQ, puis personnalisation avec les entites detectees chez l'utilisateur |
| 6. Apprentissage continu | modules/feedback_manager.py | Journalisation de toutes les interactions, feedback utile ou pas utile, export des cas a reviser |
| Interface | app.py | Chat Streamlit et tableau de bord analytics (Plotly) |

## Installation

```bash
cd projet8_chatbot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download fr_core_news_sm
```

Le modele fr_core_news_sm ameliore la tokenisation en francais. S'il n'est
pas telecharge, le code bascule automatiquement sur un pipeline vide
(spacy.blank("fr")), fonctionnel mais moins precis.

## Entrainement du modele d'intentions

Le modele est deja entraine et fourni dans models/intent_classifier.joblib.
Pour le reentrainer, apres ajout de donnees dans data/intents_training.csv :

```bash
python train_models.py
```

### Faut-il reentrainer apres avoir ajoute des questions dans data/faq.json ?

Non, pas obligatoirement. Il y a deux mecanismes independants dans ce
projet :

- Le matching intelligent (modules/chatbot_engine.py) lit data/faq.json
  et reconstruit sa recherche de similarite a chaque demarrage de
  l'application. Ajouter une question, une reponse, ou des variantes dans
  faq.json est donc pris en compte immediatement, sans script a lancer.
- Le classifieur d'intentions (models/intent_classifier.joblib) est un
  modele separe, entraine a partir de data/intents_training.csv. Il ne
  sert qu'a orienter le matching vers la bonne categorie de questions
  (inscription, examen, bourse, etc). Il faut le reentrainer avec
  train_models.py uniquement dans deux cas : si vous ajoutez une toute
  nouvelle intention qui n'existe pas encore dans les 16 categories
  actuelles, ou si vous remarquez que des questions bien formulees sont
  quand meme mal orientees (dans ce cas, ajoutez quelques exemples de la
  formulation posee dans intents_training.csv puis relancez
  train_models.py).

En pratique, la meilleure demarche pour enrichir la FAQ est d'ajouter des
"variantes" (differentes facons de poser la meme question) a chaque
entree de faq.json plutot que de multiplier les entrees quasi identiques :
cela ameliore directement le matching, sans toucher au classifieur.

## Lancer l'application

```bash
streamlit run app.py
```

## Déployer sur Streamlit Community Cloud (lien à partager)

1. Créez un dépôt GitHub public et poussez-y le contenu de ce dossier
   (app.py, modules/, data/, models/, requirements.txt, .gitignore).
2. Allez sur share.streamlit.io et connectez-vous avec votre compte GitHub.
3. Cliquez sur "New app", choisissez le dépôt, la branche, et indiquez
   app.py comme fichier principal.
4. Dans "Advanced settings" (ou dans Settings > General si l'app existe
   déjà), réglez la version de Python sur 3.11 ou 3.12.
5. Streamlit installe automatiquement les paquets listés dans
   requirements.txt, y compris le modèle spaCy français, puis démarre
   l'application. Vous obtenez un lien du type
   https://votre-app.streamlit.app à partager avec le groupe et le
   professeur.

Points d'attention pour le déploiement :
- Le modèle fr_core_news_sm est installé directement via son wheel dans
  requirements.txt, car "pip install -r requirements.txt" seul
  n'exécute pas la commande "python -m spacy download".
- La version de Python doit être fixée sur 3.11 ou 3.12 depuis les
  paramètres de l'application sur Streamlit Cloud (Settings > Python
  version). spacy 3.7 (requis par le modèle français) n'a pas de wheel
  précompilé pour les versions de Python les plus récentes, ce qui
  provoquerait sinon une erreur de compilation lors de l'installation.
- numpy est fixé sous la version 2.0, et scikit-learn est fixé à la
  version exacte utilisée pour entraîner le modèle. Sans ces
  contraintes, un conflit de version binaire entre numpy et thinc
  (utilisé par spacy) ou un avertissement d'incompatibilité de modèle
  scikit-learn peuvent apparaître.
- Le fichier feedback/interactions_log.csv est recréé automatiquement
  au premier lancement, mais n'est pas persistant entre deux
  redéploiements sur Streamlit Cloud : ce n'est pas une base de
  données permanente, seulement un journal de la session en cours.
- Si l'application reste inactive un moment, Streamlit Cloud la met en
  veille ; elle redémarre automatiquement à la prochaine visite, avec
  quelques secondes de délai.

## Comment ca marche

1. L'etudiant tape une question dans le chat.
2. Le classifieur d'intentions predit la categorie de la question
   (inscription, examen, bourse, etc) avec un score de confiance.
3. Le module NER extrait les entites presentes dans la question, par
   exemple un matricule, un email ou une date.
4. Le moteur de matching cherche la question la plus proche dans la FAQ,
   en priorisant les FAQ de la meme intention.
5. La reponse est generee a partir du template de la FAQ trouvee, avec
   insertion des entites propres a cette FAQ, puis personnalisee si
   l'utilisateur a fourni des informations exploitables.
6. En cas de faible confiance sur l'intention ou la similarite, le chatbot
   le signale, propose des questions frequentes proches, et oriente
   l'etudiant vers le numero d'assistance de l'UV-BF au lieu d'inventer
   une reponse.
7. Chaque echange est journalise. L'etudiant peut noter la reponse comme
   utile ou pas utile. Le tableau de bord permet de suivre la repartition
   des intentions, le taux de reponses a faible confiance, et d'exporter
   les questions mal comprises pour enrichir le futur jeu d'entrainement.

## Numero d'assistance quand le chatbot ne comprend pas

Le chatbot n'essaie jamais d'inventer une reponse. Des que la confiance
sur l'intention ou la similarite avec la FAQ est trop faible, il le dit
clairement, propose 3 questions frequentes proches de la meme categorie,
et ajoute le numero d'assistance de l'UV-BF (50 54 35 35) pour que
l'etudiant puisse joindre quelqu'un directement.

Ce numero est defini une seule fois, via la constante NUMERO_ASSISTANCE
dans modules/chatbot_engine.py, a modifier facilement si le numero change.

## Origine des donnees de la FAQ

Le contenu de data/faq.json combine trois sources :

- Des informations verifiees sur le site officiel uv.bf et sa foire aux
  questions (plateformes numeriques, reconnaissance des diplomes par le
  CAMES, localisation du siege, filieres proposees, CITADEL).
- Des contributions de membres du groupe, avec des informations tres
  concretes et a jour sur les dispositifs specifiques aux etudiants de
  l'UV-BF : le programme "un etudiant, un ordinateur" (1E1O), la carte SIM
  etudiante, les groupes WhatsApp et communautes virtuelles par filiere,
  et les consignes de securite face aux fausses annonces.
- Des verifications complementaires faites sur le web pour recouper ces
  informations avant de les integrer (par exemple le pourcentage de
  subvention du programme 1E1O, qui a varie dans le temps : 60% en 2020,
  80% en octobre 2025 selon les annonces officielles. Le chiffre retenu
  dans la FAQ peut donc evoluer d'une annee a l'autre ; en cas de doute,
  verifiez-le aupres du Point Focal 1E1O de l'UV-BF).

En fusionnant ces sources, deux entrees existantes se contredisaient
legerement sur le deroulement des examens et la gestion des absences
justifiees (une ancienne reponse generique face a une information plus
precise et plus recente apportee par le groupe). Les entrees ont ete
harmonisees pour ne conserver que l'information la plus fiable.

La liste des filieres a egalement ete corrigee en cours de route : une
premiere version listait des filieres a plat (issues d'une source datant
de 2022), alors que le site officiel actuel de l'UV-BF les organise en 3
programmes (Sciences du numerique, Sciences fondamentales, Sciences
transversales), avec des filieres recemment ajoutees (sciences
economiques et de gestion) qui n'apparaissaient pas dans la premiere
version. La reponse precise aussi explicitement qu'une filiere qui n'est
pas proposee par l'UV-BF (MPCI, en realite hebergee par d'autres centres
universitaires publics) ne doit pas etre confondue avec l'offre reelle de
l'UV-BF.

Certaines informations comme les montants exacts des frais de scolarite
changent chaque annee academique. Elles ne sont donc pas donnees en chiffre
fixe dans les reponses, pour eviter de communiquer une information perimee
aux etudiants ; le chatbot renvoie plutot vers le service de la scolarite
ou le site officiel. Avant la soutenance, verifiez les informations les
plus sensibles (montants, pourcentages, numeros de telephone) avec les
administrateurs de l'UV-BF.

## Pour aller plus loin

- Ajouter davantage d'exemples dans data/intents_training.csv pour chaque
  intention, afin d'augmenter la robustesse du classifieur.
- Remplacer le TF-IDF par des embeddings de phrases (par exemple
  sentence-transformers ou CamemBERT) pour un matching semantique plus fin.
- Ajouter un pipeline de reentrainement automatique declenche quand le
  nombre de questions a reviser depasse un seuil.
- Etendre le module NER avec un modele spaCy entraine sur des annotations
  manuelles, en complement des regles et regex actuelles.
- Continuer a enrichir les variantes de chaque FAQ dans data/faq.json au
  fil des vraies questions posees par les etudiants (voir la section
  ci-dessous sur l'ajout de nouvelles donnees).

## Structure du projet

```
projet8_chatbot/
    app.py                     interface Streamlit
    train_models.py            script d'entrainement
    requirements.txt
    data/
        faq.json                base de connaissances
        intents_training.csv    donnees d'entrainement du classifieur
    models/
        intent_classifier.joblib
    feedback/
        interactions_log.csv    genere automatiquement au premier lancement
    modules/
        ner_module.py
        intent_classifier.py
        chatbot_engine.py
        feedback_manager.py
```
