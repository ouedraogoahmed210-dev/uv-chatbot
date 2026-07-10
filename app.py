"""
Interface Streamlit du chatbot FAQ intelligent UV-BF.

Deux onglets :
- Chatbot : interface de conversation, avec l'intention et les entites
  detectees affichees en detail, et des boutons de feedback. Quand le
  chatbot n'est pas sur de sa reponse, il oriente l'etudiant vers le
  numero d'assistance de l'UV-BF au lieu d'inventer une reponse.
- Tableau de bord : statistiques sur les interactions passees
  (etape 6 du projet, apprentissage continu).
"""
import os
import pandas as pd
import streamlit as st
import plotly.express as px

from modules.chatbot_engine import ChatbotEngine
from modules.feedback_manager import FeedbackManager, FEEDBACK_POSITIF, FEEDBACK_NEGATIF

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAQ_PATH = os.path.join(BASE_DIR, "data", "faq.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "intent_classifier.joblib")
LOG_PATH = os.path.join(BASE_DIR, "feedback", "interactions_log.csv")

st.set_page_config(page_title="Chatbot FAQ UV-BF", layout="wide")


@st.cache_resource
def load_engine():
    return ChatbotEngine(FAQ_PATH, MODEL_PATH)


@st.cache_resource
def load_feedback_manager():
    return FeedbackManager(LOG_PATH)


def render_chat_tab(engine, fbm):
    st.subheader("Assistant FAQ des etudiants de l'UV-BF")
    st.caption(
        "Posez une question sur l'inscription, les cours, les examens, "
        "les bourses, les frais, le mot de passe, etc."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "meta" in msg:
                meta = msg["meta"]
                col1, col2, col3 = st.columns([1, 1, 6])
                with col1:
                    if st.button("Utile", key="up_" + str(i)):
                        fbm.update_feedback(meta["log_id"], FEEDBACK_POSITIF)
                        st.toast("Merci pour votre retour.")
                with col2:
                    if st.button("Pas utile", key="down_" + str(i)):
                        fbm.update_feedback(meta["log_id"], FEEDBACK_NEGATIF)
                        st.toast("Merci, nous allons ameliorer cette reponse.")

    user_input = st.chat_input("Ecrivez votre question ici...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        result = engine.get_response(user_input)
        log_id = fbm.log_interaction(user_input, result)
        result["log_id"] = log_id

        with st.chat_message("assistant"):
            st.markdown(result["response"])
        st.session_state.messages.append({
            "role": "assistant", "content": result["response"], "meta": result
        })
        st.rerun()


def render_dashboard_tab(fbm):
    st.subheader("Tableau de bord analytics")
    stats = fbm.get_stats()

    if stats["total"] == 0:
        st.info("Aucune interaction enregistree pour le moment. "
                "Discutez avec le chatbot pour generer des statistiques.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total d'interactions", stats["total"])
    col2.metric("Faible confiance", stats["low_confidence_count"])
    col3.metric("Feedback utile", stats["positive_feedback"])
    col4.metric("Feedback pas utile", stats["negative_feedback"])

    df_intents = pd.DataFrame(
        list(stats["intents_distribution"].items()), columns=["Intention", "Nombre"]
    ).sort_values("Nombre", ascending=False)
    fig = px.bar(df_intents, x="Intention", y="Nombre",
                 title="Repartition des intentions posees par les etudiants",
                 color="Nombre", color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

    if stats["questions_to_review"]:
        st.warning("Questions mal comprises (feedback negatif) a revoir :")
        st.write(stats["questions_to_review"])
        if st.button("Exporter les questions a revoir (CSV)"):
            out_path = os.path.join(BASE_DIR, "feedback", "a_reviser.csv")
            n = fbm.export_review_candidates(out_path)
            st.success(str(n) + " interactions exportees vers " + out_path)

    with st.expander("Journal complet des interactions"):
        st.dataframe(pd.read_csv(fbm.log_path))


def main():
    st.sidebar.title("Chatbot FAQ UV-BF")
    st.sidebar.markdown(
        "Chatbot avec NER personnalise et classification d'intentions, "
        "pour repondre aux questions frequentes des etudiants de l'UV-BF."
    )
    tab = st.sidebar.radio("Navigation", ["Chatbot", "Tableau de bord"])

    engine = load_engine()
    fbm = load_feedback_manager()

    if tab == "Chatbot":
        render_chat_tab(engine, fbm)
    else:
        render_dashboard_tab(fbm)


if __name__ == "__main__":
    main()
