# Home
import streamlit as st
from utils import setup_sidebar, save_settings_to_session
import os

st.set_page_config(
    page_title="GPT Challenge Home",
    page_icon="🤖",
    layout="wide",
)

st.title("GPT Challenge Home")

with st.sidebar:
    api_key, model_name, temperature = setup_sidebar()
    if api_key and api_key.strip():
        save_settings_to_session(api_key, model_name, temperature)
        os.environ["OPENAI_API_KEY"] = api_key

st.markdown(
    """
    Welcome to GPT Challenge!
    
    Choose an application from the sidebar:
    - **Document Chat**: Upload documents and chat with AI about them
    - **Quiz Generator**: Generate quizzes from documents or Wikipedia articles
    - **SiteGPT**: Enter a URL and the app will create a chatbot that can answer questions about the site's content.
    - **OpenAI Assistants**: A more advanced chatbot that can use tools to answer your questions.
"""
)
