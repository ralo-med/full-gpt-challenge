# Home
import streamlit as st
from utils import setup_sidebar, save_settings_to_session
import os

st.set_page_config(
    page_title="Full-Stack GPT-4-Turbo App",
    page_icon="🤖",
    layout="wide",
)

st.title("Full-Stack GPT-4-Turbo App")

with st.sidebar:
    api_key, model_name, temperature = setup_sidebar()
    if api_key:
        save_settings_to_session(api_key, model_name, temperature)
        os.environ["OPENAI_API_KEY"] = api_key

st.markdown(
    """
Welcome to this Full-Stack GPT-4-Turbo application.

Here's a quick rundown of the apps you'll find in the sidebar:

- **Document Chat**: Upload a document and ask questions about it. The app will use embeddings to find the most relevant parts of the document and answer your questions.
- **Quiz Generator**: Upload a document and the app will generate a quiz based on its content. You can then take the quiz and see your score.
- **SiteGPT**: Enter a URL and the app will create a chatbot that can answer questions about the site's content.
- **OpenAI Assistants**: A more advanced chatbot that can use tools to answer your questions.
"""
)
