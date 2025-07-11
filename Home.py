# Home
import streamlit as st
from utils import setup_sidebar, validate_api_key, save_settings_to_session

st.set_page_config(
    page_title="GPT Challenge Home",
    page_icon="🤖",
    layout="wide",
)

# 사이드바에서 설정
with st.sidebar:
    # API 키 유효성 검사를 먼저 수행
    api_key, model_name, temperature = setup_sidebar()
    
    # API 키 유효성 검사
    if not validate_api_key(api_key):
        st.error("❌ Please enter your OpenAI API key!")
        st.stop()
    else:
        # 설정을 세션 상태에 저장
        save_settings_to_session(api_key, model_name, temperature)

# 메인 콘텐츠
st.title("GPT Challenge Home")
st.markdown(
    """
    Welcome to GPT Challenge!
    
    Choose an application from the sidebar:
    - **Document Chat**: Upload documents and chat with AI about them
    - **Quiz Generator**: Generate quizzes from documents or Wikipedia articles
    """
)
