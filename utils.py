import streamlit as st
import os
from dotenv import load_dotenv


def setup_sidebar():
    """
    사이드바를 설정하고 st.session_state를 사용해 상태를 관리합니다.
    개발 모드에서는 .env 파일의 API 키를 우선적으로 사용합니다.
    """
    load_dotenv()

    # ---- 위젯 렌더링 전 세션 상태 초기화 (권장 패턴) ----
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "model_name" not in st.session_state:
        st.session_state.model_name = "gpt-4.1-nano"
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.1

    # 개발 모드일 경우 .env에서 API 키를 우선적으로 로드
    dev_mode = os.getenv("DEV_MODE", "False").lower() == "true"
    if dev_mode and os.getenv("OPENAI_API_KEY"):
        st.session_state.api_key = os.getenv("OPENAI_API_KEY")

    # ---- 사이드바 위젯 렌더링 ----
    st.sidebar.header("OpenAI API Key")
    if dev_mode and st.session_state.api_key:
        st.sidebar.success("✅ 개발 모드: API 키를 로드했습니다.")
        st.sidebar.text_input(
            "API 키 (.env)",
            value=st.session_state.api_key,
            type="password",
            disabled=True,
        )
    else:
        if dev_mode:
            st.sidebar.warning("개발 모드이지만 .env에 API 키가 없습니다.")
        st.sidebar.text_input(
            "API 키를 입력하세요.",
            type="password",
            key="api_key",
            help="API 키는 세션이 지속되는 동안 저장됩니다.",
        )

    st.sidebar.header("모델 설정")
    st.sidebar.selectbox(
        "모델을 선택하세요.",
        ("gpt-4.1-nano", "gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"),
        key="model_name",
    )
    st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        step=0.1,
        key="temperature",
        value=st.session_state.temperature,  # 세션 상태의 값을 기본값으로 사용
    )

    # st.session_state에서 직접 값을 반환합니다.
    return (
        st.session_state.api_key,
        st.session_state.model_name,
        st.session_state.temperature,
    )


def save_settings_to_session(api_key, model_name, temperature):
    """
    이 함수는 이제 위젯 키 사용으로 인해 중복됩니다.
    호환성을 위해 남겨두지만 아무 작업도 수행하지 않습니다.
    """
    pass
