import streamlit as st
import os
from dotenv import load_dotenv


def setup_sidebar():

    load_dotenv()

 
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "model_name" not in st.session_state:
        st.session_state.model_name = "gpt-4.1-nano"
    if "temperature" not in st.session_state:
   
        pass  # 기본값은 아래 슬라이더에서 처리

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

        # API Key 입력 / 표시
        if st.session_state.api_key:
            # 이미 저장된 키가 있으면 읽기 전용으로 표시
            st.sidebar.text_input(
                "OpenAI API Key (세션 유지/저장x)",
                value=st.session_state.api_key,
                key="api_key_display",
                type="password",
                disabled=True,
            )

            # 아래쪽에 삭제 버튼
            if st.sidebar.button("❌ API 키 삭제", key="api_key_clear"):
                st.session_state.api_key = ""
                st.rerun()
        else:
            # 최초 입력
            first_key = st.sidebar.text_input(
                "OpenAI API Key 입력",
                key="api_key_input",
                value="",
            )
            if first_key:
                st.session_state.api_key = first_key
                os.environ["OPENAI_API_KEY"] = first_key
                st.rerun()

    st.sidebar.header("모델 설정")
    st.sidebar.selectbox(
        "모델을 선택하세요.",
        ("gpt-4.1-nano", "gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"),
        key="model_name",
    )
    # 슬라이더를 생성하고 반환값을 session_state.temperature 로 자동 반영
    initial_temp = st.session_state.get("temperature", 0.1)
    st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        step=0.1,
        value=initial_temp,
        key="temperature",
    )

    # st.session_state에서 직접 값을 반환합니다.
    if st.session_state.api_key:
        os.environ["OPENAI_API_KEY"] = st.session_state.api_key

    return (
        st.session_state.api_key,
        st.session_state.model_name,
        st.session_state.temperature,
    )


# 호환성을 위해 남겨둔 빈 함수 (이제 필요 없음)

def save_settings_to_session(*args, **kwargs):
    """이전 코드와의 호환성을 위해 유지되는 빈 함수."""
    pass


