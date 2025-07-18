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

        if st.session_state.api_key:
            # 이미 키가 저장됨 → 메시지만 표시하고, 수정하려면 expander 사용
            st.sidebar.success("✅ API 키가 저장되었습니다.")

            def _update_key():
                st.session_state.api_key = st.session_state.new_api_key or ""

            with st.expander("🔑 API Key 변경"):
                st.text_input(
                    "새 API 키를 입력하세요.",
                    type="password",
                    key="new_api_key",
                    help="입력 후 Enter 를 누르세요.",
                    on_change=_update_key,
                )
        else:
            def _save_first_key():
                st.session_state.api_key = st.session_state.first_api_key or ""

            st.sidebar.text_input(
                "API 키를 입력하세요.",
                type="password",
                key="first_api_key",
                help="세션 동안 보존됩니다.",
                on_change=_save_first_key,
            )

        # API 키 삭제(로그아웃) 버튼
        if st.session_state.api_key:
            def _clear_key():
                st.session_state.api_key = ""
            st.sidebar.button("❌ API 키 삭제", on_click=_clear_key)

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
