import streamlit as st
import os
from langchain.chat_models import ChatOpenAI

def setup_sidebar():
    """공통 사이드바 설정"""
    st.write("설정")
    
    # 실제 개발 환경인지 확인 (환경변수에 DEV_MODE가 설정되어 있는지 확인)
    is_actual_dev_mode = os.getenv("DEV_MODE", "false") == "true"
    
    # 실제 개발 모드일 때만 환경변수 사용
    if is_actual_dev_mode:
        st.success("개발 모드: 환경변수 API 키 사용중")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("❌ 개발 모드에서 OPENAI_API_KEY 환경변수가 설정되지 않았습니다!")
            return None, None, None
    else:
        # 배포 모드: 사용자 입력 받기
        st.info("API 키를 입력해주세요.")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="OpenAI API 키를 입력하세요"
        )
        st.write("API 키는 저장되지 않습니다.")
        
        # API 키 유효성 검사를 여기서 즉시 수행
        if not validate_api_key(api_key):
            st.error("❌ Please enter your OpenAI API key!")
            return None, None, None

    # 모델 선택
    model_name = st.selectbox(
        "모델 선택",
        ["gpt-4.1-nano", "gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
        index=0,
        help="사용할 OpenAI 모델을 선택하세요"
    )
    
    # 온도 설정
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.1,
        step=0.1,
        help="응답의 창의성을 조절합니다 (낮을수록 일관성, 높을수록 창의성)"
    )
    
    st.divider()
    
    # 소스코드 링크를 맨 아래로 이동
    st.write("소스코드")
    st.markdown("[GitHub Repository](https://github.com/ralo-med/full-gpt-challenge)")
    st.markdown("[Live App](https://full-gpt-challenge-wchmjbuyozz8xhnrgiatmb.streamlit.app/)")
    
    return api_key, model_name, temperature

def validate_api_key(api_key):
    """API 키 유효성 검사"""
    if api_key:
        # API 키가 입력되었으면 환경변수에 설정
        os.environ["OPENAI_API_KEY"] = api_key
        
        # API 키 유효성 확인 (가장 싼 모델, 최소 토큰)
        try:
            test_llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0,
                max_tokens=1
            )
            test_llm.invoke("hi")
            return True
        except Exception:
            st.error("❌ Invalid API key. Please check your OpenAI API key.")
            st.markdown(
                """
                Get your API key from [OpenAI Platform](https://platform.openai.com/account/api-keys)
                """
            )
            st.stop()
    else:
        # API 키가 비어있으면 환경변수에서 삭제
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        return False

def save_settings_to_session(api_key, model_name, temperature):
    """설정을 세션 상태에 저장"""
    st.session_state["api_key"] = api_key
    st.session_state["model_name"] = model_name
    st.session_state["temperature"] = temperature



def create_llm(model_name, temperature, callbacks=None):
    """LLM 인스턴스 생성"""
    if callbacks is None:
        callbacks = []
    
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        streaming=True,
        callbacks=callbacks
    ) 

def check_api_key():
    """API 키가 설정되었는지 확인하고 오류 메시지 표시"""
    if not os.getenv("OPENAI_API_KEY"):
        st.error("❌ OpenAI API key is not set! Please set API key in sidebar.")
        return False
    return True

def setup_page_with_sidebar(title, icon="🔥", layout="wide"):
    """페이지 설정과 사이드바를 한 번에 설정"""
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout=layout,
    )
    
    # API 키 체크
    check_api_key()
    
    # 사이드바 설정
    api_key, model_name, temperature = setup_sidebar()
    
    # API 키가 있을 때만 설정을 세션 상태에 저장
    if api_key:
        save_settings_to_session(api_key, model_name, temperature)
    
    return api_key, model_name, temperature

def create_llm_safe(model_name, temperature, callbacks=None):
    """안전한 LLM 생성 (API 키 체크 포함)"""
    if not os.getenv("OPENAI_API_KEY"):
        return None
    return create_llm(model_name, temperature, callbacks)
