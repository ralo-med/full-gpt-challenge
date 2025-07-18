#!/usr/bin/env python3
import os, json, warnings, time
from typing import Dict, List, Any, Tuple

import streamlit as st
from openai import OpenAI, InternalServerError
from requests.exceptions import RequestException

from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader

from utils import setup_sidebar, save_settings_to_session

warnings.filterwarnings(
    "ignore",
    message=r"This package \(`duckduckgo_search`\) has been renamed to `ddgs`!",
    category=RuntimeWarning,
)

os.environ.setdefault(
    "USER_AGENT",
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    ),
)


def wiki_search(inputs: Dict[str, Any]) -> str:
    return WikipediaAPIWrapper().run(inputs["query"])

def ddg_search(inputs: Dict[str, Any]) -> str:
    wrapper = DuckDuckGoSearchAPIWrapper(safesearch="off")
    return wrapper.run(inputs["query"])

def fetch_webpage(inputs: Dict[str, Any]) -> str:
    url = inputs["url"]
    try:
        loader = WebBaseLoader(
            url,
            requests_kwargs={
                "timeout": 15,
                "headers": {"User-Agent": os.environ["USER_AGENT"]},
            },
        )
        docs = loader.load()
    except RequestException as e:
        return f"[SKIPPED] {url} ({e.__class__.__name__}: {e})"
    content = "\n\n".join(doc.page_content for doc in docs)
    cleaned = "\n".join(line.strip() for line in content.splitlines() if line.strip())
    return cleaned if cleaned else f"[EMPTY] {url}"

def save_to_file(inputs: Dict[str, Any]) -> str:
    try:
        content = inputs.get("content", "")
        if not content:
            return "오류: 저장할 내용이 없습니다."
        with open("result.txt", "w", encoding="utf-8") as f:
            f.write(content)
        if os.path.exists("result.txt"):
            file_size = os.path.getsize("result.txt")
            return f"파일 저장 완료: result.txt ({file_size} 바이트)"
        else:
            return "오류: 파일이 생성되지 않았습니다."
    except Exception as e:
        return f"파일 저장 오류: {str(e)}"

functions_map = {
    "wiki_search": wiki_search,
    "ddg_search": ddg_search,
    "fetch_webpage": fetch_webpage,
    "save_to_file": save_to_file,
}


tools = [
    {
        "name": "wiki_search",
        "type": "function",
        "description": "Return a concise summary from Wikipedia for the query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "ddg_search",
        "type": "function",
        "description": "Return DuckDuckGo text search results for the query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "fetch_webpage",
        "type": "function",
        "description": "Download and return raw textual content from a webpage URL.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string", "format": "uri"}},
            "required": ["url"],
            "additionalProperties": False,
        },
    },
    {
        "name": "save_to_file",
        "type": "function",
        "description": "Save provided text to result.txt on disk.",
        "parameters": {
            "type": "object",
            "properties": {"content": {"type": "string"}},
            "required": ["content"],
            "additionalProperties": False,
        },
    },
]

SYSTEM_PROMPT = (
    "You are an advanced research assistant. Your primary goal is to produce a high-quality, comprehensive research report based on the user's query.\n\n"
    "## General Workflow:\n"
    "Your standard process is a 4-step sequence: `wiki_search` -> `ddg_search` -> `fetch_webpage` -> `save_to_file`.\n\n"
    "## CRITICAL INSTRUCTIONS:\n"
    "1. **REACT TO FAILURES (Recovery):** This is your most important instruction. If a tool fails or provides poor results (e.g., `fetch_webpage` returns `[SKIPPED]` or `[EMPTY]`), you MUST NOT proceed blindly. You MUST attempt to recover. For example, use `ddg_search` again to find a new, more reliable URL, then try `fetch_webpage` on that new URL. Your goal is a complete report; do not give up easily.\n"
    "2. **USER COMMANDS FIRST:** If the user gives a direct command (e.g., 'save now', 'search for this'), that command overrides the general workflow. Execute it immediately.\n"
    "3. **NO PREMATURE SUMMARIES:** Do not provide any text summary or answer to the user until you have successfully called `save_to_file`. Your only outputs should be tool calls until the final step.\n"
    "4. **MANDATORY SAVE & COMPREHENSIVE CONTENT:** The `save_to_file` function is the MANDATORY final step. The 'content' for this function must be a detailed, multi-paragraph report synthesizing ALL information you have gathered.\n"
)


def safe_create_responses(client: OpenAI, **kwargs):
    delay = 1.0
    for attempt in range(3):
        try:
            return client.responses.create(**kwargs)
        except InternalServerError:
            if attempt == 2:
                raise
            time.sleep(delay)
            delay *= 2


def render_progress_steps(progress_steps: List[Any], placeholder=None):
    """진행 단계들을 UI에 렌더링하는 함수"""
    container = placeholder.container() if placeholder else st
    for step in progress_steps:
        if isinstance(step, dict) and step.get("type") == "result_dropdown":
            with container.expander(f"📄 단계 {step['step']} 결과 - {step['tool_name']} ({step['tool']})"):
                st.write(f"**사용 도구**: {step['tool']}")
                st.write(f"**도구 설명**: {step['tool_name']}")
                st.write("**결과**:")
                st.text(step['result'])
        elif "🔍 **단계" in step:
            container.info(step)
        elif "✅ **단계" in step:
            container.success(step)
        elif "🎉 **연구 완료**" in step:
            container.success(step)
        elif "🤔 AI assistant" in step:
            container.info(step)
        else:
            container.write(step)


def research_assistant_streaming(
    client: OpenAI,
    user_msg: str,
    history: List[Dict[str, Any]],
    progress_placeholder,
    model_name: str,
) -> Tuple[List[Dict[str, Any]], str]:
    if not history or history[-1].get("content") != user_msg or history[-1].get("role") != "user":
        history.append({"role": "user", "content": user_msg})

    st.session_state.progress_steps = []
    progress_steps = st.session_state.progress_steps
    final_summary = ""

    progress_steps.append("🤔 AI assistant 분석 시작...")
    render_progress_steps(progress_steps, progress_placeholder)

    step_count = 0
    while True: 
        step_count += 1
        resp = safe_create_responses(
            client,
            model=model_name,
            input=history,
            tools=tools,
            temperature=st.session_state.temperature
        )

        response_message = resp.output[0] if resp.output else None
        
        if not response_message or response_message.type != "function_call":
            is_save_done = any(
                isinstance(h, dict) and h.get("name") == "save_to_file"
                for h in history
            )
            # 저장이 완료된 후의 텍스트 응답은 최종 답변으로 간주하고 루프 종료
            if is_save_done:
                break
            
            # 저장이 완료되지 않았는데 텍스트를 생성하면, 규칙을 다시 알려주고 계속 진행하도록 유도
            premature_text = response_message.text.strip() if response_message and hasattr(response_message, "text") else ""
            if premature_text:
                progress_steps.append(f"⚠️ AI가 중간 요약을 시도했습니다. 규칙에 따라 다음 도구를 사용하도록 지시합니다.")
                render_progress_steps(progress_steps, progress_placeholder)
                history.append({"role": "assistant", "content": premature_text})
            
            history.append({
                "role": "user", 
                "content": "You must not generate a text response yet. You must call a tool. Either continue researching with another tool or, if you have sufficient information, call `save_to_file` to create the final report."
            })
            continue # 다음 루프를 돌며 AI가 도구를 사용하도록 강제

        call = response_message
        tool_name = call.name
        tool_descriptions = {
            "wiki_search": "📚 위키백과 검색", "ddg_search": "🔍 DuckDuckGo 웹 검색", 
            "fetch_webpage": "🌐 웹페이지 내용 추출", "save_to_file": "💾 결과를 파일로 저장"
        }
        tool_ui_name = tool_descriptions.get(tool_name, '도구 실행 중')
        
        progress_steps.append(f"🔍 **단계 {step_count}** {tool_ui_name} 중...")
        render_progress_steps(progress_steps, progress_placeholder)
        
        result = functions_map[tool_name](json.loads(call.arguments))
        
        history.extend([call, {"type": "function_call_output", "call_id": call.call_id, "output": result}])
        
        progress_steps.append(f"✅ **단계 {step_count} 완료**: {tool_ui_name}")
        progress_steps.append({"type": "result_dropdown", "step": step_count, "tool": tool_name, "tool_name": tool_ui_name, "result": result})
        render_progress_steps(progress_steps, progress_placeholder)

        if tool_name == "save_to_file":
            final_summary = json.loads(call.arguments).get("content", "")
            progress_steps.append("🎉 **연구 완료**: 파일 저장이 완료되었습니다!")
            render_progress_steps(progress_steps, progress_placeholder)
            # AI의 추가적인 답변 생성 대신, 저장된 요약을 최종 답변으로 설정
            response_message = f"🎯 [최종 연구 요약]\n{final_summary}" if final_summary else "연구를 완료했지만, 요약이 생성되지 않았습니다."
            break

        progress_steps.append("🤔 AI assistant 다음 단계 분석 중...")
        render_progress_steps(progress_steps, progress_placeholder)

    # 최종 답변 생성
    if isinstance(response_message, str):
        final_answer = response_message
    elif hasattr(response_message, 'text') and response_message.text:
        final_answer = response_message.text.strip()
    else:
        final_answer = f"🎯 [최종 연구 요약]\n{final_summary}" if final_summary else "연구를 완료했지만, 요약이 생성되지 않았습니다."
        
    return history, final_answer


st.set_page_config(
    page_title="ResearchGPT",
    page_icon="🔍",
    layout="wide",
)

st.title("ResearchGPT 🔍")
st.markdown("""
웹 검색과 도구를 사용하여 질문에 답변하는 AI 연구 어시스턴트입니다.

**기능:**
- 📚 위키백과 검색
- 🔍 DuckDuckGo 웹 검색
- 🌐 웹페이지 내용 추출
- 💾 결과를 파일로 저장
- ⭐️ 저장이 이루어 지지 않을시 분석이 완료 된것이 아니니 계속 대화를 이어가거나 저장해 달라 하세요!

""")

with st.sidebar:
    st.header("설정")
    api_key, model_name, temperature = setup_sidebar()
    if api_key:
        save_settings_to_session(api_key, model_name, temperature)
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        st.info("사이드바에서 OpenAI API 키를 입력해주세요.")
        st.stop()


    st.header("📁 저장된 파일")
    if os.path.exists("result.txt"):
        try:
            with open("result.txt", "r", encoding="utf-8") as f:
                file_content = f.read()
            st.success("✅ result.txt 저장 완료")
            st.write(f"**파일 크기:** {len(file_content)} 문자")
            st.download_button(
                label="📥 result.txt 다운로드",
                data=file_content,
                file_name="result.txt",
                mime="text/plain"
            )
            if st.button("🗑️ 파일 삭제", disabled=st.session_state.get("processing", False)):
                os.remove("result.txt")
                st.success("파일이 삭제되었습니다!")
                st.rerun()
        except Exception as e:
            st.error(f"파일 읽기 오류: {str(e)}")
    else:
        st.info("📝 저장된 파일이 없습니다.")
        st.write("AI가 분석 후 파일을 저장합니다.")

client = None
if st.session_state.get("api_key"):
    try:
        client = OpenAI(api_key=st.session_state.get("api_key"))
    except Exception as e:
        st.error(f"❌ OpenAI 클라이언트 생성 중 오류: {str(e)}")
else:
    st.warning("❗️ OpenAI API 키를 사이드바에서 입력해주세요.")

if "history" not in st.session_state:
    st.session_state.history = [{"role": "system", "content": SYSTEM_PROMPT}]
if "messages" not in st.session_state:
    st.session_state.messages = []
if not isinstance(st.session_state.messages, list):
    st.session_state.messages = []
st.session_state.messages = [
    m for m in st.session_state.messages 
    if m.get("content", "").strip() and m.get("role") in ["user", "assistant"]
]


for m in st.session_state.messages:
    content = m.get("content", "")
    if content and content.strip():
        with st.chat_message(m["role"]):
            if m["role"] == "assistant" and "progress_steps" in m:
                render_progress_steps(m["progress_steps"])
            st.markdown(content)


if "processing" not in st.session_state:
    st.session_state.processing = False

if prompt := st.chat_input("웹에서 무엇을 찾아볼까요?", disabled=st.session_state.processing):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.processing = True
    st.rerun()

if st.session_state.processing and len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
    user_prompt = st.session_state.messages[-1]["content"]
    with st.chat_message("assistant"):
        if client is None:
            error_message = "❌ OpenAI API 키가 필요합니다. 사이드바에서 API 키를 입력해주세요."
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            st.session_state.processing = False
            st.rerun()
        else:
            progress_placeholder = st.empty()
            try:
                # API 호출 시 temperature 전달 추가
                st.session_state.history, answer = research_assistant_streaming(
                    client, user_prompt, st.session_state.history, progress_placeholder, model_name
                )
                
                final_message = {
                    "role": "assistant", 
                    "content": answer, 
                    "progress_steps": st.session_state.get("progress_steps", [])
                }
                st.session_state.messages.append(final_message)
                
                st.session_state.processing = False
                st.session_state.progress_steps = []
                st.rerun()

            except Exception as e:
                progress_placeholder.empty()
                error_message = f"❌ 연구 중 오류가 발생했습니다: {str(e)}\n\n💡 API 키와 모델 설정을 확인해주세요."
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                st.session_state.processing = False
                st.rerun()


if st.sidebar.button("🗑️ 대화 초기화", disabled=st.session_state.get("processing", False)):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.history = [{"role": "system", "content": SYSTEM_PROMPT}]
    st.session_state.messages = []
    st.session_state.progress_steps = []
    st.success("✅ 대화가 초기화되었습니다!")
    st.rerun()
