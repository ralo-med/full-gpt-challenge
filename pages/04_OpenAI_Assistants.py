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
            return f"파일 저장 완료: result.txt ({file_size} 바이트)\n\n🎯 [최종 연구 요약]\n{content}"
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
    "You are an investigative research assistant. You MUST follow this EXACT 4-step process:\n"
    "1. ALWAYS start with wiki_search to get basic information\n"
    "2. ALWAYS use ddg_search to find additional web sources\n" 
    "3. ALWAYS use fetch_webpage to extract content from at least one web page\n"
    "4. ALWAYS finish with save_to_file to save all findings\n\n"
    "CRITICAL: You MUST call save_to_file as the final step before providing your answer.\n"
    "For save_to_file, combine ALL the information you gathered from the previous steps into a comprehensive summary.\n"
    "Include the original question, findings from Wikipedia, web search results, and webpage content.\n"
    "You cannot skip any of these steps. Each tool must be used at least once in this order.\n"
    "DO NOT provide your final answer until you have called save_to_file.\n"
    "After save_to_file, provide a comprehensive final answer.\n\n"
    "MANDATORY: You are NOT allowed to finish without calling save_to_file. This is a requirement, not a suggestion.\n"
    "If you try to provide a final answer without calling save_to_file, you will be forced to call it first.\n"
    "The save_to_file function is MANDATORY and must be called before any final response."
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
    history.append({"role": "user", "content": user_msg})

    st.session_state.progress_steps = []
    progress_steps = st.session_state.progress_steps
    required_tools = {"wiki_search": False, "ddg_search": False, "fetch_webpage": False}
    save_completed = False
    final_summary = ""

    progress_steps.append("🤔 AI assistant 분석 시작...")
    render_progress_steps(progress_steps, progress_placeholder)

    resp = safe_create_responses(
        client,
        model=model_name,
        input=history,
        tools=tools,
    )

    step_count = 0
    while resp.output and resp.output[0].type == "function_call" and not save_completed:
        step_count += 1
        call = resp.output[0]
        tool_descriptions = {
            "wiki_search": "📚 위키백과 검색",
            "ddg_search": "🔍 DuckDuckGo 웹 검색", 
            "fetch_webpage": "🌐 웹페이지 내용 추출",
            "save_to_file": "💾 결과를 파일로 저장"
        }
        tool_name = tool_descriptions.get(call.name, '도구 실행 중')
        progress_steps.append(f"🔍 **단계 {step_count}** {tool_name} 중...")
        result = functions_map[call.name](json.loads(call.arguments))
        if call.name in required_tools:
            required_tools[call.name] = True
        if call.name == "save_to_file":
            save_completed = True
            try:
                args = json.loads(call.arguments)
                final_summary = args.get("content", "")
            except Exception:
                final_summary = ""

        progress_steps.append(f"✅ **단계 {step_count} 완료**: {tool_name}")
        progress_steps.append({
            "type": "result_dropdown",
            "step": step_count,
            "tool": call.name,
            "tool_name": tool_name,
            "result": result
        })
        render_progress_steps(progress_steps, progress_placeholder)
        
        history.extend(
            [
                call,
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": result,
                },
            ]
        )
        
        if save_completed:
            progress_steps.append("🎉 **연구 완료**: 파일 저장이 완료되었습니다!")
            render_progress_steps(progress_steps, progress_placeholder)
            break
        if not save_completed:
            progress_steps.append("🤔 AI assistant 도구 선택 중...")
            render_progress_steps(progress_steps, progress_placeholder)
        resp = safe_create_responses(
            client,
            model=model_name,
            input=history,
            tools=tools,
        )

    # 최종 답변: resp.output_text(assistant 답변) 또는 summary(혹시 output_text가 비어있을 때)
    final_answer = resp.output_text.strip() if resp.output_text and resp.output_text.strip() else f"🎯 [최종 연구 요약]\n{final_summary}"
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
- ⭐️ 저장이 이루어 지지 않을시 분석이 완료되지 않은것이니 계속 대화를 이어가면 됩니다!

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
