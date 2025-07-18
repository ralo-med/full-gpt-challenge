import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
import os
from langchain.document_loaders import UnstructuredFileLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema.output_parser import StrOutputParser
import json
from utils import setup_sidebar, save_settings_to_session


# -------------------- 1. 모든 함수 및 클래스 정의 --------------------

class JsonOutputParser(StrOutputParser):
    def parse(self, text: str):
        text = text.replace("```json", "").replace("```", "")
        return json.loads(text)


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


@st.cache_data(show_spinner=False)
def split_file(file):
    file_content = file.read()
    cache_dir = "./.cache/quiz_files"
    os.makedirs(cache_dir, exist_ok=True)
    file_path = f"{cache_dir}/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n", chunk_size=600, chunk_overlap=100
    )
    loader = UnstructuredFileLoader(file_path, mode="elements")
    docs = loader.load()
    docs = splitter.split_documents(docs)
    return docs


@st.cache_data(show_spinner=False)
def run_quiz(llm, _docs, topic, difficulty="easy"):
    """Function calling을 사용하여 퀴즈를 생성합니다."""
    
    quiz_schema = {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {"type": "string"},
                                    "correct": {"type": "boolean"},
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    }
    
    difficulty_prompts = {
        "easy": "쉬운 난이도로 기본적인 사실과 개념을 묻는 질문을 만들어주세요. 명확하고 직관적인 답변을 포함해주세요.",
        "hard": "어려운 난이도로 심화된 분석과 추론이 필요한 질문을 만들어주세요. 세부사항과 복잡한 개념을 다루는 질문을 포함해주세요.",
    }
    difficulty_instruction = difficulty_prompts.get(difficulty, difficulty_prompts["easy"])
    
    quiz_generation_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
        You are a helpful assistant that is role playing as a teacher.
        Based ONLY on the following context, create 10 questions to test the user's knowledge about the text.
        Each question should have exactly 4 answers, with three incorrect answers and one correct answer.
        IMPORTANT: Randomize the position of correct answers. Do not always put the correct answer in the first or second position. 
        Distribute correct answers evenly across all 4 positions (1st, 2nd, 3rd, 4th) throughout the quiz.
        {difficulty_instruction}
        Make sure the questions are diverse and test different aspects of the content.
        Questions should be clear, concise, and appropriate for the {difficulty} difficulty level.
        Context: {{context}}
        """,
            ),
            ("user", f"Generate a {difficulty} difficulty quiz based on the provided context."),
        ]
    )
    
    output_parser = JsonOutputParser()

    llm_with_functions = llm.bind(
        functions=[
            {
                "name": "generate_quiz",
                "description": f"Generate a {difficulty} difficulty quiz with questions and answers based on the provided context",
                "parameters": quiz_schema,
            }
        ],
        function_call={"name": "generate_quiz"},
    )
    
    chain = {"context": format_docs} | quiz_generation_prompt | llm_with_functions
    
    try:
        result = chain.invoke(docs)
        if hasattr(result, "additional_kwargs") and "function_call" in result.additional_kwargs:
            function_call = result.additional_kwargs["function_call"]
            if function_call and function_call.get("name") == "generate_quiz":
                arguments = json.loads(function_call.get("arguments", "{}"))
                return arguments
        try:
            return json.loads(result.content)
        except (json.JSONDecodeError, AttributeError):
            st.error("퀴즈 생성 중 모델의 응답을 파싱하는 데 실패했습니다. 일반 텍스트로 반환된 내용일 수 있습니다.")
            return {"questions": []}
    except Exception as e:
        st.error(f"퀴즈 생성 중 오류 발생: {e}")
        return {"questions": []}


def run_wikipedia_quiz(topic):
    retrieval = WikipediaRetriever(top_k_results=5, search_kwargs={"srsearch": topic})
    docs = retrieval.get_relevant_documents(topic)
    return docs


# -------------------- 2. 페이지 설정 및 사이드바 --------------------

st.set_page_config(page_title="Quiz Generator", page_icon="❓")
st.title("Quiz Generator")

with st.sidebar:
    api_key, model_name, temperature = setup_sidebar()
    if api_key:
        save_settings_to_session(api_key, model_name, temperature)
        os.environ["OPENAI_API_KEY"] = api_key

if not api_key:
    st.info("Please enter your OpenAI API key to proceed.")
    st.stop()

llm = ChatOpenAI(model_name=model_name, temperature=temperature)

st.write("This is a quiz application built with Streamlit and OpenAI.")
st.divider()

# -------------------- 3. 메인 UI 및 로직 --------------------

st.write("### 퀴즈 소스 선택")

docs = None
topic_name = None
choice = st.selectbox("퀴즈를 생성할 소스를 선택하세요", ("File", "Wikipedia Article"))
st.write("")  # 드롭다운 아래에 패딩 추가

if choice == "File":
    file = st.file_uploader("파일 업로드", type=["pdf", "txt", "docx"])
    if file:
        with st.spinner("파일을 처리 중입니다..."):
            docs = split_file(file)
        topic_name = file.name
        st.success("파일이 성공적으로 처리되었습니다!")
elif choice == "Wikipedia Article":
    topic = st.text_input("위키피디아 주제 입력")
    if topic:
        with st.spinner("Wikipedia에서 문서를 검색 중입니다..."):
            retrieved_docs = run_wikipedia_quiz(topic)
        if retrieved_docs:
            docs = retrieved_docs
            topic_name = topic
            st.success("완료! 아래에서 퀴즈를 생성하세요.")
        else:
            st.error("해당 주제에 대한 Wikipedia 문서를 찾을 수 없습니다. 다른 주제를 시도해주세요.")


if not docs:
    st.markdown(
        """Welcome to QuizGPT.           
I will make a quiz from Wikipedia Article or Documents.              
Get started by selecting a topic.
"""
    )
else:
    st.divider()
    difficulty = st.radio(
        "난이도",
        ["easy", "hard"],
        format_func=lambda x: {"easy": "쉬움", "hard": "어려움"}[x],
        index=0,
        horizontal=True,
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("Generate Quiz", type="secondary"):
            with st.spinner("퀴즈를 생성하고 있습니다..."):
                st.session_state.quiz_result = run_quiz(
                    llm, docs, topic_name, difficulty
                )
    with col2:
        if st.button("New Quiz", type="secondary"):
            with st.spinner("새로운 퀴즈를 생성하고 있습니다..."):
                run_quiz.clear()
                st.session_state.quiz_result = run_quiz(
                    llm, docs, topic_name, difficulty
                )
            st.rerun()

    if "quiz_result" in st.session_state and st.session_state.quiz_result.get("questions"):
        st.divider()
        st.write("### 퀴즈")
        result = st.session_state.quiz_result
        with st.form("quiz_form"):
            for i, question in enumerate(result["questions"]):
                st.write(f"**질문 {i+1}:** {question['question']}")
                value = st.radio(
                    "정답을 선택하세요.",
                    [answer["answer"] for answer in question["answers"]],
                    key=f"question_{i}",
                    index=None,
                    label_visibility="collapsed",
                )
            if st.form_submit_button("퀴즈 제출"):
                correct_count = 0
                for i, q in enumerate(result["questions"]):
                    user_answer = st.session_state.get(f"question_{i}")
                    correct_answer = next(
                        (a["answer"] for a in q["answers"] if a["correct"]), None
                    )
                    if user_answer == correct_answer:
                        correct_count += 1
                total_questions = len(result["questions"])
                st.success(f"**최종 결과: {correct_count}/{total_questions} 정답**")
                if correct_count == total_questions:
                    st.balloons()
    elif "quiz_result" in st.session_state:
        st.error("퀴즈 생성에 실패했습니다. 다시 시도해주세요.")









