import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
import os
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser, output_parser
import json
from utils import setup_sidebar, validate_api_key, save_settings_to_session, create_llm

# Function calling을 위한 스키마 정의
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
                                "correct": {"type": "boolean"}
                            },
                            "required": ["answer", "correct"]
                        }
                    }
                },
                "required": ["question", "answers"]
            }
        }
    },
    "required": ["questions"]
}

class JsonOutputParser(BaseOutputParser):
    def parse(self, text: str):
        text=text.replace("```json","").replace("```","")
        return json.loads(text)
    
output_parser = JsonOutputParser()

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# 초기 설정 (나중에 사이드바에서 업데이트됨)
model_name = "gpt-4.1-nano"
temperature = 0.1

llm = create_llm(model_name, temperature, [StreamingStdOutCallbackHandler()])

# Function calling을 활용한 퀴즈 생성 프롬프트
quiz_generation_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a helpful assistant that is role playing as a teacher.
    
    Based ONLY on the following context, create 10 questions to test the user's knowledge about the text.
    
    Each question should have exactly 4 answers, with three incorrect answers and one correct answer.
    
    Make sure the questions are diverse and test different aspects of the content.
    Questions should be clear, concise, and appropriate for the difficulty level of the content.
    
    Context: {context}
    """),
    ("user", "Generate a quiz based on the provided context.")
])

# Function calling을 활용한 퀴즈 생성 함수 (난이도 추가)
def generate_quiz_with_function_calling(docs, difficulty="easy"):
    """Function calling을 사용하여 퀴즈를 생성합니다."""
    try:
        # 난이도별 프롬프트 설정
        difficulty_prompts = {
            "easy": "쉬운 난이도로 기본적인 사실과 개념을 묻는 질문을 만들어주세요. 명확하고 직관적인 답변을 포함해주세요.",
            "hard": "어려운 난이도로 심화된 분석과 추론이 필요한 질문을 만들어주세요. 세부사항과 복잡한 개념을 다루는 질문을 포함해주세요."
        }
        
        difficulty_instruction = difficulty_prompts.get(difficulty, difficulty_prompts["easy"])
        
        # 난이도별 퀴즈 생성 프롬프트
        quiz_generation_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""
            You are a helpful assistant that is role playing as a teacher.
            
            Based ONLY on the following context, create 10 questions to test the user's knowledge about the text.
            
            Each question should have exactly 4 answers, with three incorrect answers and one correct answer.
            
            IMPORTANT: Randomize the position of correct answers. Do not always put the correct answer in the first or second position. 
            Distribute correct answers evenly across all 4 positions (1st, 2nd, 3rd, 4th) throughout the quiz.
            
            {difficulty_instruction}
            
            Make sure the questions are diverse and test different aspects of the content.
            Questions should be clear, concise, and appropriate for the {difficulty} difficulty level.
            
            Context: {{context}}
            """),
            ("user", f"Generate a {difficulty} difficulty quiz based on the provided context.")
        ])
        
        # LLM에 function calling 설정
        llm_with_functions = llm.bind(functions=[{
            "name": "generate_quiz",
            "description": f"Generate a {difficulty} difficulty quiz with questions and answers based on the provided context",
            "parameters": quiz_schema
        }])
        
        # 체인 생성
        chain = {"context": format_docs} | quiz_generation_prompt | llm_with_functions
        
        # 퀴즈 생성
        result = chain.invoke(docs)
        
        # Function calling 결과 파싱
        if hasattr(result, 'additional_kwargs') and 'function_call' in result.additional_kwargs:
            function_call = result.additional_kwargs['function_call']
            if function_call and function_call.get('name') == 'generate_quiz':
                arguments = json.loads(function_call.get('arguments', '{}'))
                return arguments
        
        # Fallback: 일반 텍스트 응답을 JSON으로 파싱 시도
        try:
            return json.loads(result.content)
        except:
            st.error("퀴즈 생성 중 오류가 발생했습니다.")
            return {"questions": []}
            
    except Exception as e:
        st.error(f"퀴즈 생성 중 오류가 발생했습니다: {str(e)}")
        return {"questions": []}

st.set_page_config(
    page_title="QuizGPT",
    page_icon="🤔",
    layout="wide",
)

st.title("QuizGPT")
st.write("This is a quiz application built with Streamlit and OpenAI.")

@st.cache_data(show_spinner=False)  # 스피너 제거
def split_file(file):
    if not os.getenv("OPENAI_API_KEY"):
        st.error("❌ OpenAI API key is not set! Please set API key in Home page.")
        return None
    
    os.makedirs("./.cache/quiz_files", exist_ok=True)
    os.makedirs("./.cache/quiz_embeddings", exist_ok=True)
    
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path, mode="elements")
    docs = loader.load()
    docs = splitter.split_documents(docs)
    return docs

@st.cache_data(show_spinner=False)  # 스피너 제거
def run_quiz(_docs, topic, difficulty="easy"):
    """Function calling을 사용하여 퀴즈를 생성합니다."""
    return generate_quiz_with_function_calling(_docs, difficulty)

def run_wikipedia_quiz(topic):
    retrieval = WikipediaRetriever(top_k_results=5, search_kwargs={"srsearch": topic})
    docs = retrieval.get_relevant_documents(topic)
    return docs

# 퀴즈 생성 메인 로직
with st.sidebar:
    st.write("퀴즈 소스 선택")
    docs = None
    choice = st.selectbox("퀴즈를 생성할 소스를 선택하세요",("File","Wikipedia Article"))

    if choice == "File":
        file = st.file_uploader("파일 업로드", type=["pdf","txt","docx"])
        if file:
            st.write("파일이 성공적으로 업로드되었습니다!")
            docs = split_file(file)

    if choice == "Wikipedia Article":
        topic = st.text_input("위키피디아 주제 입력")
        if topic:
            docs = run_wikipedia_quiz(topic)
            if docs:
                st.success("완료! 퀴즈 생성을 눌러주세요.")
    
    st.divider()
    
    # 공통 사이드바 설정
    api_key, model_name, temperature = setup_sidebar()

    # API 키 유효성 검사
    if validate_api_key(api_key):
        save_settings_to_session(api_key, model_name, temperature)
    else:
        st.error("❌ Please enter your OpenAI API key!")
        st.stop()

if not docs:
     st.markdown("""Welcome to QuizGQP.           
I will make a quiz from Wikipedia Article or Documents.              
Get started by selecting a topic.
                 
""")
else:
    # 난이도 설정
    difficulty = st.radio(
        "난이도",
        ["easy", "hard"],
        format_func=lambda x: {"easy": "쉬움", "hard": "어려움"}[x],
        index=0,
        horizontal=True
    )
    
    # 퀴즈 생성 버튼
    if st.button("Generate Quiz", type="secondary"):
        with st.spinner("퀴즈를 생성하고 있습니다..."):
            result = run_quiz(docs, topic if 'topic' in locals() and topic else file.name if 'file' in locals() else "unknown", difficulty)
            
            if result and "questions" in result:
                # 세션 상태에 결과 저장
                st.session_state.quiz_result = result
                st.success("퀴즈가 성공적으로 생성되었습니다!")
            else:
                st.error("퀴즈 생성에 실패했습니다. 다시 시도해주세요.")
    
    # 퀴즈 표시 (생성된 경우에만)
    if hasattr(st.session_state, 'quiz_result') and st.session_state.quiz_result:
        result = st.session_state.quiz_result
        
        with st.form("quiz_form"):
            for i, question in enumerate(result["questions"]):
                st.write(f"**질문 {i+1}:** {question['question']}")
                value = st.radio(
                    "",
                    [answer["answer"] for answer in question["answers"]],
                    key=f"question_{i}",
                    index=None
                )
                
                # 각 질문 아래에 정답 체크 표시
                if value is not None:
                    correct_answer = next((ans["answer"] for ans in question["answers"] if ans["correct"]), None)
                    if value == correct_answer:
                        st.success("✅ 정답입니다!")
                    else:
                        st.error(f"❌ 틀렸습니다!")
            
            submit = st.form_submit_button("퀴즈 제출")
            if submit:
                # 최종 결과만 표시
                correct_count = sum(1 for i, question in enumerate(result["questions"]) 
                                 if st.session_state.get(f"question_{i}") == 
                                 next((ans["answer"] for ans in question["answers"] if ans["correct"]), None))
                total_questions = len(result["questions"])
                st.write(f"**최종 결과: {correct_count}/{total_questions} 정답**")
                if correct_count == total_questions:
                    st.balloons()









