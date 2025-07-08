import time
import os
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder

st.set_page_config(
    page_title="Streamlit is 🔥",
    page_icon="🔥",
    layout="wide",
)

class ChatCallbackHandler(BaseCallbackHandler):

    def __init__(self):
        self.message = ""
        self.message_box = None

    def on_llm_start(self, *args, **kwargs):
        try:
            self.message = ""
            self.message_box = st.empty()
        except Exception:
            pass
        
    def on_llm_end(self, *args, **kwargs):
        try:
            save_message(self.message, "ai")
        except Exception:
            pass

    def on_llm_new_token(self, token: str, **kwargs):
        try:
            self.message += token
            if self.message_box is not None:
                self.message_box.markdown(self.message)
        except Exception:
            pass

# 실제 개발 환경인지 확인 (환경변수에 DEV_MODE가 설정되어 있는지 확인)
is_actual_dev_mode = os.getenv("DEV_MODE", "false") == "true"

# 사이드바에서 모델과 API 키 설정
with st.sidebar:
    st.write("설정")
    
    # 실제 개발 모드일 때만 환경변수 사용
    if is_actual_dev_mode:
        st.success("개발 모드: 환경변수 API 키 사용중")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("❌ 개발 모드에서 OPENAI_API_KEY 환경변수가 설정되지 않았습니다!")
            st.stop()
    else:
        # 배포 모드: 사용자 입력 받기
        st.info("API 키를 입력해주세요")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="OpenAI API 키를 입력하세요"
        )

    st.divider()
    
    st.write("파일 업로드")
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )
    
    st.divider()
    
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
    
   
    
    st.write("대화 관리")
    if st.button("대화 기록 초기화"):
        try:
            if "memory" in st.session_state:
                st.session_state.memory.clear()
            st.session_state["messages"] = []
            st.success("대화 기록이 초기화되었습니다!")
        except Exception:
            pass
    
    st.divider()
    
    st.write("소스코드")
    st.markdown("[GitHub Repository](https://github.com/your-username/full-gpt-challenge)")
    st.markdown("[Streamlit App Code](https://github.com/your-username/full-gpt-challenge/blob/main/home.py)")

# API 키가 설정되었는지 확인
if api_key:
    # API 키가 입력되었으면 환경변수에 설정
    os.environ["OPENAI_API_KEY"] = api_key
    llm = ChatOpenAI(
        model=model_name, 
        temperature=temperature, 
        streaming=True, 
        callbacks=[ChatCallbackHandler()]
    )
else:
    # API 키가 비어있으면 환경변수에서 삭제
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]
    
    # 간단한 웰컴 메시지 표시
    st.title("🔥 DocumentGPT")
    
    st.markdown(
        """
        Welcome! 🔥
                
        Use this chatbot to ask questions to an AI about your files!
        
        **Please enter your OpenAI API key in the sidebar to get started!**
        """
    )
    
    st.stop()

# 메모리를 session_state에 저장
if "memory" not in st.session_state:
    try:
        st.session_state.memory = ConversationBufferMemory(
            llm=llm,
            max_token_limit=120,
            return_messages=True,
            memory_key="history"
        )
    except Exception:
        pass

# messages도 session_state에 저장
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# memory 변수 안전하게 할당
try:
    memory = st.session_state.memory
except Exception:
    memory = ConversationBufferMemory(
        llm=llm,
        max_token_limit=120,
        return_messages=True,
        memory_key="history"
    )
    st.session_state.memory = memory

@st.cache_data(show_spinner="Embedding file..." )
def load_and_split(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n", chunk_size=600, chunk_overlap=100
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

def embed_and_retrieve(docs, file):
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    return vectorstore.as_retriever()

st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload a file on the sidebar to get started!
"""
)

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def save_message(message, role):
    st.session_state["messages"].append({"role": role, "message": message})

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that can answer questions about documents and previous conversations. 

Use both the document context and conversation history to provide accurate answers. If the user asks about something we discussed before, refer to that information.
If you don't know the answer just say you don't know, don't make it up:
Document context: {context}
"""),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{question}"),
])

def docs_to_context(docs):
    return "\n\n".join([doc.page_content for doc in docs])

if file:
    retriever = embed_and_retrieve(load_and_split(file), file)

    if retriever:
        st.success("🎉 파일이 성공적으로 처리되었습니다!")
        send_message("I'm ready to answer your questions!", "ai", save=False)
        paint_history()

        def ask(question):
            try:
                memory_vars = st.session_state.memory.load_memory_variables({})
                history = memory_vars.get("history", [])
            except Exception:
                history = []
            
            docs = retriever.invoke(question)
            context = docs_to_context(docs)
            
            result = prompt.invoke({
                "question": question, 
                "context": context,
                "history": history
            })
            
            response = llm.invoke(result)
            
            try:
                st.session_state.memory.save_context(
                    {"input": question}, 
                    {"output": response.content}
                )
            except Exception:
                pass
            
            return response.content

        message = st.chat_input("Ask me anything!")
        if message:
            send_message(message, "human")
            with st.chat_message("ai"):
                response = ask(message)
    else:
        st.warning("파일 처리를 완료할 수 없습니다. 위의 오류 메시지를 확인해주세요.")
else:
    st.session_state["messages"] = []

