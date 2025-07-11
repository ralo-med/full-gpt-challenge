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
from utils import setup_sidebar, validate_api_key, save_settings_to_session, create_llm

st.set_page_config(
    page_title="Streamlit is 🔥",
    page_icon="🔥",
    layout="wide",
)

# API 키가 설정되었는지 확인 (페이지 제일 위에 표시)
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OpenAI API key is not set! Please set API key in sidebar.")

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

# 사이드바에서 파일 업로드
with st.sidebar:
    st.write("파일 업로드")
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
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
    
    # 공통 사이드바 설정
    api_key, model_name, temperature = setup_sidebar()

    # API 키가 있을 때만 설정을 세션 상태에 저장
    if api_key:
        save_settings_to_session(api_key, model_name, temperature)

# API 키가 있을 때만 LLM 초기화 및 메모리 설정
if os.getenv("OPENAI_API_KEY"):
    # 실제 LLM 초기화
    llm = create_llm(model_name, temperature, [ChatCallbackHandler()])

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
    
    # 캐시 디렉토리 생성
    cache_dir = "./.cache/files"
    os.makedirs(cache_dir, exist_ok=True)
    
    file_path = f"{cache_dir}/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n", chunk_size=600, chunk_overlap=100
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

def embed_and_retrieve(docs, file):
    # 임베딩 캐시 디렉토리 생성
    embedding_cache_dir = f"./.cache/embeddings/{file.name}"
    os.makedirs(embedding_cache_dir, exist_ok=True)
    
    cache_dir = LocalFileStore(embedding_cache_dir)
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
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("API 키를 설정해주세요.")
    else:
        with st.spinner("📄 문서를 처리하고 있습니다..."):
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

