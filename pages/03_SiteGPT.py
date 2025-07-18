import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.document_transformers import Html2TextTransformer
from utils import setup_sidebar, save_settings_to_session
import os
import re


class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.message = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.message += token
        self.container.markdown(self.message)


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🤖",
)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5 (ex:4.7).

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 4.8
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0.0
                                                  
    Your turn!

    Question: {question}
"""
)

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)

html2text_transformer = Html2TextTransformer()


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", " ").replace("\xa0", " ")


def extract_score(text):
    match = re.search(r"Score:\s*([0-5](?:\.\d+)?)", text)
    return float(match.group(1)) if match else 0.0


def get_answer(inputs):
    llm = inputs["llm"]
    question = inputs["question"]
    docs = inputs["docs"]

    answer_chain = answers_prompt | llm
    answers = []

    for doc in docs:
        try:
            content = doc.page_content
            result = answer_chain.invoke({"context": content, "question": question})
            score = extract_score(result.content)
            answers.append(
                {
                    "answer": result.content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "date": doc.metadata.get("lastmod", "Unknown"),
                    "score": score,
                }
            )
        except Exception as e:
            st.warning(f"문서 처리 중 오류: {str(e)}")
            continue

    return {"question": question, "answer": answers, "llm": llm}


def choose_answer(inputs):
    llm = inputs["llm"]
    question = inputs["question"]
    answers = inputs["answer"]

    choose_chain = choose_prompt | llm
    condensed_answer = "\n\n".join(
        f"{answer['answer']}\nSource: {answer['source']}\nDate: {answer['date']}\nScore: {answer['score']}\n\n"
        for answer in answers
    )
    return choose_chain.invoke({"answers": condensed_answer, "question": question})


@st.cache_resource(show_spinner="Loading sitemap...", ttl=3600)
def load_sitemap_docs(url):
    """사이트맵을 로드하고 문서를 분할합니다. API 키가 필요하지 않습니다."""
    import urllib3
    import requests
    from requests.adapters import HTTPAdapter

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    with requests.Session() as session:
        session.verify = False
        adapter = HTTPAdapter()
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500, chunk_overlap=100
        )

        loader = SitemapLoader(
            url,
            filter_urls=[
                r"^(.*\/ai-gateway\/).*",
                r"^(.*\/vectorize\/).*",
                r"^(.*\/workers-ai\/).*",
            ],
            parsing_function=parse_page,
            session=session,
        )
        loader.requests_per_second = 10  # 속도 조절
        docs = loader.load()

    split_docs = text_splitter.split_documents(docs)
    return split_docs


def create_retriever(docs):
    """분할된 문서로부터 벡터 스토어와 리트리버를 생성합니다. API 키가 필요합니다."""
    try:
        vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
        return vectorstore.as_retriever()
    except Exception as e:
        if "api_key" in str(e).lower() or "openai_api_key" in str(e).lower():
            st.error("❌ OpenAI API 키가 필요합니다. 사이드바에서 API 키를 입력해주세요.")
            st.info("💡 사이트맵은 로드되었지만, 임베딩 생성을 위해 API 키가 필요합니다.")
        else:
            st.error(f"❌ 임베딩 생성 중 오류가 발생했습니다: {str(e)}")
        return None


st.title("SiteGPT")
st.write("Cloudflare AI 제품 문서 질의응답")


st.markdown(
    """
이 시스템은 Cloudflare의 AI Gateway, Vectorize, Workers AI 제품에 대한 질문을 답변할 수 있습니다:
- **AI Gateway**: AI 모델 API 통합 및 관리
- **Vectorize**: 벡터 데이터베이스 및 임베딩 서비스  
- **Workers AI**: 서버리스 AI 추론 서비스

Cloudflare 공식 문서가 자동으로 로드됩니다.
"""
)

with st.sidebar:
    api_key, model_name, temperature = setup_sidebar()
    if api_key:
        save_settings_to_session(api_key, model_name, temperature)
        os.environ["OPENAI_API_KEY"] = api_key

if not api_key:
    st.info("Please enter your OpenAI API key to proceed.")
    st.stop()

llm = ChatOpenAI(model_name=model_name, temperature=temperature)
url = "https://developers.cloudflare.com/sitemap.xml"

if url:
    if url.endswith(".xml"):
        if st.button("🔄 다시 로드", key="reload_button"):
            st.cache_resource.clear()
            st.success("캐시가 지워졌습니다. 페이지를 새로고침하세요.")

        try:
            # 1단계: 사이트맵 로딩 (API 키 불필요)
            docs = load_sitemap_docs(url)
            st.success("✅ 사이트맵 로딩 완료!")

            # 2단계: 임베딩 생성 (API 키 필요)
            retriever = create_retriever(docs)

            if retriever:
                query = st.text_input("질문을 입력하세요:", key="query")
                if query:
                    if llm is None:
                        st.error("API 키를 설정해주세요.")
                    else:
                        # 문서 검색 - 모든 관련 문서 검색
                        retrieved_docs = retriever.invoke(query)

                        chain = (
                            {
                                "docs": lambda x: retrieved_docs,
                                "question": RunnablePassthrough(),
                                "llm": lambda x: llm,
                            }
                            | RunnableLambda(get_answer)
                            | RunnableLambda(choose_answer)
                        )
                        result = chain.invoke(query)
                        st.write("**답변:**")
                        st.write(result.content)
            else:
                st.warning("⚠️ 임베딩이 생성되지 않아 질의응답을 사용할 수 없습니다.")
                st.info("💡 사이드바에서 OpenAI API 키를 입력한 후 페이지를 새로고침하세요.")

        except Exception as e:
            st.error(f"❌ 사이트맵 로딩 중 오류가 발생했습니다: {str(e)}")
            st.info("💡 팁: 네트워크 연결을 확인하거나 잠시 후 다시 시도해보세요.")
    else:
        st.error("Please enter a Sitemap URL.")
else:
    st.markdown(
        """# Cloudflare AI 제품 문서
                 
이 시스템은 Cloudflare의 AI Gateway, Vectorize, Workers AI 제품에 대한 질문을 답변할 수 있습니다:
- **AI Gateway**: AI 모델 API 통합 및 관리
- **Vectorize**: 벡터 데이터베이스 및 임베딩 서비스  
- **Workers AI**: 서버리스 AI 추론 서비스

Cloudflare 공식 문서가 로드되어 있습니다.
                 
"""
    )





