import streamlit as st
from langchain.document_loaders import SitemapLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser, output_parser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable import RunnableLambda
import re
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.1)

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

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🌐",
    layout="wide",
)

html2text_transformer = Html2TextTransformer()

def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    elif footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", " ").replace("\xa0", " ")


def extract_score(text):
    match = re.search(r"Score:\s*([0-5](?:\.\d+)?)", text)
    return float(match.group(1)) if match else 0.0

def get_answer(inputs):
    question = inputs["question"]
    docs = inputs["docs"]

    answer_chain = answers_prompt | llm
    answers = []
    

    for doc in docs:
        try:
       
            content = doc.page_content
            result = answer_chain.invoke({"context": content, "question": question})
            score = extract_score(result.content)
            answers.append({
                "answer": result.content,
                "source": doc.metadata.get("source", "Unknown"),
                "date": doc.metadata.get("lastmod", "Unknown"),
                "score": score
            })
        except Exception as e:
            st.warning(f"문서 처리 중 오류: {str(e)}")
            continue
    
    return {"question": question, "answer": answers}

def choose_answer(inputs):
    question = inputs['question']
    answers = inputs['answer']
    choose_chain = choose_prompt | llm
    condensed_answer = '\n\n'.join(
        f"{answer['answer']}\nSource: {answer['source']}\nDate: {answer['date']}\nScore: {answer['score']}\n\n"
        for answer in answers
    )
    return choose_chain.invoke({'answers': condensed_answer, 'question': question})

@st.cache_resource(show_spinner="Loading sitemap...", ttl=3600)
def load_sitemap(url):
    import urllib3
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # SSL 검증을 완전히 비활성화하는 세션 생성
    session = requests.Session()
    session.verify = False
    
    # 어댑터 설정
    adapter = HTTPAdapter()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=100)

    
    loader = SitemapLoader(
        url,
        filter_urls=[
          r"^(.*\/ai-gateway\/).*",
          r"^(.*\/vectorize\/).*", 
          r"^(.*\/workers-ai\/).*"
        ],
        parsing_function=parse_page,
        session=session
    )
    loader.requests_per_second = 10  # 속도 조절
    docs = loader.load()
    split_docs = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(split_docs, OpenAIEmbeddings())
    return vectorstore.as_retriever()

st.title("SiteGPT")
st.write("Cloudflare AI 제품 문서 질의응답")

with st.sidebar:
    st.write("Cloudflare 공식 문서")
    url = "https://developers.cloudflare.com/sitemap-0.xml"
    st.info(f"📄 사이트맵: {url}")

if url:
    if url.endswith(".xml"):
        if st.button("🔄 다시 로드", key="reload_button"):
            st.cache_data.clear()
        
        try:
            retriever = load_sitemap(url)
            query = st.text_input("질문을 입력하세요:", key="query")
            if query:       
                # 문서 검색 - 모든 관련 문서 검색
                docs = retriever.invoke(query)
                
                chain = (
                    {
                        "docs": lambda x: docs,
                        "question": RunnablePassthrough(),
                    }
                    | RunnableLambda(get_answer)
                    | RunnableLambda(choose_answer)
                )
                result = chain.invoke({"question": query})
                st.write("**답변:**")
                st.write(result.content)
        except Exception as e:
            st.error(f"❌ 사이트맵 로딩 중 오류가 발생했습니다: {str(e)}")
    else:
        with st.sidebar:
            st.error("Please enter a Sitemap URL.")
else:
    st.markdown("""# Cloudflare AI 제품 문서
                 
이 시스템은 Cloudflare의 AI Gateway, Vectorize, Workers AI 제품에 대한 질문을 답변할 수 있습니다:
- **AI Gateway**: AI 모델 API 통합 및 관리
- **Vectorize**: 벡터 데이터베이스 및 임베딩 서비스  
- **Workers AI**: 서버리스 AI 추론 서비스

Cloudflare 공식 문서가 로드되어 있습니다.
                 
""")





