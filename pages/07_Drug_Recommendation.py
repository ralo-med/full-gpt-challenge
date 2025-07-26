import time
import os
import numpy as np
import pandas as pd
import streamlit as st
import faiss
import google.generativeai as genai
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from utils import setup_sidebar, save_settings_to_session

# 환경설정
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

st.set_page_config(page_title="약물 식별 챗봇", page_icon="🔍")
st.title("🔍 약물 식별 챗봇")

with st.sidebar:
    api_key, model_name, temperature = setup_sidebar()
    if api_key and api_key.strip():
        save_settings_to_session(api_key, model_name, temperature)
        os.environ["OPENAI_API_KEY"] = api_key
    
    st.divider()
    
    st.write("Google API 키")
    google_api_key = st.text_input(
        "Google API Key", 
        type="password",
        value=os.environ.get("GOOGLE_API_KEY", "")
    )
    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key

if not (api_key and api_key.strip()):
    st.info("OpenAI API 키를 입력해주세요.")
    st.stop()

if not google_api_key:
    st.info("Google API 키를 입력해주세요.")
    st.stop()

# API 설정
genai.configure(api_key=google_api_key)
MODEL = "gemini-embedding-001"

# 약물 데이터 로드 함수
@st.cache_data
def load_drug_data():
    try:
        drug_texts = pd.read_csv("pill_embed_texts.csv")
        embeddings = np.load("pill_embeddings.npy")
        index = faiss.read_index("pill.index")
        return drug_texts, embeddings, index
    except Exception as e:
        st.error(f"약물 데이터를 로드할 수 없습니다: {e}")
        return None, None, None

# 사용자 질문을 임베딩하는 함수
def embed_query(query):
    try:
        res = genai.embed_content(model=MODEL, content=[query])
        return np.array(res['embedding'], dtype="float32").reshape(1, -1)
    except Exception as e:
        st.error(f"질문 임베딩 중 오류 발생: {e}")
        return None

# 약물 식별 검색 함수
def search_similar_drugs(query_embedding, index, drug_texts, top_k=5):
    try:
        scores, indices = index.search(query_embedding, top_k)
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(drug_texts):
                drug_info = drug_texts.iloc[idx]['embed_text']
                results.append({
                    'rank': i + 1,
                    'score': float(score),
                    'info': drug_info
                })
        return results
    except Exception as e:
        st.error(f"약물 검색 중 오류 발생: {e}")
        return []

# JSON 파싱 함수
def parse_pill_json(json_str):
    try:
        import json
        return json.loads(json_str)
    except Exception as e:
        return None

# 단일 약물 검색 함수
def search_single_pill(pill_data, hint=""):
    # 검색 텍스트 생성
    search_parts = []
    if 'dosage_form' in pill_data:
        search_parts.append(f"제형: {pill_data['dosage_form']}")
    if 'shape' in pill_data:
        search_parts.append(f"모양: {pill_data['shape']}")
    if 'color' in pill_data:
        search_parts.append(f"색깔: {pill_data['color']}")
    if 'score_line' in pill_data:
        search_parts.append(f"선: {pill_data['score_line']}")
    
    search_text = ", ".join(search_parts)
    if hint:
        search_text += f" | 힌트: {hint}"
    
    return search_text

# 메인 UI
st.markdown("## 약물 JSON 정보를 입력하세요")

# JSON 입력
json_input = st.text_area(
    "약물 JSON:",
    placeholder="""{
  "pill_count": 3,
  "pills": [
    {
      "dosage_form": "정제",
      "shape": "원형",
      "color": "하양",
      "score_line": "없음"
    },
    {
      "dosage_form": "정제",
      "shape": "장방형",
      "color": "하양",
      "score_line": "없음"
    },
    {
      "dosage_form": "정제",
      "shape": "원형",
      "color": "분홍",
      "score_line": "없음"
    }
  ],
  "confidence": 0.9
}""",
    height=300
)

# 약물 데이터 로드
drug_texts, embeddings, index = load_drug_data()

if drug_texts is not None and index is not None:
    st.success(f"💊 {len(drug_texts)}개의 약물 정보 로드 완료")
    
    # JSON 파싱
    if json_input.strip():
        pill_data = parse_pill_json(json_input)
        
        if pill_data and 'pills' in pill_data:
            pills = pill_data['pills']
            st.markdown(f"## 약물 {len(pills)}개 감지됨")
            
            # 각 약물별 입력 및 검색
            for i, pill in enumerate(pills):
                st.markdown(f"### 약물 {i+1}")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write("**특성:**")
                    st.write(f"제형: {pill.get('dosage_form', '미상')}")
                    st.write(f"모양: {pill.get('shape', '미상')}")
                    st.write(f"색깔: {pill.get('color', '미상')}")
                    st.write(f"선: {pill.get('score_line', '미상')}")
                
                with col2:
                    hint = st.text_input(
                        f"약물 {i+1} 힌트:",
                        placeholder="예: 타이레놀, 소화제 등",
                        key=f"hint_{i}"
                    )
                    
                    if st.button(f"약물 {i+1} 검색", key=f"search_{i}"):
                        search_text = search_single_pill(pill, hint)
                        query_embedding = embed_query(search_text)
                        
                        if query_embedding is not None:
                            results = search_similar_drugs(query_embedding, index, drug_texts)
                            
                            if results:
                                st.write(f"**약물 {i+1} 검색 결과:**")
                                for j, result in enumerate(results[:3], 1):
                                    with st.expander(f"{j}. 유사도: {result['score']:.3f}"):
                                        st.write(result['info'])
                            else:
                                st.warning("검색 결과가 없습니다.")
                        else:
                            st.error("검색 중 오류가 발생했습니다.")
                
                st.divider()
        else:
            if json_input.strip():
                st.error("올바른 JSON 형식이 아닙니다.")
else:
    st.error("약물 데이터 로드 실패")
    st.write("필요 파일: pill_embed_texts.csv, pill_embeddings.npy, pill.index") 
