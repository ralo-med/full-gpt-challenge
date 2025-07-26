#!/usr/bin/env python3
import os, time, numpy as np, pandas as pd
from tqdm import tqdm
import google.generativeai as genai
import faiss

# .env 파일에서 GOOGLE_API_KEY를 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
MODEL  = "gemini-embedding-001"
BATCH  = 100

texts = pd.read_csv("pill_embed_texts.csv").embed_text.tolist()
vecs  = np.empty((len(texts), 3072), dtype="float32")

for i in tqdm(range(0, len(texts), BATCH), unit="req"):
    batch = texts[i:i+BATCH]
    # 'contents'가 아니라 'content'가 올바른 파라미터 이름
    res   = genai.embed_content(model=MODEL, content=batch)
    # 결과 파싱 방식도 변경됨: res['embedding']이 바로 벡터 리스트
    vecs[i:i+len(batch)] = np.array(res['embedding'], dtype="float32")
    time.sleep(0.05)   # 무료쿼터 RPM 100 보호

np.save("pill_embeddings.npy", vecs)
idx = faiss.IndexFlatIP(vecs.shape[1]); idx.add(vecs)
faiss.write_index(idx, "pill.index")
print("✅ 끝!  shape:", vecs.shape)
