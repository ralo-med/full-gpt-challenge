#!/usr/bin/env python3
"""
Drug Classifier v2 – GPT‑4.1‑nano + Streamlit
추가 속성: dosage_form, shape, color, score_line
비용 최소화를 위해:
  • 긴 변 1024 px 이하 리사이즈
  • JPEG 품질 85
  • OpenAI vision 옵션 detail:"low" 사용 (≈ 85 토큰/장)
"""
import os, re, json, asyncio, base64
from io import BytesIO
from typing import Dict, Any, List

import streamlit as st
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from utils import setup_sidebar, save_settings_to_session

# LangSmith 설정 (기본 활성화)
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = "Drug-Classifier"

# ────────────────────────────── 상수 ──────────────────────────────
MAX_SIZE = 1024          # px
JPEG_QUALITY = 85        # %
SHAPES = [
    "타원형","장방형","반원형","삼각형","사각형",
    "마름모형","오각형","육각형","팔각형","기타"
]
COLORS = [
    "하양","노랑","주황","분홍","빨강","갈색","연두","초록",
    "청록","파랑","남색","자주","보라","회색","검정","투명","기타"
]
SCORES = ["없음","(+)형","(-)형","기타"]

# ───────────────────────── 이미지 전처리 ──────────────────────────
def preprocess_image(img_bytes: bytes) -> bytes:
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_SIZE:
        ratio = MAX_SIZE / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, "JPEG", quality=JPEG_QUALITY, optimize=True)
    return buf.getvalue()

# ───────────────────────── PillClassifier ────────────────────────
class PillClassifier:
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4.1-mini",
            api_key=api_key,
            temperature=0.1,
          
        )

    async def classify(self, img_bytes: bytes, user_pill_count: int = None) -> Dict[str, Any]:
        img_b64 = base64.b64encode(img_bytes).decode()
        pill_count_hint = f"사진에 보이는 알약 개수는 {user_pill_count}개입니다. 반드시 'pill_count'와 'pills' 배열의 길이가 {user_pill_count}여야 합니다." if user_pill_count else ""
        prompt = f"""
각 약(알약 하나하나)을 식별하여 아래 JSON 구조로만 응답하세요.
반드시 'pill_count'와 'pills' 배열의 길이가 같아야 합니다.
{pill_count_hint}

{{
"pill_count": <총 개수>,
"pills": [
  {{
    "dosage_form": "정제|경질캡슐|연질캡슐|기타",
    "shape": "{'|'.join(SHAPES)}",
    "color": "{'|'.join(COLORS)}",
    "score_line": "{'|'.join(SCORES)}"
  }}
  ...
],
"confidence": <0~1 부동소수>
}}

규칙:
1) 각 필드는 목록 중 하나만 사용, 불확실하면 "기타".
2) 사진에 안 보이는 정보는 추측 금지.
3) 반드시 'pill_count'와 'pills' 배열의 길이가 같아야 함.
"""
        
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}",
                        "detail": "low"
                    }
                }
            ]
        )
        
        response = await self.llm.ainvoke([message])
        return self._parse_json(response.content)

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        try:
            data = json.loads(re.search(r"\{.*\}", text, re.S).group())
            pill_count = data.get("pill_count", 0)
            pills = data.get("pills", [])
            # pills 개수 보정
            if len(pills) < pill_count:
                pills += [{"dosage_form":"기타","shape":"기타","color":"기타","score_line":"기타"}] * (pill_count - len(pills))
            elif len(pills) > pill_count:
                pills = pills[:pill_count]
            data["pills"] = pills
            return data
        except Exception:
            return {"pill_count": 0, "pills": [], "confidence": 0.0}

# ───────────────────────── Streamlit UI ─────────────────────────
def display_result(data: Dict[str, Any]):
    st.success(f"✅ 총 {data['pill_count']}개 식별 (신뢰도 {data['confidence']*100:.1f}%)")
    st.json(data)
    st.download_button(
        "📥 JSON 다운로드",
        json.dumps(data, ensure_ascii=False, indent=2),
        "drug_classification.json",
        "application/json"
    )

def main():
    st.set_page_config(page_title="Drug Classifier", page_icon="💊", layout="wide")
    st.title("💊 Drug Classifier")
    st.markdown("휴대폰으로 찍은 약 사진을 업로드하면 **제형·모양·색상·분할선**을 자동 분석합니다.")

    # 사이드바: API Key 입력
    with st.sidebar:
        api_key, _, _ = setup_sidebar()
        if not api_key:
            st.stop()
        save_settings_to_session(api_key, "gpt-4.1-nano", 0.0)
        os.environ["OPENAI_API_KEY"] = api_key
        
        # LangSmith 상태 표시
        if os.getenv("LANGCHAIN_API_KEY"):
            st.success("✅ LangSmith 추적 활성화됨")
        else:
            st.warning("⚠️ LangSmith API 키가 설정되지 않았습니다")

    file = st.file_uploader("약품 이미지 업로드 (JPG/PNG)", ["jpg", "jpeg", "png"])
    if not file:
        return
    st.image(file, caption="업로드 이미지", use_container_width=True)

    # 알약 개수 입력 (선택)
    user_pill_count = st.number_input("알약 개수(선택)", min_value=1, max_value=100, step=1, value=None, format="%d")

    if st.button("💊 분석 시작", type="primary"):
        with st.spinner("AI 분석 중..."):
            try:
                img = preprocess_image(file.getvalue())
                classifier = PillClassifier(api_key)
                # classify에 pill_count 전달
                result = asyncio.run(classifier.classify(img, user_pill_count if user_pill_count else None))
                display_result(result)
            except Exception as e:
                st.error(f"❌ 오류: {e}")

if __name__ == "__main__":
    main()
